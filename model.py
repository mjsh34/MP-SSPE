from util import ConfigBase, get_device, glg, get_root_dir, dtype_np, dtype_torch
from data_wrangling import totorch, rot_mat_to_vec, rot_6d_to_mat, get_model_path, \
     compute_joints, compute_foot_slides, get_converter, set_get_device_func, \
     recover_root_transform, \
     SMPL_UPPER_BODY_JOINTS, SMPL_LOWER_BODY_JOINTS, SMPL_LEG_JOINTS, SMPL_JOINT_INDICES, \
     SMPL_UPPER_EXTREMITY_JOINTS, SMPL_HEAD_JOINT, \
     ACTIONS_TOP50, IDX2ACT
from data_loading import load_torch_model, load_smpl_model
from rnvp import LinearRNVP
import action2motion as a2m
from pytorch3d_funcs import so3_relative_angle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

import os
import os.path as osp
import datetime
import time
import math
import json
from collections import defaultdict, OrderedDict, namedtuple
import random
import math
import pickle
import functools
import logging
import contextlib
import re


LOSS_TYPES = ['MSE', 'R6DNORM', 'RMATNORM', 'MPJRE_DEG',
           'MPJPE', 'MPJVE', #'MPJPE_RC', 'MPJVE_RC', 'ROOT_POS', 'ROOT_VELN'
           'VEL', 'MPJPE_REL', 'MPJVE_REL', 'VEL_REL',
           'ACC_REG', 'AVEL', 'AACC_REG', 'DAVEL',
           'RMAT_UNIT', 'RMAT_ORTHO',
           'VAEHMD_POS', 'VAEHMD_ROT',
           # Evaluation Metrics
           'JITTER_GT', 'JITTER_PRED',
           'FS_0.06_GT', 'FS_0.15_GT', 'FS_1.0_GT', 'FS_3.3_GT',
           'FS_0.06_PRED', 'FS_0.15_PRED', 'FS_1.0_PRED', 'FS_3.3_PRED']
LOSSES_SEP = '+'
is_reg = lambda lt_: lt_.endswith("_REG")
HIDDEN_STATE_TYPES = ['no_randomisation', 'randomise_all', 'sometimes_randomise']
HIDDEN_STATE_RANDOMISATION_METHODS = ['zero', 'xavier_uniform', 'xavier_normal']

TEST_RESULTS_RAW_FN = "test_results.pkl"

ExportWindow = namedtuple('ExportWindow',
        ['path', 'start_frame', 'end_frame', 'nframes_clip', 'stats'])


def get_required_convs_for_loss(loss_name, is_pred):
    if not is_pred:
        if is_reg(loss_name) or loss_name in ('RMAT_UNIT', 'RMAT_ORTHO'):
            return None

    d = {
        'R6DNORM': 'rot_6d',
        'RMATNORM': 'rot_mats',
        'MSE': 'rot_6d',
        'MPJRE_DEG': 'rot_mats',
        'MPJPE': 'joints',
        #'MPJPE_RC': 'joints+rot_mats',  # to correct root rotation
        #'ROOTJ_POS': 'joints',  # global?
        'MPJVE': 'vel',
        #'MPJVE_RC': 'vel',
        #'ROOTJ_VELN': 'vel',
        'VEL': 'vel',
        'MPJPE_REL': 'joints_rel',
        'VEL_REL': 'vel_rel',
        'MPJVE_REL': 'vel_rel',
        'AAC_REG': 'acc',
        'AVEL': 'avel',
        'AACC_REG': 'aacc',
        'DAVEL': 'davel',
        'RMAT_UNIT': 'rot_6d',
        'RMAT_ORTHO': 'rot_6d',
        'JITTER_GT': 'jitter',
        'JITTER_PRED': 'jitter',
        'VAEHMD_POS': 'joints',
        'VAEHMD_ROT': 'rot_mats_g',
    }
    if loss_name in d:
        return d[loss_name]
    elif loss_name.startswith("FS_"):
        return 'joints'
    else:
        raise ValueError("get_required_convs_for_loss - Unknown loss type: {}".format(
                    loss_name))


def get_loss_func(model_config, loss_name, apply_multiplier=True, keep_batch_dims=False):
    if loss_name in('MPJPE', 'MPJPE_REL', 'R6DNORM', 'RMATNORM', 'MPJVE', 'MPJVE_REL',
            'JITTER_GT', 'JITTER_PRED'):
        if loss_name in ('MPJPE', 'MPJPE_REL', 'MPJVE', 'MPJVE_REL'):  # positional
            rj_coeff = model_config['root_joint_pos_error_coeff']
        else:  # rotational
            rj_coeff = model_config['root_joint_rot_error_coeff']
        if loss_name.endswith('_GT'):
            self_op = 'gt'
        elif loss_name.endswith('_PRED'):
            self_op = 'pred'
        else:
            self_op = None

        if math.isclose(1.0, rj_coeff) and self_op is None:
            if keep_batch_dims:
                lf = lambda p, o: torch.linalg.vector_norm(p - o, dim=-1)
            else:
                lf = lambda p, o: torch.linalg.vector_norm(p - o, dim=-1).mean()
        else:
            def lf(p, o):
                if self_op is None:
                    norms = torch.linalg.vector_norm(p - o, dim=-1)
                elif self_op == 'gt':
                    norms = torch.linalg.vector_norm(o, dim=-1)
                else:
                    assert self_op == 'pred'
                    norms = torch.linalg.vector_norm(p, dim=-1)

                norms[..., 0] *= rj_coeff
                if keep_batch_dims:
                    return norms
                else:
                    return norms.mean()
    elif loss_name == 'MPJRE_DEG':
        RAD2DEG = 180 / math.pi
        def lf(p, o):
            d = torch.transpose(o, -1, -2) @ p
            trace = d.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
            angl = torch.acos(torch.clamp((trace - 1) / 2, 0.0, 1.0))
            angl_abs = torch.abs(angl)
            angl_deg_abs = RAD2DEG * angl_abs
            if keep_batch_dims:
                return angl_deg_abs
            else:
                return angl_deg_abs.mean()
    elif loss_name == 'RMAT_UNIT':
        if keep_batch_dims:
            lf = lambda p, o: torch.linalg.vector_norm(p, dim=-2)
        else:
            # (nb x)* nf x nj x 3 x nvecs
            lf = lambda p, o: torch.linalg.vector_norm(p, dim=-2).mean()
    elif loss_name == 'RMAT_ORTHO':
        if keep_batch_dims:
            lf = lambda p, o: torch.linalg.cross(p[..., 0], p[..., 1])
        else:
            # (nb x)* nf x nj x 3 x 2
            lf = lambda p, o: torch.linalg.cross(p[..., 0], p[..., 1]).mean()
    elif loss_name.endswith('_REG'):
        lf = lambda p, o: nn.functional.mse_loss(p, torch.zeros_like(p),
                reduction='none' if keep_batch_dims else 'mean')
    elif loss_name == 'VAEHMD_POS':
        def lf(p, o):
            if keep_batch_dims:
                raise NotImplementedError
            mseval = nn.functional.mse_loss(p, o, reduction='none')
            mseval = mseval.sum(dim=(-1, -2))
            return mseval.mean()
    elif loss_name == 'VAEHMD_ROT':
        def lf(p, o):
            if keep_batch_dims:
                raise NotImplementedError
            nj = p.shape[-3]
            assert nj == 22
            p3 = p.reshape(-1, 3, 3)
            o3 = o.reshape(-1, 3, 3)
            angles = so3_relative_angle(p3, o3)
            assert len(angles.shape) == 1
            angles = angles.reshape(-1, nj).sum(dim=-1)
            return angles.mean()
    else:
        lf = lambda p, o: nn.functional.mse_loss(p, o,
                reduction='none' if keep_batch_dims else 'mean')

    if apply_multiplier:
        multiplier = model_config['loss_multiplier_{}'.format(loss_name)]
        return lambda p, o=None: multiplier * lf(p, o)
    else:
        return lambda p, o=None: lf(p, o)


class ModelConfig(ConfigBase):
    TARGET_FRAMERATE_DEFAULT = 30 or loss_name == 'MPJVE_REL'

    def __init__(self, data=None):
        super().__init__()

        self._set_default_values()
        if data:
            self.load_from_dict(data)

        self._verify()

    def _set_default_values(self):
        self._set_default('do_correct_amass_rotation', True)
        self._set_default('checkpoint_path', None)
        self._set_default('checkpoints_save_directory', "./checkpoints")
        self._set_default('checkpoint_save_interval', 10) # checkpoint will be saved every # epochs

        self._set_default('window_size', 20)
        self._set_default('loading_window_size', 20+64)
        self._set_default('test_loading_window_size', 20+64)
        self._set_default('batch_size', 128)
        self._set_default('test_batch_size', 128)

        self._set_default('shuffle_opts', {})
        self._set_default('shuffle_opts_test', {})

        self._set_default('test_results_raw_fn', TEST_RESULTS_RAW_FN)
        self._set_default('start_evaluation_from_frame', 20)
        self._set_default('fid_nshufs', 50)

        # Input/Output Types
        self._set_default('input_global_joint_positions', False)
        self._set_default('global_joints_grounded', True)

        self._set_default('input_positions', True)
        self._set_default('input_rotations', True)
        self._set_default('input_prev_frame_pose', False)
        self._set_default('input_prev_frame_pose__use_gt', False)
        self._set_default('input_delta_translation', False)
        self._set_default('input_velocities', False)
        self._set_default('input_angular_velocities', False)
        self._set_default('input_motion_embeddings', False)

        self._set_default('input_9d_rot', False)
        self._set_default('output_9d_rot', False)

        # Overall Architecture
        self._set_default('nn_architecture', 'RNN')

        self._set_default('prefer_6d_rotation', True)
        self._set_default('beta_size', 10)
        self._set_default('target_framerate', 30)
        self._set_default('normalise_shape', True)
        self._set_default('njoints_cutoff', 22)
        self._set_default('input_fullpose', False)
        self._set_default('loss', 'MSE')
        self._set_default('loss_multiplier_MSE', 1.0)
        self._set_default('loss_multiplier_R6DNORM', 1.0)
        self._set_default('loss_multiplier_RMATNORM', 1.0)
        self._set_default('loss_multiplier_MPJRE_DEG', math.pi / 180)
        self._set_default('loss_multiplier_MPJPE', 1.0)
        self._set_default('loss_multiplier_MPJPE_REL', 1.0)
        self._set_default('loss_multiplier_MPJVE', 1.0)
        self._set_default('loss_multiplier_VEL', 1.0)
        self._set_default('loss_multiplier_MPJVE_REL', 1.0)
        self._set_default('loss_multiplier_VEL_REL', 1.0)
        self._set_default('loss_multiplier_ACC_REG', 0.2)
        self._set_default('loss_multiplier_AVEL', 1.0)
        self._set_default('loss_multiplier_AACC_REG', 0.2)
        self._set_default('loss_multiplier_DAVEL', 1.0)
        self._set_default('loss_multiplier_RMAT_UNIT', 0.1)
        self._set_default('loss_multiplier_RMAT_ORTHO', 0.1)
        self._set_default('loss_multiplier_VAEHMD_POS', 1/(0.02**2))
        self._set_default('loss_multiplier_VAEHMD_ROT', 1/(0.1**2))

        self._set_default('loss_multiplier_MOTION_EMBEDDING', 0.01)

        self._set_default('root_joint_pos_error_coeff', 1.0)
        self._set_default('root_joint_rot_error_coeff', 1.0)

        self._set_default('normalise_global_joint_positions', True)
        self._set_default('normalise_global_joint_positions_y', False)
        self._set_default('normalise_global_joint_positions_divbystd', False)

        # Motion Embedding
        # By default the "motion_embedding" properties refer to those used for training
        self._set_default('use_motion_embeddings', False)
        self._set_default('input_motion_embeddings__reduced_dim', 64) # 0 for no reduction
        # You can override this via cmd opt
        # also path for specific split will override general path
        self._set_default('preprocessed_motion_embeddings_base_path', None) 
        self._set_default('train_split_preprocessed_motion_embeddings_path', None)
        self._set_default('validation_split_preprocessed_motion_embeddings_path', None)
        self._set_default('test_split_preprocessed_motion_embeddings_path', None)

        self._set_default('motion_embedding_guidance', None)
        self._set_default('motion_embedding_evaluation', 'MotionCLIP')
        self._set_default('incomplete_motion_embeddings_policy', 'randomise')

        self._set_default('motion_embedding_train_noise_scale', 0.0)
        self._set_default('motion_embedding_sparse_input', True)
        self._set_default('motion_embedding_type', 'me')
        self._set_default('motion_embedding_model_type', 'MotionCLIP')
        self._set_default('motion_embedding_guidance_model_type', 'MotionCLIP')
        self._set_default('motion_embedding_evaluation_model_type', 'MotionCLIP')

        self._set_default('include_motion_embedder', False)
        self._set_default('input_dp', False)
        self._set_default('motion_embedder_encoder_freeze', True)
        self._set_default('motion_embedder_use_logvar', False)

        self._set_default('motion_embedding_input_method', 'concat')
        self._set_default('motion_embedding_catt_nheads', 4)
        #self._set_default('motion_embedding_catt_me_as_query', True)

        # FP input
        self._set_default('motionclip_checkpoint_path_for_guidance',
                "SMPL/MotionCLIP/checkpoints/paper_0100.pth.tar")
        # FP input
        self._set_default('motionclip_checkpoint_path_for_evaluation',
                "SMPL/MotionCLIP/checkpoints/classes_0200.pth.tar")
        # Must match checkpoint used to preprocess MEs for accurate evaluation
        self._set_default('motionclip_checkpoint_path_for_otf', None)

        # Positional Encoding
        self._set_default('positional_encoding', False)
        self._set_default('input_encoder', 'none')
        self._set_default('simple_input_encoder_layernorm', False)
        self._set_default('input_encoder_dim', 256)

        self._set_default('pe_L_j3', 4)
        self._set_default('pe_L_rotations', 4)
        self._set_default('pe_L_velocities', 4)
        self._set_default('pe_L_angular_velocities', 4)
        self._set_default('pe_L_dtransls', 4)

        self._set_default('pe_max_freq_log2_j3', self['pe_L_j3'] - 1)
        self._set_default('pe_max_freq_log2_rotations', self['pe_L_rotations'] - 1)
        self._set_default('pe_max_freq_log2_velocities', self['pe_L_velocities'] - 1)
        self._set_default('pe_max_freq_log2_angular_velocities', self['pe_L_angular_velocities'] - 1)
        self._set_default('pe_max_freq_log2_dtransls', self['pe_L_dtransls'] - 1)

        # ActionRecognizer
        self._set_default('ar_nactions', 50)
        self._set_default('ar_input_size', 6*22) # 6drot (TODO?)
        self._set_default('ar_hidden_size', 128)
        self._set_default('ar_fid_size', 128) # A2M uses 30
        self._set_default('ar_hidden_layers', 3) # A2M uses 2
        self._set_default('ar_window_size', 60)
        self._set_default('ar_rnn_dropout', 0.2)
        self._set_default('ar_train_with_soft_labels', False)

        # RNN
        self._set_default('hidden_size', 128)
        self._set_default('rnn_layers', 3)
        self._set_default('rnn_dropout', 0.2)
        self._set_default('hidden_state_type', 'sometimes_randomise')
        self._set_default('hidden_state_randomisation_ratio', 0.8)
        self._set_default('hidden_state_randomisation_method', 'xavier_uniform')
        self._set_default('clip_grad_norm', None)
        self._set_default('me_after_rnn', False)
        self._set_default('net_after_rnn', '1linear')
        self._set_default('hidden_dim_after_rnn', 256)

        # AvatarPoser
        self._set_default('ap_winsize', 40)

        # VAE-HMD
        self._set_default('input_seq_length', 1)
        self._set_default('resnet_kdiv3', 1)
        self._set_default('encoder_njoints', 3)
        self._set_default('latent_dim', 30)  # 15 30 60
        self._set_default('betavae_beta', 1)
        self._set_default('vaehmd_input_features_dim', -1)
        self._set_default('vaehmd_frozen_decoder', None) # Path to checkpoint for frozen decoder

        # MSP
        self._set_default('msp_encoder', 'MotionCLIP')
        self._set_default('msp_decoder', 'LastPose')
        self._set_default('msp_decoder_stabilizer', 'ShallowMLP')

        self._set_default('msp_train_sequentially', False)

        # AvatarPoser
        #self._set_default('checkpoint_avatarposer', None)

        # Training
        self._set_default('learning_rate', 0.002)
        self._set_default('lr_decay_step_size', 15)
        self._set_default('lr_decay_gamma', 0.5)
        self._set_default('lr_lower_bound', 5e-5)
        self._set_default('epochs', 60)
        self._set_default('log_every_n_global_steps', 150)

        self._set_default('random_root_rotation', True) # Randomly rotate root joint before data load

        # Misc
        self._set_default('target_framerate', self.TARGET_FRAMERATE_DEFAULT)

        self._set_default('viz_global_pose', False)

        # Deprecated settings
        self._set_default('training_data_percentage', 0.9)
        self._set_default('nsamples', 6)

    def _verify(self):
        def throwerr(field, defaultv=None, msg=None):
            raise ValueError("Invalid {}: {}{}".format(field, self.get(field, defaultv),
                        " ({})".format(msg) if msg else ""))

        if any(lt not in LOSS_TYPES for lt in self['loss'].split(LOSSES_SEP)):
            throwerr('loss')
        if self['hidden_state_type'] not in HIDDEN_STATE_TYPES:
            throwerr('hidden_state_type')
        if self['hidden_state_randomisation_method'] not in HIDDEN_STATE_RANDOMISATION_METHODS:
            throwerr('hidden_state_randomisation_method')
        if not (0 <= self.get('hidden_state_randomisation_ratio', 0) <= 1):
            throwerr('hidden_state_randomisation_ratio')
        if self['input_prev_frame_pose'] and not self['input_prev_frame_pose__use_gt']:
            raise NotImplementedError("With the current data loaders it is difficult to "
                    "feed previous frame pose prediction (at least during training")
        possible_inp_encs = ['none', 'simple']
        if self['input_encoder'] not in possible_inp_encs:
            throwerr('input_encoder', "Must be one of: {}".format(possible_inp_encs))
        possible_incomplete_me_policies = ['randomise', 'zero', 'use_as_is']
        if self['incomplete_motion_embeddings_policy'] not in possible_incomplete_me_policies:
            throwerr('incomplete_motion_embeddings_policy', "Must be one of: {}".format(
                        possible_incomplete_me_policies))
        possible_me_guids = [None, 'mse', 'cos']
        if self['motion_embedding_guidance'] not in possible_me_guids:
            throwerr('motion_embedding_guidance', "Must be one of: {}".format(
                        possible_me_guid))
        possible_me_eval = [None, 'MotionCLIP']
        if self['motion_embedding_evaluation'] not in possible_me_eval:
            throwerr('motion_embedding_evaluation', "Must be one of: {}".format(
                        possible_me_eval))
        possible_me_types = ['me', 'dp', 'me+dp']
        if self['motion_embedding_type'] not in possible_me_types:
            throwerr('motion_embedding_type', "Must be one of: {}".format(
                        possible_me_types))
        possible_me_modeltypes = ['MotionCLIP', 'KLD']
        for mts in [self['motion_embedding_model_type'],
                    self['motion_embedding_guidance_model_type'],
                    self['motion_embedding_evaluation_model_type']]:
            if mts not in possible_me_modeltypes:
                throwerr('motion_embedding_model_type', "Must be one of: {}".format(
                            possible_me_modeltypes))
        possible_me_inmeths = ['concat', 'crossattention']
        if self['motion_embedding_input_method'] not in possible_me_inmeths:
            throwerr('motion_embedding_input_method', "Must be one of: {}".format(
                        possible_me_inmeths))

    def get_raw_motion_embedding_dim(self):
        return 512

    def get_motion_embedding_dim(self):
        med = self['input_motion_embeddings__reduced_dim']
        if med:
            return med
        else:
            return self.get_raw_motion_embedding_dim()

    @property
    def motion_embedding_length(self):
        return 60

    @property
    def input_dim(self):
        sz = self.input_njoints * 3
        if self['input_rotations']:
            sz += self.input_njoints * self.in_rot_d
        if self['input_prev_frame_pose']:
            sz += self.in_rot_d * self.output_njoints
        if self['input_delta_translation']:
            sz += 3
        if self['input_velocities']:
            sz += self.input_njoints * 3
        if self['input_angular_velocities']:
            sz += self.input_njoints * self.in_rot_d
        if self['input_motion_embeddings']:
            sz += self.get_motion_embedding_dim()
        return sz

    @property
    def rot_d(self):
        if self['prefer_6d_rotation']:
            return 6
        else:
            return 3
    
    @property
    def in_rot_d(self):
        if self['input_9d_rot']:
            return 9
        else:
            return self.rot_d

    @property
    def out_rot_d(self):
        if self['output_9d_rot']:
            return 9
        else:
            return self.rot_d

    @property
    def output_dim(self):
        return self.out_rot_d * self.output_njoints

    @property
    def input_joints(self):
        if self['input_fullpose']:
            return list(range(22))
        else:
            return [15, 20, 21]  # head, wrist, lwrist

    @property
    def input_njoints(self):
        return len(self.input_joints)

    @property
    def output_njoints(self):
        return 22

    @property
    def upper_body_joints(self):
        return tuple(k for k in SMPL_UPPER_BODY_JOINTS if k < self.output_njoints)

    @property
    def upper_body_joints_nohmd(self):
        return tuple(set(SMPL_UPPER_BODY_JOINTS)
                .difference(SMPL_UPPER_EXTREMITY_JOINTS)
                .difference([SMPL_HEAD_JOINT]))

    @property
    def leg_joints(self):
        return SMPL_LEG_JOINTS

    @property
    def lower_body_joints(self):
        return SMPL_LOWER_BODY_JOINTS

    def __getitem__(self, key):
        dk_props = {'input_dim', 'rot_d', 'output_dim', 'input_joints', 'input_njoints', 'output_njoints',
        'leg_joints', 'lower_body_joints', 'upper_body_joints', 'in_rot_d', 'out_rot_d',
        'motion_embedding_length'}
        if key in dk_props:
            return getattr(self, key)
        else:
            return super().__getitem__(key)


class BaseNN(nn.Module):
    def __init__(self, config, cvt_kws=None):
        super().__init__()

        self.config = config
        self.mc = config['model_config']
        self.lg = glg()
        self.device = get_device()

        self.name = self.mc['name']
        self.do_export_windows = config['export_windows']
        self.exported_windows_path = config['exported_windows_path']

    def start_training(self, training_data_loader, validation_data_loader=None,
            checkpoints_save_dir=None, checkpoint_path=None, train_graph_path=None,
            nepochs=None, window_sz=None, **kw):
        global_step = 0
        nepochs = nepochs or self.mc['epochs']
        training_datas = set()
        val_stepsize = kw.get('validation_stepsize', 3)

        if checkpoint_path:
            glg().info("Loading checkpoint: %s", checkpoint_path)

            missing_keys, unexpected_keys = load_torch_model(self, checkpoint_path, strict=False)
            check_torch_model_keys(
                    missing_keys=missing_keys, unexpected_keys=unexpected_keys,
                    checkpoint_path=checkpoint_path)

        checkpoints_save_dir = checkpoints_save_dir or self.mc['checkpoints_save_directory']
        if not train_graph_path:
            train_graph_path = os.path.join(checkpoints_save_dir, 'train_graph.png')
            os.makedirs(os.path.dirname(train_graph_path), exist_ok=True)

        optimizer, scheduler = self._get_optim()

        cur_step_custom_data = {}
        train_init_data = self._training_init_data()

        start_dt = datetime.datetime.now()
        glg().info("Started training '%s' at %s", self.name, start_dt)

        lg = self.lg
        losses_cur_epoch = []
        all_train_losses = {}
        all_val_losses = {}
        global_step == 0
        with SummaryWriter(self.mc['tensorboard_logdir']) as writer:
            for epoch in range(nepochs):
                lg.info("Epoch %d/%d", epoch+1, nepochs)
                self.train()
                train_loss_array = []
                n_processed_inputs_this_epoch = 0
                t0 = time.time()
                tepoch = t0
                file_idx = -1000

                dloader_kw = {}
                if 'loading_window_sz' in kw:
                    dloader_kw['window_sz'] = kw['loading_window_sz']

                for i_data, d in enumerate(training_data_loader(**dloader_kw)):
                    file_idx_new = d.get('file_idx', -1)
                    if i_data > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                        same_file = True
                    else:
                        same_file = False
                    file_idx = file_idx_new

                    lg.debug("Data loading time: %f", time.time() - t0)

                    inp = totorch(d['input'], device=self.device)
                    outp = totorch(d['output'], device=self.device)

                    t0 = time.time()
                    self._pre_zerograd_callback(inp, outp, d)
                    lg.debug("Pre-zerograd time: %f", time.time() - t0)

                    self.meta = {
                        'same_file': same_file,
                        'writer': writer,
                        'data': d,
                        'window_sz': window_sz or self.mc['window_size']
                    }

                    optimizer.zero_grad()

                    t0 = time.time()

                    losses, cur_step_custom_data = \
                        self._training_step_callback(inp, outp, meta=self.meta,
                                prev_step_custom_data=cur_step_custom_data,
                                train_init_data=train_init_data,
                                train_get=lambda k: cur_step_custom_data.get(k,
                                    train_init_data.get(k, None)))

                    lg.debug("Train step time: %f", time.time() - t0)

                    optimizer.step()

                    loss = losses[0]
                    loss_detail = losses[1]
                    if loss is not None:
                        losses_cur_epoch.append(
                                loss.item() if isinstance(loss, torch.Tensor) else loss)

                    if isinstance(d['path'], str):
                        training_datas.add(d['path'])
                    else:  # is a list of paths
                        training_datas.update(d['path'])

                    if global_step % self.mc['log_every_n_global_steps'] == 0:
                        cum_loss = np.mean(losses_cur_epoch)
                        lg.info("%s - epoch: %d; global step: %d;"
                                " cumulative loss: %f; last loss: %s",
                            datetime.datetime.now(), epoch, global_step, cum_loss,
                            loss_detail)

                    global_step += 1
                    t0 = time.time()
                
                loss_epoch = np.mean(losses_cur_epoch or [np.nan])
                loss_epoch_std = np.std(losses_cur_epoch or [np.nan])
                loss_epoch_med = np.median(losses_cur_epoch or [np.nan])
                losses_cur_epoch.clear()
                all_train_losses[epoch] = loss_epoch
                lg.info(("The train loss of '{}' is {:.6f} (std={:.4f}, median={:.6f})."
                        " Epoch {} took {:.1f}s").format(
                            self.name, loss_epoch, loss_epoch_std, loss_epoch_med,
                            epoch+1, time.time() - tepoch))

                val_loss = None
                val_loss_std = None
                val_loss_med = None
                if validation_data_loader is not None:
                    if epoch > 0 and epoch % val_stepsize == 0:
                        lg.info("Validating...")
                        self.eval()
                        with torch.no_grad():
                            val_loss_d = self._validate(validation_data_loader,
                                    writer=writer, train_init_data=train_init_data,
                                    window_sz=self.meta['window_sz'])
                            val_loss = val_loss_d['mean']
                            val_loss_std = val_loss_d['std']
                            val_loss_med = val_loss_d['median']
                        self.train()
                else:
                    lg.warning("Validation data not provided. Skipping validation.")

                if val_loss is not None:
                    all_val_losses[epoch] = val_loss
                    lg.info(("Validation loss of '{}' is {:.6f}"
                            " (std={:.4f}, median={:.6f})").format(
                                self.name, val_loss, val_loss_std, val_loss_med))

                fig = self._plot_losses(all_train_losses, all_val_losses)
                lg.info("Saving training graph at: %s", train_graph_path)
                fig.savefig(train_graph_path)

                try:
                    writer.add_scalar("'{}' Train Loss".format(self.name),
                            loss_epoch, epoch)
                    if val_loss is not None:
                        writer.add_scalar("'{}' Validation Loss".format(self.name),
                                val_loss, epoch)
                except:
                    lg.warning("Failed to write train loss to writer")

                if ((epoch > 0 and epoch % self.mc['checkpoint_save_interval'] == 0) or
                        self.mc['checkpoint_save_interval'] == 1 or
                        epoch == nepochs - 1):
                    checkpoint_save_path = os.path.join(checkpoints_save_dir,
                            "checkpoint_{}_epoch_{:02}_of_{:02}({}).chkpt".format(
                        start_dt.strftime('%Y%m%dT%H%M%S'), epoch+1, nepochs,
                        datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
                    os.makedirs(checkpoints_save_dir, exist_ok=True)


                    state_dict_wo_clip = {k: v for k,v in self.state_dict().items() if not 'clip_model.' in k}
                    torch.save(state_dict_wo_clip, checkpoint_save_path)
                    lg.info("Saved checkpoint at %s", checkpoint_save_path)
                    training_datas_path = "{}_training_data.json".format(
                            os.path.splitext(checkpoint_save_path)[0])
                    with open(training_datas_path, 'w', encoding='utf-8') as f:
                        json.dump(list(training_datas), f)

                if scheduler is not None and scheduler.get_last_lr()[0] > self.mc['lr_lower_bound']:
                    scheduler.step()
                    lg.info("Last lr: {}".format(scheduler.get_last_lr()))

                try:
                    writer.flush()
                except:
                    lg.warning("Failed to flush writer")

            lg.info("Training done")


    def _validate(self, validation_data_loader, writer, train_init_data, window_sz=None):
        file_idx = -1000
        losses = []
        cur_step_custom_data = {}
        for i_data, d in enumerate(validation_data_loader(window_sz=window_sz)):
            file_idx_new = d.get('file_idx', -1)
            if i_data > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                same_file = True
            else:
                same_file = False
            file_idx = file_idx_new

            inp = totorch(d['input'], device=self.device)
            outp = totorch(d['output'], device=self.device)

            if self._pre_validation_step_callback is not None:
                self._pre_validation_step_callback(inp, outp, d)

            self.meta = {
                'same_file': same_file,
                'writer': writer,
                'data': d,
                'window_sz': window_sz or self.mc['window_sz']
            }

            losses_all, cur_step_custom_data = \
                self._validation_step_callback(inp, outp, meta=self.meta,
                        prev_step_custom_data=cur_step_custom_data,
                        train_init_data=train_init_data,
                        train_get=lambda k: cur_step_custom_data.get(k,
                            train_init_data.get(k, None)))
            loss = losses_all[0]

            if loss is not None:
                losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        mean = np.mean(losses)
        std = np.std(losses)
        med = np.median(losses)
        return {
            'mean': mean,
            'std': std,
            'median': med
        }
    
    def _get_eval_startfrm_for_output(self):
        out_startfrm = self.meta.get('frames_range', (0, -1))[0]
        eval_startfrm = self.mc['start_evaluation_from_frame']
        if eval_startfrm < out_startfrm:
            raise ValueError(("Output starts at frame {}, but evaluation "
                "must begin at frame {}.").format(out_startfrm, eval_startfrm))
        eval_startfrm_offset = eval_startfrm - out_startfrm
        return eval_startfrm_offset

    def start_testing(self, testing_data_loader, results_save_dir=None, window_sz=None,
            inference_callback=None, infer_global_pose=False, dset_path_index=-5,
            motion_embedding_test_noise_scale=0.0):
        if results_save_dir:
            os.makedirs(results_save_dir, exist_ok=True)
        else:
            glg().info("No results_save_dir specified; test results will not be saved on disk")

        def gcf(l):
            lf = get_loss_func(self.mc, l, apply_multiplier=False, keep_batch_dims=True)
            def _compute_flattened_loss(p, o):
                #if p is not None:
                #    p = p[:, eval_startfrm_offset:]
                #if o is not None:
                #    o = o[:, eval_startfrm_offset:]
                _arr = lf(p, o).detach().cpu().numpy()
                # Average across joints/output values for a single frame
                _arr = np.mean(_arr, axis=-1)
                return _arr.flatten()
            return _compute_flattened_loss

        loss_funcs = {k: gcf(k) for k in
            ['MSE', 'MPJPE', 'MPJRE_DEG', 'MPJVE', 'JITTER_GT', 'JITTER_PRED']}

        test_init_data = self._testing_init_data()

        self.nf4fid = 60
        self.gt_fid_inps = []
        self.pred_fid_inps = []

        self.lg.info("Loading classifier for FID...")
        self.fid_cls = a2m.load_classifier_for_fid(device=self.device,
                input_size_raw=3*24, dataset_type='humanact12',
                initial_hs_random=False)

        result_lists = {
            'losses': defaultdict(lambda: np.array([], dtype_np)),
            'windows': [], # List of ExportWindow
            'datasets': [],
            'groups': [],
            'genders': [],
            'nframes': []
        }

        def concat_losses_b(key, lss, _nb):
            if isinstance(lss, np.ndarray):# or isinstance(lss, torch.Tensor):
                lss = lss.reshape(-1)
            nl = len(lss)
            if nl > 0:
                if nl != _nb:
                    if nl < _nb:
                        assert nl == 1
                        new_l = []
                        for _ in range(_nb):
                            new_l.append(lss[0])
                    elif _nb != 0: # nl > _nb
                        assert nl % _nb == 0
                        cnt = nl // _nb
                        new_l = []
                        for i in range(_nb):
                            new_l.append(np.mean(lss[i*cnt:(i+1)*cnt]))
                    else: # _nb == 0
                        new_l = []
                    lss = new_l
                assert len(lss) == _nb
                result_lists['losses'][key] = np.append(result_lists['losses'][key], lss)

        file_idx = -1000
        cur_step_custom_data = {}

        self.eval()
        with torch.no_grad():
            #rest_pose_rmat = torch.zeros((1, 3, 22, 3, 3)).to(self.device)
            #rest_pose_rmat[..., :, :] = torch.eye(3)
            #conv_rest = self.test_converter(
            #        rest_pose_rmat[..., :, :2].reshape((1, 3, 22*6)))
            #rest_joints = conv_rest['joints'][:, [0], ...]

            nbatch_total = 0
            for i, d in enumerate(tqdm(testing_data_loader())):
                file_idx_new = d.get('file_idx', -1)
                if i > 0 and file_idx_new >= 0 and file_idx_new == file_idx:
                    same_file = True
                else:
                    same_file = False

                inp = totorch(d['input'], device=self.device)
                outp = totorch(d['output'], device=self.device)
                intm = d['intm']

                self._pre_testing_step_callback(inp, outp, d)

                self.meta = {
                    'same_file': same_file,
                    'data': d,
                    'index': i,
                    'intm': intm,
                    'window_sz': window_sz
                }

                (pred, outp), cur_step_custom_data = \
                    self._testing_step_callback(inp, outp, meta=self.meta,
                            prev_step_custom_data=cur_step_custom_data,
                            testing_init_data=test_init_data,
                            test_get=lambda k: cur_step_custom_data.get(k,
                                test_init_data.get(k, None)),
                            is_inference=inference_callback is not None)

                # Batchify
                if len(inp.shape) == 2:
                    isbatch = False
                    nbatch = 1
                elif len(inp.shape) == 3:
                    isbatch = True
                    nbatch = inp.shape[0]
                else:
                    raise ValueError
                if not isbatch:
                    outp = outp[None]
                    pred = pred[None]
                concat_losses = functools.partial(concat_losses_b, _nb=nbatch)
                nbatch_total += nbatch

                paths = list(d['path']) if not isinstance(d['path'], str) \
                        else nbatch * [d['path']]

                dsets = []
                groups = []
                nfrms = list(d['nframes']) if not (isinstance(d['nframes'], int) or isinstance(d['nframes'], np.int64)) \
                        else nbatch * [d['nframes']]
                genders = list(d['gender']) if not isinstance(d['gender'], str) \
                          else nbatch * [d['gender']]
                preprocessed_rootdir_abs = os.path.abspath(
                        get_root_dir(self.config, 'preprocessing'))
                for ip in range(nbatch):
                    #paths[ip] = os.path.relpath(os.path.abspath(paths[ip]), preprocessed_rootdir_abs)
                    #ds = os.path.normpath(paths[ip]).replace("..{}".format(os.path.sep), "") \
                    #    .split(os.path.sep)[1]
                    pathsplit = os.path.normpath(paths[ip]).split(os.path.sep)
                    ds = pathsplit[dset_path_index]
                    motion_name = os.path.sep.join(pathsplit[-2:])
                    dsets.append(ds)
                    groups.append("{}: {}".format(ds, motion_name))

                result_lists['datasets'].extend(dsets)
                result_lists['groups'].extend(groups)
                result_lists['nframes'].extend(nfrms)
                result_lists['genders'].extend(genders)

                c_pred = self.test_converter(pred)
                if 'c_outp' in d:
                    c_outp = d['c_outp']
                else:
                    c_outp = self.test_converter(outp)

                self._testing_eval_step_callback(c_pred, c_outp, intm, 
                        concat_losses=concat_losses,
                        loss_funcs=loss_funcs, result_lists=result_lists,
                        inference_callback=inference_callback, infer_global_pose=infer_global_pose)

            concat_losses = functools.partial(concat_losses_b, _nb=nbatch_total)
            self._testing_eval_finalise_callback(concat_losses, result_lists)

            if results_save_dir is not None:
                analyse_losses(losses=result_lists['losses'],
                        meta={k: v for (k, v) in result_lists.items() if k != 'losses'},
                        save_dir=results_save_dir, model_config=self.mc)

                raw_file_fp = os.path.join(results_save_dir, self.mc['test_results_raw_fn'])
                glg().info("Saving raw file to '%s'...", raw_file_fp)
                result_lists['losses'] = dict(result_lists['losses'])
                with open(raw_file_fp, 'wb') as f:
                    d = {
                        'config': self.config,
                        'model_config': self.mc,
                        'lists': result_lists
                    }
                    pickle.dump(d, f)

            if self.do_export_windows:
                glg().info("Exporting %d windows to '%s'",
                        len(result_lists['windows']), self.exported_windows_path)
                os.makedirs(os.path.dirname(self.exported_windows_path), exist_ok=True)
                with open(self.exported_windows_path, 'wb') as f:
                    d = {
                        'windows': list(map(lambda ew: dict(ew._asdict()),
                                    result_lists['windows']))
                    }
                    pickle.dump(d, f)

    def _get_optim(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.mc['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                step_size=self.mc['lr_decay_step_size'],
                gamma=self.mc['lr_decay_gamma'])
        return (optimizer, scheduler)

    def _plot_losses(self, train_losses: dict, validation_losses: dict):
        train_col = "#1E90FF"
        valid_col = "#FF8C00"

        hasvalid = len(validation_losses) > 0

        ephs_train = np.zeros(len(train_losses), dtype=np.int)
        trainlosses_a = np.zeros(len(train_losses), dtype=np.float32)
        for i, (eph, lss) in enumerate(train_losses.items()):
            ephs_train[i] = eph
            trainlosses_a[i] = lss
        mintrainloss_i = np.argmin(trainlosses_a)
        mintrainloss = trainlosses_a[mintrainloss_i]
        mineph, maxeph = np.min(ephs_train), np.max(ephs_train)

        if hasvalid:
            ephs_valid = np.zeros(len(validation_losses), dtype=np.int)
            validlosses_a = np.zeros(len(validation_losses), dtype=np.float32)
            for i, (eph, lss) in enumerate(validation_losses.items()):
                ephs_valid[i] = eph
                validlosses_a[i] = lss
            minvalidloss_i = np.argmin(validlosses_a)
            minvalidloss = validlosses_a[minvalidloss_i]
        else:
            ephs_valid = None
            validlosses_a = None
            minvalidloss_i = None
            minvalidloss = None

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(ephs_train, trainlosses_a, label='Train', color=train_col)
        if hasvalid:
            ax.plot(ephs_valid, validlosses_a, label='Validation', color=valid_col)
        ax.legend(loc="upper right");

        ax.plot([mineph, maxeph], [mintrainloss, mintrainloss], color='#A9A9A9', linestyle='--')
        mintrain_p = (ephs_train[mintrainloss_i], mintrainloss)

        ax.plot([mintrain_p[0]], [mintrain_p[1]], color=train_col, marker='o')
        ax.text(*mintrain_p, "({}, {:.03f})".format(*mintrain_p), color="#000080",
                horizontalalignment='left', verticalalignment='bottom')

        if hasvalid:
            ax.plot([mineph, maxeph], [minvalidloss, minvalidloss], color='#676767', linestyle='--')
            minvalid_p = (ephs_valid[minvalidloss_i], minvalidloss)
            ax.plot([minvalid_p[0]], [minvalid_p[1]], color=valid_col, marker='o')
            ax.text(*minvalid_p, "({}, {:.03f})".format(*minvalid_p), color="#A0522D",
                    horizontalalignment='left', verticalalignment='bottom')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        #fig.tight_layout()
        return fig

    def _training_init_data(self):
        return {}

    def _testing_init_data(self):
        return {}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            training_init_data, train_get):
        raise AssertionError("Training step callbaack not implemented")
        loss = None
        return loss, prev_step_custom_data

    def _pre_zerograd_callback(self, inp, outp, d):
        pass

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            training_init_data, train_get):
        glg().warning("Validation step callback not implemented")
        return None, prev_step_custom_data

    def _pre_validation_step_callback(self, inp, outp, d):
        pass

    def _pre_testing_step_callback(self, inp, outp, d):
        pass

    def _testing_eval_step_callback(self, c_pred, c_outp, intm, *,
            concat_losses: Callable[[str, np.ndarray], None], loss_funcs, result_lists,
            inference_callback=None, infer_global_pose=False):
        pass

    def _testing_eval_finalise_callback(self, concat_losses, result_lists):
        pass

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        raise AssertionError("Testing step callbaack not implemented")
        return (None, None), prev_step_custom_data


class ActionRecognizer(BaseNN):
    def __init__(self, config):
        super().__init__(config=config)
        self.name == 'ActionRecognizer'

        input_size = self.mc['ar_input_size']
        hidden_size = self.mc['ar_hidden_size']
        fid_nfeats = self.mc['ar_fid_size']
        nhidden = self.mc['ar_hidden_layers']
        self.nactions = self.mc['ar_nactions']
        self.window_for_recog = self.mc['ar_window_size']
        self.train_with_soft_labels = self.mc['ar_train_with_soft_labels']
        if self.nactions > len(ACTIONS_TOP50):
            raise ValueError
        dropout_pct = self.mc['ar_rnn_dropout']
        self.actions_to_use = list(ACTIONS_TOP50[:self.nactions]) + ['other'] # 'other' is LAST
        if 'transition' in self.actions_to_use:
            self.actions_to_use.remove('transition')
        else:
            self.actions_to_use = self.actions_to_use[:-1]
        self.actions_to_use = tuple(self.actions_to_use)
        assert len(self.actions_to_use) == self.nactions
        act_idxs = []
        for act in self.actions_to_use:
            if act == 'other':
                act_idxs.append(0)
            else:
                idx = IDX2ACT.index(act)
                act_idxs.append(idx)
        act_idxs = np.array(act_idxs)
        sum_idxs = np.array(list(set(range(len(IDX2ACT)))
                    .difference(act_idxs).difference([IDX2ACT.index('transition')])))
        ACTS_THRESH = 1 / self.nactions

        def act_cat_to_output_fun(act_cat, training=False):
            """
                act_cat: (nb x)* len(IDX2ACT)
            """
            assert len(act_cat.shape) == 2 and act_cat.shape[-1] == len(IDX2ACT)
            output = act_cat[..., act_idxs]
            output[..., -1] = act_cat[..., sum_idxs].sum(-1)
            output[output.sum(-1) < ACTS_THRESH] = \
                torch.FloatTensor((self.nactions - 1) * [0] + [1]).type(act_cat.dtype).to(self.device)
            if training and self.train_with_soft_labels:
                output = nn.functional.softmax(output, dim=-1)
            else:
                output = output.argmax(-1)
            return output
        self.act_cat_to_output = act_cat_to_output_fun

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                num_layers=nhidden, batch_first=True, dropout=dropout_pct)
        self.linear1 = nn.Linear(hidden_size, fid_nfeats)
        self.final_layer = nn.Linear(fid_nfeats, self.nactions)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, hidden=None):
        """
            x: nb x self.window_for_recog x input_size
        """
        assert x.shape[-2] == self.window_for_recog
        rnn_o, hidden = self.rnn(x, hidden)
        fid_feats = self.linear1(rnn_o[..., -1, :])
        logits = self.final_layer(fid_feats)

        return {
            'logits': logits,
            'fid_feats': fid_feats,
            'hidden': hidden,
        }

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True, rand_start=True, return_losses=True):
        if not self.mc['input_fullpose']:
            raise ValueError("Full pose must be input to train ActionRecognizer")

        assert len(inp.shape) == 3  # nb x nf x nfeats
        nf = inp.shape[-2]
        assert self.window_for_recog <= nf

        if rand_start:
            st = random.randrange(0, nf - self.window_for_recog)
        else:
            st = 0
        ed = st + self.window_for_recog - 1

        act_cat = meta['data']['act_cat'][..., st:ed+1, :].mean(1) # nb x NACTS
        if not isinstance(act_cat, torch.Tensor):
            act_cat = torch.from_numpy(act_cat).to(self.device)
        gt_acts = self.act_cat_to_output(act_cat, training=True) # nb x self.nactions

        d = self(inp[..., st:ed+1, :])

        celoss = self.ce(d['logits'], gt_acts)

        loss = celoss

        if is_training:
            loss.backward()

        if return_losses:
            return ((loss.item(), {'CE': celoss.item()}), {})
        else:
            pred_acts_inds = d['logits'].argmax(-1).detach()
            if len(gt_acts.shape) == 1:
                gt_acts_inds = gt_acts
            elif len(gt_acts.shape) == 2:
                gt_acts_inds = gt_acts.argmax(-1)
            else:
                raise ValueError("Invaid gt_acts: {}".format(gt_acts.shape))
            ((pred_acts_inds, gt_acts_inds), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False, return_losses=False)


class PoseEstimatorBase(BaseNN):
    def __init__(self, config, cvt_kws=None):
        super().__init__(config=config)

        self._convs_p = []
        self._convs_o = []
        self._loss_fns = []
        for lss in self.mc['loss'].split(LOSSES_SEP):
            conv_p = get_required_convs_for_loss(lss, is_pred=True)
            conv_o = get_required_convs_for_loss(lss, is_pred=False)
            if conv_p is not None:
                self._convs_p.append(conv_p)
            if conv_o is not None:
                self._convs_o.append(conv_o)
            self._loss_fns.append(
                    (lss, is_reg(lss), get_loss_func(self.mc, lss), conv_p, conv_o))

        smplmodel_path = get_model_path(config, 'smpl', 'male')
        smplmodel = load_smpl_model(smplmodel_path, as_class=False)
        self.convert_pred = nn_out_converter(config=config, targets=self._convs_p,
                training=True, model=smplmodel, **(cvt_kws or {}))
        self.convert_outp = nn_out_converter(config=config, targets=self._convs_o,
                training=True, model=smplmodel, **(cvt_kws or {}))

        self.name = "PE__{}".format(self.mc['name'])

        self.test_converter = nn_out_converter(config=config,
                targets=['rot_mats_g', 'joints', 'vel', 'jitter'], training=False)

        # Motion Embedding
        self.me_guid = self.mc['motion_embedding_guidance']
        self.me_eval = self.mc['motion_embedding_evaluation']
        self.me_inp = self.mc['input_motion_embeddings']
        self.dp_inp = self.mc['input_dp']
        self.me_len = self.mc['motion_embedding_length']
        #self.mo_embedder_guid = None # Must accept full pose
        #self.mo_embedder_eval = None # Must accept full pose
        #self.mo_embedder_otf = None
        self.me_mapper = None
        self.concat_me_to_inp = True
        self.me_train_inp_ns = self.mc['motion_embedding_train_noise_scale']
        self.me_type = self.mc['motion_embedding_type']
        self.motion_priors = {}
        self.me_otf_everywhere = self.mc['include_motion_embedder']
        self.me_use_logvar = self.mc['motion_embedder_use_logvar']
        self.me_inmeth = self.mc['motion_embedding_input_method']
        self.me_catt = None
        #self.me_catt_me_as_query = self.mc['motion_embedding_catt_me_as_query']
        if self.dp_inp and not self.me_otf_everywhere:
            raise NotImplementedError

        if self.dp_inp: # TODO
            raise NotImplementedError

        if self.me_inp:
            med = self.mc['input_motion_embeddings__reduced_dim']
            if med and med > 0:
                self.me_mapper = nn.Linear(self.mc.get_raw_motion_embedding_dim(), med)

            me_otf_path = self.mc['motionclip_checkpoint_path_for_otf']
            if me_otf_path:
                glg().info("MotionCLIP for OTF: '%s'", me_otf_path)
                self.motion_priors['trainable'] = MotionPrior(
                        modeltype=self.mc['motion_embedding_model_type'],
                        checkpoint_path=me_otf_path,
                        sparse_encoder=self.mc['motion_embedding_sparse_input'],
                        use_logvar=self.mc['motion_embedder_use_logvar'],
                        freeze=('both' if self.mc['motion_embedder_encoder_freeze'] else
                            'decoder'))
                if self.mc['include_motion_embedder']:
                    self.add_module('me_otf', self.motion_priors['trainable'])
                elif not self.mc['motion_embedder_encoder_freeze']:
                    raise ValueError("Motion Embedder (OTF) encoder wants encoder "
                            "frozen but Motion Embedder is not included")

        if self.me_guid:
            glg().info("MotionCLIP for guidance: '%s'",
                    self.mc['motionclip_checkpoint_path_for_guidance'])
            self.motion_priors['guidance'] = MotionPrior(
                    self.mc['motion_embedding_guidance_model_type'],
                    checkpoint_path=self.mc['motionclip_checkpoint_path_for_guidance'],
                    sparse_encoder=False)
        if self.me_eval:
            glg().info("MotionCLIP for evaluation: '%s'",
                    self.mc['motionclip_checkpoint_path_for_evaluation'])
            self.motion_priors['evaluation'] = MotionPrior(
                    self.mc['motion_embedding_evaluation_model_type'],
                    checkpoint_path=self.mc['motionclip_checkpoint_path_for_evaluation'],
                    sparse_encoder=False)

        self.use_fp_for_otf_me_calc = not self.mc['motion_embedding_sparse_input']

    @property
    def mo_embedder_guid(self):
        return self.motion_priors['guidance']

    @property
    def mo_embedder_eval(self):
        return self.motion_priors['evaluation']

    @property
    def mo_embedder_otf(self):
        return self.motion_priors['trainable']

    def motion_embedding_loss(self, rmats_p, *, rmats_o=None, gt_mes=None,
            apply_loss_multiplier=True, keepdim=False, method=None,
            use_eval_model=False):
        pad = torch.zeros(*rmats_p.shape[:-3], 25-22, 3, 3)
        pad[..., :2, :, :] = torch.eye(3)
        rmats25_p = torch.concat((rmats_p, pad.to(self.device)), dim=-3)
        if method is None:
            method = self.mc['motion_embedding_guidance']
        assert method

        rmats25_o = None
        if gt_mes is None:
            rmats25_o = torch.concat(
                    (rmats_o, pad.clone().to(self.device)), dim=-3)

        nf = rmats_p.shape[-4]
        if nf < self.me_len:
            raise RuntimeError(("Window size too small ({}) to compute ME "
                    "(len={}) for guidance.").format(nf, self.me_len))
        embedder = self.mo_embedder_guid if not use_eval_model else self.mo_embedder_eval
        losses = []
        for pose_frm in range(self.me_len, nf):
            assert len(rmats_p.shape) == 5
            ed = pose_frm - 1
            st = ed - self.me_len + 1
            me_inp = rmats25_p[:, st:ed+1]

            # nb x medim
            me_pred = embedder.embed(me_inp, with_no_grad=False).to(self.device)
            if gt_mes is None:
                me_gt = embedder.embed(rmats25_o[:, st:ed+1]).to(self.device)
            else:
                me_gt = gt_mes[:, pose_frm]

            # nb x 1
            if method == 'mse':
                loss = nn.functional.mse_loss(me_pred, me_gt, reduction='none').sum(-1)
            elif method == 'cos':
                loss = 1.0 - \
                       nn.functional.cosine_similarity(me_pred, me_gt, dim=-1)
            else:
                raise ValueError("Unknown ME guidance metric: {}".format(method))
            losses.append(loss)

        losses = torch.concat(losses, dim=0)
        if keepdim:
            return losses if not apply_loss_multiplier else \
                self.mc['loss_multiplier_MOTION_EMBEDDING'] * losses
        else:
            me_loss = torch.mean(losses)
            return me_loss if not apply_loss_multiplier else \
                self.mc['loss_multiplier_MOTION_EMBEDDING'] * me_loss

    def criterion(self, pred_convs, outp_convs):
        loss_sum = None
        loss_sum_view = None
        loss_detail = None
        loss_detail_view = {}
        for loss_name, loss_isreg, lfn, conv_p, conv_o in self._loss_fns:
            pred_e = pred_convs.get(conv_p, None)
            outp_e = outp_convs.get(conv_o, None)
            loss = lfn(pred_e, outp_e)
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum += loss

            if not loss_isreg:
                if loss_sum_view is None:
                    loss_sum_view = loss.item()
                else:
                    loss_sum_view += loss.item()
            loss_detail_view[loss_name] = loss.item()

        if self.me_guid:
            #gt_me = self.meta['data']['raw_me_unperturbed_torch']
            me_loss = self.motion_embedding_loss(rmats_p=pred_convs['rot_mats'],
                    #gt_mes=gt_me[:, self.meta.get('loss_startidx', 0):],
                    rmats_o=outp_convs['rot_mats'],
                    method=self.mc['motion_embedding_guidance'])
            loss_sum += me_loss
            loss_sum_view += me_loss

        return loss_sum, loss_sum_view, loss_detail, loss_detail_view

    def _testing_init_data(self):
        # SIP, PIPAE, PIPJE
        PIP_IGNORE_JOINTS = [0, 7, 8, 10, 11, 20, 21, 22, 23]
        self.SIP_JOINTS = [1, 2, 16, 17]
        self.PIP_JOINTS = [j for j in range(22) if j not in PIP_IGNORE_JOINTS]
        # Footskate/Footslide
        self.FS_THRESH_LIST = []
        for lt in LOSS_TYPES:
            if lt.startswith("FS_") and lt.endswith("_GT"):
                i1 = len("FS_")
                i2 = len(lt) - len("_GT")
                th_m = float(lt[i1:i2]) / 100
                self.FS_THRESH_LIST.append(th_m)

        return {}

    def _pre_zerograd_callback(self, inp, outp, d):
        self._prepare_me(inp, outp, d)

    def _pre_validation_step_callback(self, inp, outp, d):
        self._prepare_me(inp, outp, d)

    def _pre_testing_step_callback(self, inp, outp, d):
        self._prepare_me(inp, outp, d)

    def _testing_eval_step_callback(self, c_pred, c_outp, intm, *,
            concat_losses: Callable[[str, np.ndarray], None], loss_funcs, result_lists,
            inference_callback=None, infer_global_pose=False):
        outp_startfrm = self._get_eval_startfrm_for_output()
        for cc in [c_pred, c_outp]:
            for k, v in cc.items():
                if len(v.shape) > 2: # nb x nf x ...
                    cc[k] = v[:, outp_startfrm:]

        # Motion Embedding Error
        me_err_cos = None
        if self.me_eval:
            nfrms = c_pred['rot_6d'].shape[1]
            if nfrms < self.me_len:
                self.lg.info("Not enough frames (%d) to compute ME error (need %d)",
                        nfrms, self.me_len)

                for method in ['mse', 'cos']:
                    k = 'motion_embedding_err_{}'.format(method)
                    if len(result_lists['losses'][k]) > 0:
                        e = np.mean(result_lists['losses'][k])
                    else:
                        e = 0.0
                    me_err = np.array(e, dtype=np.float32)
                    if method == 'cos':
                        me_err_cos = me_err
                    concat_losses(k, me_err)
            else:
                for method in ['mse', 'cos']:
                    me_err = self.motion_embedding_loss(
                        rmats_p=c_pred['rot_mats'],
                        #gt_mes=self.meta['data']['raw_me_unperturbed_torch'][:,
                        #    self.meta.get('loss_startidx', 0):],
                        rmats_o=c_outp['rot_mats'],
                        method=method, use_eval_model=True,
                        apply_loss_multiplier=False, keepdim=True).detach().cpu().numpy()
                    if method == 'cos':
                        me_err_cos = me_err
                    concat_losses('motion_embedding_err_{}'.format(method), me_err)

        # Loss funcs
        compute_mse = loss_funcs['MSE']
        compute_mpjpe = loss_funcs['MPJPE']
        compute_mpjre = loss_funcs['MPJRE_DEG']
        compute_mpjve = loss_funcs['MPJVE']
        compute_jitter_pred = loss_funcs['JITTER_PRED']
        compute_jitter_gt = loss_funcs['JITTER_GT']

        # Recover global information
        rm_p, rm_o = c_pred['rot_mats'], c_outp['rot_mats']
        j_p, j_o = c_pred['joints'], c_outp['joints']

        j22 = totorch(intm['j22'], device=self.device)
        head_idx = SMPL_JOINT_INDICES['Head']
        head_idx_in_input = self.mc.input_joints.index(SMPL_JOINT_INDICES['Head'])
        head_pos = totorch(intm['j3'][..., head_idx_in_input, :],
                device=self.device)
        #head_rot = totorch(intm['r3'][..., head_idx_in_input, :, :],
        #        device=self.device)
        eval_startfrm = self.mc['start_evaluation_from_frame']
        head_pos = head_pos[..., eval_startfrm:, :]
        #head_rot = head_rot[..., eval_startfrm:, :, :]
        j22 = j22[..., eval_startfrm:, :, :]

        # Compute root joint
        #root_tfm_gt = recover_root_transform(
        #        head_pos, head_idx, rm_o, c_outp['kintree'],
        #        joints=j_o)
        #        #start_rmat=head_rot, joints_rest=rest_joints)
        #assert torch.isclose(root_tfm_gt[..., :3, :3], rm_o[..., 0, :, :]).all()
        #j_o_g = (j_o - j_o[..., [0], :]) + root_tfm_gt[..., None, :3, 3]
        #assert (j22 - j_o_g).mean().item() < 0.0001
        j_o_g = j22.to(self.device)
        root_tfm_pred = recover_root_transform(
                head_pos, head_idx, rm_p, c_outp['kintree'],
                joints=j_p).to(self.device)
                #start_rmat=head_rot, joints_rest=rest_joints)
        j_p_g = (j_p - j_p[..., [0], :]) + root_tfm_pred[..., None, :3, 3]

        nf = j_o_g.shape[1]
        if nf >= self.nf4fid:
            gt22ofst = (j_o_g[:, :self.nf4fid] - j_o_g[:, 0, 0, :][:, None, None]).detach()
            gt24ofst = model_output_joints_to_smpl_joints(gt22ofst)
            gtfidinp = gt24ofst.reshape(*gt24ofst.shape[:-2], -1)
            pred22ofst = (j_p_g[:, :self.nf4fid] - j_p_g[:, 0, 0, :][:, None, None]).detach()
            pred24ofst = model_output_joints_to_smpl_joints(pred22ofst)
            predfidinp = pred24ofst.reshape(*pred24ofst.shape[:-2], -1)
            assert gtfidinp.shape[-1] == 24*3 and predfidinp.shape[-1] == 24*3
 
            self.gt_fid_inps.append((gtfidinp, None))
            self.pred_fid_inps.append((predfidinp, None))
        else:
            glg().warning("Minumum of %d frames required to process FID; has %d frames",
                    self.nf4fid, nf)

        # Compute losses (2)
        l_mpjre = compute_mpjre(rm_p, rm_o)
        l_mpjre_ub = compute_mpjre(rm_p[..., self.mc.upper_body_joints, :, :],
                rm_o[..., self.mc.upper_body_joints, :, :])
        l_mpjre_ub2 = compute_mpjre(rm_p[..., self.mc.upper_body_joints_nohmd, :, :],
                rm_o[..., self.mc.upper_body_joints_nohmd, :, :])
        l_mpjre_lb = compute_mpjre(rm_p[..., self.mc.lower_body_joints, :, :],
                rm_o[..., self.mc.lower_body_joints, :, :])
        l_mpjre_legs = compute_mpjre(rm_p[..., self.mc.leg_joints, :, :],
                rm_o[..., self.mc.leg_joints, :, :])
        l_pipae = compute_mpjre(rm_p[..., self.PIP_JOINTS, :, :],
                rm_o[..., self.PIP_JOINTS, :, :])
        l_sip = compute_mpjre(rm_p[..., self.SIP_JOINTS, :, :],
                rm_o[..., self.SIP_JOINTS, :, :])
        l_mpjpe = compute_mpjpe(j_p, j_o)
        l_mpjpe_ub = compute_mpjpe(j_p[..., self.mc.upper_body_joints, :],
                j_o[..., self.mc.upper_body_joints, :])
        l_mpjpe_ub2 = compute_mpjpe(j_p[..., self.mc.upper_body_joints_nohmd, :],
                j_o[..., self.mc.upper_body_joints_nohmd, :])
        l_mpjpe_lb = compute_mpjpe(j_p[..., self.mc.lower_body_joints, :],
                j_o[..., self.mc.lower_body_joints, :])
        l_mpjpe_legs = compute_mpjpe(j_p[..., self.mc.leg_joints, :],
                j_o[..., self.mc.leg_joints, :])
        l_mpjpe_global = compute_mpjpe(j_p_g, j_o_g)

        l_pipje = compute_mpjpe(j_p[..., self.PIP_JOINTS, :],
                j_o[..., self.PIP_JOINTS, :])
        l_mpjve = compute_mpjve(c_pred['vel'], c_outp['vel'])

        jitter_pred = compute_jitter_pred(j_p_g[..., self.PIP_JOINTS, :], None)
        jitter_gt = compute_jitter_gt(None, j_o_g[..., self.PIP_JOINTS, :])

        fs_d = {
            'pred': compute_foot_slides(j_p_g, thresh=self.FS_THRESH_LIST, fps=1),
            'gt': compute_foot_slides(j_o_g, thresh=self.FS_THRESH_LIST, fps=1)
        }

        concat_losses('mpjre', l_mpjre)
        concat_losses('mpjre_upper_body', l_mpjre_ub)
        concat_losses('mpjre_ub-hmd', l_mpjre_ub2)
        concat_losses('mpjre_lower_body', l_mpjre_lb)
        concat_losses('mpjre_legs', l_mpjre_legs)
        concat_losses('pipae', l_pipae)
        concat_losses('sip', l_sip)
        concat_losses('mpjpe', l_mpjpe)
        concat_losses('mpjpe_upper_body', l_mpjpe_ub)
        concat_losses('mpjpe_ub-hmd', l_mpjpe_ub2)
        concat_losses('mpjpe_lower_body', l_mpjpe_lb)
        concat_losses('mpjpe_legs', l_mpjpe_legs)
        concat_losses('mpjpe_global', l_mpjpe_global)
        concat_losses('pipje', l_pipje)
        concat_losses('mpjve', l_mpjve)
        concat_losses('jitter_pred', jitter_pred)
        concat_losses('jitter_gt', jitter_gt)
        for ith, th in enumerate(self.FS_THRESH_LIST):
            for k in ['freq', 'speed_all', 'speed_exp_all']:
                for t in ['pred', 'gt']:  # Footskate
                    v = fs_d[t][ith][k].detach().cpu().numpy()
                    if k == 'freq':
                        v *= 100
                        k_lbl = k + '_pct'
                    elif k == 'speed' or k == 'speed_all':
                        v *= self.mc['target_framerate']
                        k_lbl = k
                    else:
                        k_lbl = k
                    concat_losses("fs_{:.02f}_{}_{}".format(
                                100 * th, k_lbl.replace('_all', ''), t), v)

        if self.do_export_windows or inference_callback is not None:
            losses_d = {
                 'mpjre': l_mpjre,
                 'mpjre_upper_body': l_mpjre_ub,
                 'mpjre_ub-hmd': l_mpjre_ub2,
                 'mpjre_lower_body': l_mpjre_lb,
                 'mpjre_legs': l_mpjre_legs,
                 'pipae': l_pipae,
                 'sip': l_sip,
                 'mpjpe': l_mpjpe,
                 'mpjpe_upper_body': l_mpjpe_ub,
                 'mpjpe_ub-hmd': l_mpjpe_ub2,
                 'mpjpe_lower_body': l_mpjpe_lb,
                 'mpjpe_legs': l_mpjpe_legs,
                 'mpjpe_global': l_mpjpe_global,
                 'pipje': l_pipje,
                 'mpjve': l_mpjve,
                 'jitter_pred': jitter_pred,
                 'jitter_gt': jitter_gt,
                 'me_err': me_err_cos
            }

            # Export window
            if self.do_export_windows:
                nb = rm_o.shape[0]
                dd = self.meta['data']

                nframes_total_b = dd['nframes']
                st_frm_b = dd['start_frame']
                ed_frm_b = dd['end_frame']
                fp_d = dd['path']

                mean_losses_d = {}
                for l_name, l_flat in losses_d.items():
                    assert len(l_flat.shape) == 1 and l_flat.shape[0] % nb == 0
                    l_shaped = l_flat.reshape(nb, -1)
                    assert l_shaped.shape[1] <= nf
                    mean_losses_d[l_name] = np.mean(l_shaped, axis=1)

                act_cat = dd['act_cat']
                act_cat_b = np.mean(act_cat, axis=1)

                for ib in range(nb):
                    stt = {
                        'act_cat': act_cat_b[ib],
                    }
                    for l_name, l_mean_b in mean_losses_d.items():
                        stt[l_name] = l_mean_b[ib]
                    result_lists['windows'].append(ExportWindow(
                        path=fp_d[ib], start_frame=st_frm_b[ib], end_frame=ed_frm_b[ib],
                        nframes_clip=nframes_total_b[ib], stats=stt))

            # Perframe losses (perframe, MA) for inference callback (viz)
            if inference_callback is not None:
                ma_losses_ws = 60#30
                ma_losses_d = {}
                for loss_k, loss_v in losses_d.items():
                    if loss_v is None:
                        continue
                    mal = np.cumsum(loss_v)
                    mal[ma_losses_ws:] = mal[ma_losses_ws:] - mal[:-ma_losses_ws]
                    mal /= ma_losses_ws
                    ma_losses_d[loss_k] = mal

                for pflname, pfld in \
                    [('perframe_losses', losses_d), ('perframe_ma_losses', ma_losses_d)]:
                    losses_perframe_d = defaultdict(dict)
                    nfrms = next(iter(c_outp.values())).shape[1]
                    for loss_k, loss_v in pfld.items():
                        if loss_v is None or len(loss_v.shape) == 0:
                            continue
                        assert len(loss_v.shape) == 1
                        loss_nfrms = loss_v.shape[0]
                        assert nfrms >= loss_nfrms
                        for i, fr in enumerate(range(nfrms - loss_nfrms, nfrms)):
                            losses_perframe_d[fr][loss_k] = loss_v[i]
                    self.meta[pflname] = losses_perframe_d

        # Inference callback
        if inference_callback is not None:
            if not infer_global_pose:
                ic_pred_j = j_p
                ic_outp_j = j_o
            else:
                ic_pred_j = j_p_g
                ic_outp_j = j_o_g
            ic_pred_j = model_output_joints_to_smpl_joints(
                    ic_pred_j.detach().cpu().numpy())
            ic_outp_j = model_output_joints_to_smpl_joints(
                    ic_outp_j.detach().cpu().numpy())
            inference_callback(ic_pred_j, ic_outp_j,
                    meta={**self.meta,
                    'pred_rot_mats': c_pred['rot_mats'].detach().cpu().numpy(),
                    'outp_rot_mats': c_outp['rot_mats'].detach().cpu().numpy(),
                    'pred_joints': c_pred['joints'].detach().cpu().numpy(),
                    'outp_joints': c_outp['joints'].detach().cpu().numpy()})

    def _testing_eval_finalise_callback(self, concat_losses, result_lists):
        fid = a2m.evaluate_fid(ground_truth_motion_loader=self.gt_fid_inps,
                gru_classifier_for_fid=self.fid_cls,
                motion_loaders={'testdset': self.pred_fid_inps},
                device=self.device, file=None)['fid']['testdset']
        concat_losses('fid', [fid])

        assert len(self.gt_fid_inps) == len(self.pred_fid_inps)
        nfidinps = len(self.gt_fid_inps)

        fid_h = 0
        fid_h_gt = 0
        fid_h_pred = 0
        for i_fidh in tqdm(range(self.mc['fid_nshufs']), desc="Computing FID_H"):
            rinds = list(range(nfidinps))
            random.shuffle(rinds)
            gt_fid_inps_shuf = [self.gt_fid_inps[rinds[i]] for i in range(nfidinps)]
            pred_fid_inps_shuf = [self.pred_fid_inps[rinds[i]] for i in range(nfidinps)]
            nfidinps_h = nfidinps // 2
            gt_fid_inps_h1 = gt_fid_inps_shuf[:nfidinps_h]
            gt_fid_inps_h2 = gt_fid_inps_shuf[nfidinps_h:]
            pred_fid_inps_h1 = pred_fid_inps_shuf[nfidinps_h:]
            pred_fid_inps_h2 = pred_fid_inps_shuf[:nfidinps_h]

            fid_h += a2m.evaluate_fid(ground_truth_motion_loader=gt_fid_inps_h1,
                      gru_classifier_for_fid=self.fid_cls,
                      motion_loaders={'testdset': pred_fid_inps_h1},
                      device=self.device, file=None)['fid']['testdset']
            fid_h_gt += a2m.evaluate_fid(ground_truth_motion_loader=gt_fid_inps_h1,
                      gru_classifier_for_fid=self.fid_cls,
                      motion_loaders={'testdset': gt_fid_inps_h2},
                      device=self.device, file=None)['fid']['testdset']
            fid_h_pred += a2m.evaluate_fid(ground_truth_motion_loader=pred_fid_inps_h1,
                      gru_classifier_for_fid=self.fid_cls,
                      motion_loaders={'testdset': pred_fid_inps_h2},
                      device=self.device, file=None)['fid']['testdset']
        fid_h /= self.mc['fid_nshufs']
        fid_h_gt /= self.mc['fid_nshufs']
        fid_h_pred /= self.mc['fid_nshufs']
        concat_losses('fid_h', [fid_h])
        concat_losses('fid_h_gt', [fid_h_gt])
        concat_losses('fid_h_pred', [fid_h_pred])

    def _prepare_me(self, inp, outp, d):
        me = None
        if self.me_inp:
            if 'me' in d and not self.me_otf_everywhere:
                if self.me_use_logvar:
                    me_mu = totorch(d['me_mu'], device=self.device)
                    me_logvar = totorch(d['me_logvar'], device=self.device)
                    std = torch.exp(0.5 * me_logvar)
                    e = torch.randn_like(std)
                    me = me_mu + e * std
                else:
                    me = totorch(d['me'], device=self.device)
            else:
                self.lg.warning("Preprocessed motion embedding unavailable. "
                        "Computing on-the-fly...")
                if len(inp.shape) == 2:
                    isbatch = False
                elif len(inp.shape) == 3:
                    isbatch = True
                else:
                    raise ValueError

                if self.use_fp_for_otf_me_calc:
                    c_outp = self.test_converter(outp if isbatch else outp[None])
                    me_enc_inp = c_outp['rot_mats']
                    d['c_outp'] = c_outp
                else:
                    if 'intm_vals' in d:
                        me_inp_v = d['intm_vals']['me_inp']
                    else:
                        me_inp_v = d['intm']['me_inp']
                    me_enc_inp = torch.from_numpy(me_inp_v.astype(dtype_np)).to(self.device)
                medim = self.mc.get_raw_motion_embedding_dim()
                nf = inp.shape[-2]
                me = []
                nb = d['input'].shape[0]
                for pose_frm in range(nf):
                    if pose_frm < self.me_len:
                        me_cur = torch.randn(nb, 1, medim).to(self.device)
                    else:
                        ed = pose_frm
                        st = ed - self.me_len + 1
                        if self.me_type == 'me':
                            me_cur = self.mo_embedder_otf.embed(me_enc_inp[:, st:ed+1])
                        elif self.me_type == 'dp':
                            me_cur = self.mo_embedder_otf(me_enc_inp[:, st:ed+1])['dp'][-1]
                        else:
                            raise NotImplementedError

                        me_cur = me_cur[:, None].to(self.device)
                    assert len(me_cur.shape) == 3 and \
                        me_cur.shape[0] == nb and me_cur.shape[1] == 1 and \
                        me_cur.shape[2] == medim
                    me.append(me_cur)
                # TODO first me_len - 1 MEs should not be used for
                # quantitative evaluation
                me = torch.concat(me, dim=1).to(self.device)

            me_unperturbed = me
            if self.me_train_inp_ns != 0.0:
                me = me_unperturbed + self.me_train_inp_ns * torch.randn_like(me)

            d['raw_me_torch'] = me
            d['raw_me_unperturbed_torch'] = me_unperturbed
            #assert not me_unperturbed.requires_grad
            #assert not me.requires_grad

        if me is not None:
            if self.me_mapper is not None:
                me = self.me_mapper(me)
            if self.concat_me_to_inp:
                # We are assuming that the current input augmentations
                # do not alter the underlying 'motion embedding'
                # to be able to use precomputed MEs here;
                # otherwise they must be computed before the data is
                # split into equal-sized windows
                # because we usually don't have enough frames to compute
                # MEs for most of the window here.
                inp = torch.concat((inp, me), dim=-1)
            d['me'] = me

def get_input_types(model_config):
    from preprocessing import preprocessed_to_model_inout
    _, _, intm = preprocessed_to_model_inout(None, model_config=model_config,
            train_or_test='train', return_intermediate_values=True, test_run=True)
    input_types = intm['input_types']
    assert len(input_types) > 0
    return input_types


class MotionPrior(nn.Module):
    MODELTYPES = ['MotionCLIP', 'KLD']
    def __init__(self, modeltype, checkpoint_path, sparse_encoder,
            use_logvar=False, freeze='both'):
        super().__init__()
        import motionclip
        if modeltype not in self.MODELTYPES:
            raise ValueError("Unknown MotionPrior model type: {}".format(modeltype))
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                    "You must provide a valid checkpoint path for MotionCLIP: {}".format(
                        checkpoint_path))
        self.modeltype = modeltype

        pp = {}
        if freeze == 'none':
            pp['freeze_weights'] = False
            pp['decoder_freeze'] = False
        elif freeze == 'decoder':
            pp['freeze_weights'] = False
            pp['decoder_freeze'] = True
        elif freeze == 'both':
            pp['freeze_weights'] = True

        self.use_logvar = use_logvar
        pp['use_logvar'] = use_logvar
        if use_logvar:
            glg().info("Using logvar")

        self.model = motionclip.load_model(device=get_device(),
                sparse_encoder=sparse_encoder,
                checkpoint_path=checkpoint_path, **pp)
        self.me_len = 60
        self.me_dim = 512
        self.sparse = sparse_encoder
        self.device = get_device()

    def embed(self, x: torch.Tensor, *, with_no_grad=False, return_dict=False):
        """
            x expected to be Tensor with shape for full joints input:
                (nb x)? melen x nj x 3 x 3
            or for sparse input
                (nb x)? melen x nj x nfeats

            melen is 60 for MotionCLIP
            nj may be 3 (sparse), 22, 24 or 25

            outputs (nb x)? medim
        """
        self.model.eval()
        # with grad, may need to reduce batch size down to ~128 on RTX 2080 Ti (11G VRAM)
        with torch.no_grad() if with_no_grad else contextlib.nullcontext():
            unbatch = False
            if len(x.shape) == (4 if not self.sparse else 3):
                x = x[None]
                unbatch = True
            if len(x.shape) != (5 if not self.sparse else 4):
                raise ValueError("Invalid shape: {}".format(x.shape))

            if x.shape[1] != self.me_len:
                raise ValueError(
                        "Input size ({}) does not match expected ME length: {}".format(
                            x.shape[1], self.me_len))

            # 22 x rmats -> 25 x 6drot (full)
            if not self.sparse:
                if not (x.shape[-1] == 3 and x.shape[-2] == 3):
                    raise ValueError("Input must be rotation matrix form")
                nj = x.shape[-3]
                if nj != 25:
                    # Increase #joints to 25
                    if nj == 22:
                        y = torch.zeros(*x.shape[:-3], 25-22, 3, 3)
                        y[..., :2, :, :] = torch.eye(3)
                        x = torch.concat((x, y.to(self.device)), dim=-3)
                    elif nj == 24:
                        y = torch.zeros(*x.shape[:-3], 25-24, 3, 3)
                        x = torch.concat((x, y.to(self.device)), dim=-3)
                    else:
                        raise ValueError("Invalid #joints: {}".format(x.shape[2]))
                # 3x3 -> 6
                x = x[..., :2, :].reshape(*x.shape[:-2], -1)
            x = torch.moveaxis(x, 1, -1)
            embs = self._encode_motions(x, return_dict=return_dict)
            if return_dict:
                if unbatch:
                    raise NotImplementedError
                return embs
            else:
                return embs if not unbatch else embs[0]

    def forward(self, x):
        d = self.embed(x, return_dict=True)
        #movec = self.__movec_from_encoder_out(d)
        #d['z'] = movec
        d = self.model.decoder(d)
        return d

    def _encode_motions(self, x, *, return_dict=False):
        in_d = {'encoder_x': x,
                 'y': torch.zeros(x.shape[0], dtype=int, device=self.device),
                 'mask': self.model.lengths_to_mask(
                    torch.ones(x.shape[0], dtype=int, device=self.device) * 60)}
        d = self.model.encoder(in_d)

        if self.use_logvar:
            d['z'] = self.model.reparametrize(d)
        else:
            d['z'] = d['mu']

        if return_dict:
            in_d.update(d)
            return in_d
        else:
            return d['z']

    #def __movec_from_encoder_out(self, d):
    #    if self.modeltype == 'MotionCLIP':
    #        return d['mu']
    #    else:  # KLD
    #        std = torch.exp(0.5 * d['logvar'])
    #        e = torch.randn_like(std)
    #        return d['mu'] + e * std


# yenchenlin/nerf-pytorch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# yenchenlin/nerf-pytorch
def get_embedder(in_d, multires, max_freq_log2=None, i=0):
    if i == -1:
        return nn.Identity(), in_d

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : in_d,
                'max_freq_log2' : max_freq_log2 or (multires - 1),
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                #'add': False
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoder(nn.Module):
    def __init__(self, model_config, input_types=None):
        """
        input_types: from preprocessing.preprocessed_to_model_inout
        """
        super().__init__()
        self.mc = model_config
        self.input_types = input_types
        self.embedders = []
        self.out_dim = -1
        self.__setup_embedders(input_types)

    def __setup_embedders(self, input_types):
        self.embedders.clear()
        if not input_types:
            input_types = get_input_types(self.mc)

        out_dim = 0
        for (st, edp1), inp_t in input_types:
            embedder, od = get_embedder(in_d=edp1 - st,
                    multires=self.mc['pe_L_{}'.format(inp_t)],
                    max_freq_log2=self.mc['pe_max_freq_log2_{}'.format(inp_t)])
            self.embedders.append((((st, edp1), inp_t), embedder))
            out_dim += od
        self.out_dim = out_dim

    def forward(self, x):
        assert len(self.embedders) > 0

        out = []
        for i, (((st, edp1), _inp_t), embedder) in enumerate(self.embedders):
            out.append(embedder(x[..., st:edp1]))
        return torch.cat(out, dim=-1)


class SimpleSparseInputEncoder(nn.Module):
    def __init__(self, model_config, in_dim, **kw):
        super().__init__()
        self.mc = model_config
        dim = model_config['input_encoder_dim']
        self.dim = dim
        nets = [nn.Linear(in_dim, dim),
                nn.BatchNorm(dim) if kw.get('use_batchnorm', False) 
                    else nn.Dropout(kw.get('dropout_pct', 0.2)),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(dim, dim)]
        if model_config['simple_input_encoder_layernorm']:
            nets.append(nn.LayerNorm(dim))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net.forward(x)

    @property
    def out_dim(self):
        return self.dim


class RNNBasedNN(PoseEstimatorBase):
    def __init__(self, config):
        super().__init__(config=config)
        hidden_size = self.mc['hidden_size']
        nlayers = self.mc['rnn_layers']
        dropout = self.mc['rnn_dropout']
        self.me_after_rnn = self.mc['me_after_rnn']

        self.me_catt_lq = None
        self.me_catt_lk = None
        self.me_catt_lv = None
        self.me_catt_mask = None
        if self.me_inp:
            if self.me_inmeth == 'crossattention':
                if not self.me_after_rnn:
                    indim_nome = self.mc.input_dim - self.mc.get_motion_embedding_dim()
                else:
                    indim_nome = hidden_size
                if True:#self.me_catt_me_as_query:
                    qdim = self.mc.get_motion_embedding_dim()
                    kvdim = indim_nome
                else:
                    qdim = indim_nome
                    kvdim = self.mc.get_motion_embedding_dim()
                self.me_catt_lq = nn.Sequential(nn.GELU(), nn.Linear(qdim, qdim))
                self.me_catt_lk = nn.Sequential(nn.GELU(), nn.Linear(kvdim, kvdim))
                self.me_catt_lv = nn.Sequential(nn.GELU(), nn.Linear(kvdim, kvdim))
                self.me_catt = nn.MultiheadAttention(
                        embed_dim=qdim,
                        num_heads=self.mc['motion_embedding_catt_nheads'],
                        kdim=kvdim, vdim=kvdim,
                        batch_first=True)

        dim = self.mc.input_dim
        if self.me_inp:
            if self.me_after_rnn:
                dim -= self.mc.get_motion_embedding_dim()
            elif self.me_catt:
                dim = self.mc.get_motion_embedding_dim()

        if self.mc['positional_encoding']:
            self.pe = PositionalEncoder(model_config=self.mc)
            dim = self.pe.out_dim
        else:
            self.pe = None

        if self.mc['input_encoder'] == 'simple':
            self.inp_enc = SimpleSparseInputEncoder(model_config=self.mc, in_dim=dim)
            dim = self.inp_enc.out_dim
        else:
            self.inp_enc = None

        # NN
        self.rnn = nn.LSTM(input_size=dim, hidden_size=hidden_size,
                         num_layers=nlayers, batch_first=True, dropout=dropout)

        if self.me_inp and self.me_after_rnn:
            if self.me_catt:
                if True:#self.me_catt_me_as_query:
                    linear_in_sz = self.mc.get_motion_embedding_dim()
                else:
                    linear_in_sz = hidden_size
            else:
                linear_in_sz = hidden_size + self.mc.get_motion_embedding_dim()
        else:
            linear_in_sz = hidden_size

        m = re.match(r"(\d)linear", self.mc['net_after_rnn'])
        if not m:
            raise ValueError("Invalid 'net_after_rnn': '{}'".format(self.mc['net_after_rnn']))
        n_linear_layers = int(m.group(1))
        linear_layers = []
        linear_hidden_dim = self.mc['hidden_dim_after_rnn']
        if n_linear_layers > 1:
            for i in range(n_linear_layers - 1):
                linear_layers.extend([
                        nn.Linear(linear_in_sz if i == 0 else linear_hidden_dim,
                            linear_hidden_dim),
                        nn.Dropout(0.2), nn.GELU()])
            final_in_sz = linear_hidden_dim
        else:
            final_in_sz = linear_in_sz

        final_linear_layer = nn.Linear(final_in_sz, self.mc['output_dim'])
        if len(linear_layers) > 0:
            linear_layers.append(final_linear_layer)
            self.linear_out = nn.Sequential(*linear_layers)
        else:
            self.linear_out = final_linear_layer

        # Motion Embedding
        self.concat_me_to_inp = False

    def __add_me(self, inp, me):
        if self.me_inp:
            if self.me_catt is None:
                inp = torch.concat((inp, me), dim=-1)
            else:
                q = self.me_catt_lq(me)
                k = self.me_catt_lk(inp)
                v = self.me_catt_lv(inp)
                mask = ~torch.ones((q.shape[1], k.shape[1]), dtype=bool, device=self.device).tril()
                inp, _ = self.me_catt(q, k, v, attn_mask=mask)
        return inp

    def forward(self, x, hidden=None, me=None):
        if self.pe is not None:
            x = self.pe(x)
        if self.inp_enc is not None:
            x = self.inp_enc(x)

        x, hidden = self.rnn(x, hidden)
        if self.me_inp and self.me_after_rnn:
            x = self.__add_me(x, me)
        x = self.linear_out(x)
        return x, hidden

    def _training_init_data(self):
        hst = self.mc['hidden_state_type']
        return {
            'hst': hst,
        }

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        hst = train_init_data['hst']
        pred = train_get('pred')
        hidden_train = train_get('hidden_train')
        same_file = meta['same_file']

        # Motion Embedding
        me = meta['data'].get('me', None)
        if self.me_inp and not self.me_after_rnn:
            inp = self.__add_me(inp, me)

        # FP
        random_hidden = False
        if not same_file:
            hidden_train = None  # TODO?
            random_hidden = True
        pred, hidden_train = self(inp, hidden_train, me=me)

        # Determine next hidden state input - pointless w/ random shuffle?
        h_0, c_0 = hidden_train
        h_0, c_0 = h_0.detach(), c_0.detach()
        if hst == 'no_randomisation' or (hst == 'sometimes_randomise' and
                random.random() < (1 - self.mc['hidden_state_randomisation_ratio'])):
            hidden_train = (h_0, c_0)
        else:
            random_hidden = True
            hsrm = self.mc['hidden_state_randomisation_method']
            if hsrm == 'zero':
                hidden_train = (torch.zeros_like(h_0), torch.zeros_like(c_0))
            elif hsrm == 'xavier_uniform':
                hidden_train = (nn.init.xavier_uniform_(h_0),
                        nn.init.xavier_uniform_(c_0))
                assert hidden_train[0] is not None
            elif hsrm == 'xavier_normal':
                hidden_train = (nn.init.xavier_normal_(h_0),
                        nn.init.xavier_normal_(c_0))
                assert hidden_train[0] is not None
            else:
                raise ValueError("Unknown hidden_state_randomisation_method")

        window_sz = meta['window_sz']
        if random_hidden:
            # Evaluate last frame only (flat input assumed)
            loss_startidx = window_sz - 3
            pred_e = pred[..., loss_startidx:, :].contiguous()
            outp_e = outp[..., loss_startidx:, :]
            meta['loss_startidx'] = loss_startidx
        else:
            pred_e = pred
            outp_e = outp

        t0 = time.time()

        pred_cvt = self.convert_pred(pred_e)
        outp_cvt = self.convert_outp(outp_e)

        self.lg.debug("RNNBasedModel conversion time: %f", time.time() - t0)

        t0 = time.time()

        loss, loss_view, _, loss_detail_view = self.criterion(pred_cvt, outp_cvt)

        self.lg.debug("RNNBasedModel loss computation time: %f", time.time() - t0)

        t0 = time.time()

        if is_training:
            loss.backward()

            if self.mc['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.mc['clip_grad_norm'])


        self.lg.debug("RNNBasedModel BP time: %f", time.time() - t0)
        
        return ((loss_view, loss_detail_view),
                {
                    'pred': pred.detach(),
                    'hidden_train': hidden_train
                })

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        pred = test_get('pred')
        hidden_predict = test_get('hidden_predict')
        same_file = meta['same_file']

        # Motion Embedding
        me = meta['data'].get('me', None)
        if self.me_inp and not self.me_after_rnn:
            inp = self.__add_me(inp, me)

        random_hidden = False
        #if not same_file:
        if True:
            hidden_predict = None
            random_hidden = True

        pred, hidden_predict = self(inp, hidden_predict, me=me)
        h_0, c_0 = hidden_predict
        hidden_predict = (h_0.detach(), c_0.detach())
        pred = pred.detach()

        window_sz = meta['window_sz']
        if not is_inference and random_hidden:
            loss_startidx = window_sz - 3
            pred_e = pred[..., loss_startidx:, :].contiguous()
            outp_e = outp[..., loss_startidx:, :]
            meta['loss_startidx'] = loss_startidx

            nf = pred.shape[-2]
            meta['frames_range'] = (window_sz - 3, nf - 1)
        else:
            pred_e = pred
            outp_e = outp

        return ((pred_e.detach(), outp_e),
                {
                    'pred': pred,
                    'hidden_predict': hidden_predict,
                })


class LinearResidualBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 dropout_p=0.2):
        super().__init__()

        self.net = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(in_size, out_size),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(in_size, out_size),
                #nn.BatchNorm1d(out_size)
            )
        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=dtype_torch),
                                  requires_grad=True)

    def forward(self, x):
        return x + self.alpha * self.net(x)


class AddCoords(nn.Module):
    """
    https://github.com/walsvid/CoordConv/
    """
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.device = get_device()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32).to(self.device)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(nn.modules.conv.Conv1d):
    """
    https://github.com/walsvid/CoordConv/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H)
        output_tensor_shape: N,C_out,H_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class ConvolutionalResidualBlock(nn.Module):
    def __init__(self, seq_len,
            kernel_size=3, stride=1, padding=1,
            dropout_p=0.2):
        super().__init__()

        assert kernel_size % 2 == 1

        self.net = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.01),
                CoordConv1d(seq_len, seq_len, kernel_size=kernel_size,
                    stride=stride, padding=padding),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01),
                CoordConv1d(seq_len, seq_len, kernel_size=kernel_size,
                    stride=stride, padding=padding))

        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=dtype_torch),
                                  requires_grad=True)

    def forward(self, x):
        return x + self.alpha * self.net(x)


class VAEBasedNN(PoseEstimatorBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.seq_len = self.mc['input_seq_length']
        self.k = 3 * self.mc['resnet_kdiv3']
        self.latent_dim = self.mc['latent_dim']
        self.beta = self.mc['betavae_beta']

        self.linear_features_fixed = 256
        self.conv_features_fixed = 128

        in_njoints = self.mc['encoder_njoints']
        if in_njoints == 3:
            self.input_fullpose = False
        elif in_njoints == 22:
            self.input_fullpose = True
            if not self.mc['input_fullpose']:
                raise ValueError("'input_fullpose' must be set to true in ModelConfig")
        else:
            raise ValueError("Invalid #inputjoints: {}".format(in_njoints))

        self.in_size = self.mc['vaehmd_input_features_dim']
        if self.in_size <= 0:
            self.in_size = (3 if self.input_fullpose else 3) + self.mc.in_rot_d

        self.__init_encoder(in_njoints)
        self.__init_decoder()

        if self.mc['vaehmd_frozen_decoder']:
            pretrained_path = osp.expanduser(self.mc['vaehmd_frozen_decoder'])
            self.lg.info("Loading pretrained VAE-HMD weights from '%s'", pretrained_path)
            m_keys, unexp_keys = load_torch_model(self, pretrained_path, strict=False,
                    filter_prefix=["encoder."])
            if len(unexp_keys) != 0 and \
                any(map(lambda k: not k.startswith("encoder."), m_keys)):
                raise ValueError("Tried to load weights for VAE-HMD decoder."
                        "\nmissing_keys: {};\nunexpected_keys: {}".format(
                            m_keys, unexp_keys))
            self.lg.info("Freezing VAE-HMD decoder weights...")
            for p in self.decoder.parameters():
                p.requires_grad = False

    def __init_encoder(self, input_njoints):
        in_size = input_njoints * self.in_size
        modules = []
        if self.seq_len == 1:
            modules.append(nn.Linear(in_size, self.linear_features_fixed))
            for i in range(self.k):
                rb = LinearResidualBlock(
                        self.linear_features_fixed, self.linear_features_fixed)
                modules.append(rb)
            #modules.append(nn.LeakyReLU())
            modules.append(
                    nn.Linear(self.linear_features_fixed, 2*self.latent_dim))
        elif self.seq_len > 1:
            modules.append(nn.Linear(in_size, self.conv_features_fixed))
            for i in range(self.k // 3):
                rb = ConvolutionalResidualBlock(self.seq_len)
                modules.append(rb)
            modules.append(
                    nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

            for i in range(self.k // 3):
                rb = ConvolutionalResidualBlock(self.seq_len)
                modules.append(rb)
            modules.append(
                    nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

            modules.append(nn.Flatten())
            modules.append(nn.LayerNorm((self.seq_len // 4) * self.conv_features_fixed)) # 512

            modules.append(nn.Linear((self.seq_len // 4) * self.conv_features_fixed, # 512
                        self.linear_features_fixed))

            for i in range(self.k // 3):
                rb = LinearResidualBlock(
                        self.linear_features_fixed, self.linear_features_fixed)
                modules.append(rb)
            modules.append(
                    nn.Linear(self.linear_features_fixed, 2*self.latent_dim))
        else:
            raise ValueError("Invalid seq len: {}".format(self.seq_len))

        self.encoder = nn.Sequential(*modules)

    def __init_decoder(self):
        modules = []
        modules.append(nn.Linear(self.latent_dim, self.linear_features_fixed))
        for i in range(self.k):
            rb = LinearResidualBlock(
                    self.linear_features_fixed, self.linear_features_fixed)
            modules.append(rb)
        assert self.mc.out_rot_d == 6
        modules.append(
            nn.Linear(self.linear_features_fixed,
                self.mc.out_rot_d * self.mc.output_njoints))
        #modules.append(nn.LeakyReLU())
        #modules.append(
        #    nn.Linear(self.mc.out_rot_d * self.mc.output_njoints,
        #        self.mc.out_rot_d * self.mc.output_njoints))
        self.decoder = nn.Sequential(*modules)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        return e * std + mu

    def forward(self, x):
        """
            x: B x (nseq*nj*(xyz+inrot))
        """
        enc = self.encoder(x)
        mu = enc[..., :self.latent_dim]
        logvar = enc[..., self.latent_dim:]
        z = self.sample_z(mu, logvar)
        decoded = self.decoder(z)
        return {
            'recon': decoded,
            'mu': mu,
            'logvar': logvar
        }

    def compute_recon_loss(self, pred_r6d, outp_r6d):
        if len(pred_r6d.shape) == 2:
            pred_r6d = pred_r6d[:, None]
            outp_r6d = outp_r6d[:, None]
        pred_cvt = self.convert_pred(pred_r6d)
        outp_cvt = self.convert_outp(outp_r6d)
        return self.criterion(pred_cvt, outp_cvt)

    def compute_loss(self, gt, recon, mu, logvar):
        assert len(recon.shape) in (2, 3) and len(gt.shape) == 3
        if len(recon.shape) == 3:
            assert recon.shape[-2] == 1
            recon = recon[..., 0, :]
        recon_loss, rlview, _, rldetails = self.compute_recon_loss(gt[..., -1, :], recon)
        assert len(mu.shape) in (2, 3) and len(logvar.shape) in (2, 3)
        if len(mu.shape) == 3:
            assert mu.shape[-2] == 1
            mu = mu[..., 0, :]
        if len(logvar.shape) == 3:
            assert logvar.shape[-2] == 1
            logvar = logvar[..., 0, :]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0)

        loss = recon_loss + self.beta * kld_loss
        return loss, rlview + loss.item(), None, {'kld': kld_loss.item(), **rldetails}

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        assert len(inp.shape) == 3  # nb x nf=seqlen x 132
        assert inp.shape[1] == self.seq_len

        if True:#not self.input_fullpose:
            d = self(inp)
        else:
            d = self(outp)
        loss, lview, _, ldetails = self.compute_loss(
                gt=outp, recon=d['recon'], mu=d['mu'], logvar=d['logvar'])

        if is_training:
            loss.backward()

        return ((lview, ldetails), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        assert len(inp.shape) == 3  # nb x nf=seqlen x 132
        nb = inp.shape[0]
        data_seqlen = inp.shape[1]
        outp_nfeats = outp.shape[2]

        if data_seqlen < self.seq_len:
            raise NotImplementedError(
                ("Seq len of window ({}) must be larger than seq len this model was "
                "trained on ({}).").format(inp.shape, self.seq_len))

        loss_startidx = self.seq_len - 1
        meta['loss_startidx'] = loss_startidx
        meta['frames_range'] = (loss_startidx, data_seqlen - 1)
        pred = []
        for ed_frm in range(loss_startidx, data_seqlen):
            st_frm = ed_frm - self.seq_len + 1
            if True:#not self.input_fullpose:
                d = self(inp[:, st_frm:ed_frm+1])
            else:
                d = self(outp[:, st_frm:ed_frm+1])
            pred.append(d['recon'].reshape(nb, 1, outp_nfeats))
        pred = torch.cat(pred, dim=1)
        return ((pred.detach(), outp[:, loss_startidx:]), {})


# https://github.com/GuyTevet/MotionCLIP/src/models/architectures/transformer.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Inspired by https://github.com/GuyTevet/MotionCLIP
class MotionSpaceEncoder(nn.Module):
    MODELTYPES = ['MotionCLIP', 'Last2x', 'Last2']

    def __init__(self, input_dim, latent_dim=256, modeltype='MotionCLIP',
            ff_size=1024, nlayers=4, nheads=4, dropout=0.1, activation="gelu"):
        super().__init__()
        self.modeltype = modeltype
        if modeltype not in self.MODELTYPES:
            raise ValueError("Invalid MotionSpaceEncoder model type: {}".format(modeltype))
        self.latent_dim = latent_dim
        if modeltype == 'Last2x':
            actual_latent_dim = 2 * latent_dim
        else:
            actual_latent_dim = latent_dim

        self.input_projector = nn.Linear(input_dim, actual_latent_dim)
        self.pos_encoder = PositionalEncoding(actual_latent_dim, dropout)

        tlayer = nn.TransformerEncoderLayer(
                d_model=actual_latent_dim, nhead=nheads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(tlayer, num_layers=nlayers)

    def forward(self, x):
        """
            x: nb x nf x indim
        """
        assert len(x.shape) == 3
        nb, nframes = x.shape[:2]
        x = x.permute(1, 0, 2)
        x = self.input_projector(x)
        x = self.pos_encoder(x)
        tout = self.tencoder(x) # nf x nb x latent_dim
        if self.modeltype == 'MotionCLIP':
            mu = tout[0]
            #logvar = tout[1]
            logvar = torch.zeros_like(mu)
        elif self.modeltype == 'Last2x':
            mu = tout[-1][..., :self.latent_dim]
            logvar = tout[-1][..., self.latent_dim:]
        elif self.modeltype == 'Last2':
            mu = tout[-1]
            logvar = tout[-2]
        else:
            raise AssertionError

        return {'mu': mu, 'logvar': logvar, 'nframes': nframes}


class MotionSpaceDecoder(nn.Module):
    MODELTYPES = ['LastPose', 'LastPose_FineTune', 'LastPenultimateLayerStabilizer', 'LastPenultimateLayerConcat']
    STABILIZER_ARCHS = ['ShallowMLP', 'RNN']
    def __init__(self, latent_dim, njoints, modeltype,
            stabilizer_arch='ShallowMLP',
            dropout=0.1, activation="gelu"):
        super().__init__()
        self.modeltype = modeltype
        if modeltype not in self.MODELTYPES:
            raise ValueError("Invalid MotionSpaceDecoder model type: {}".format(modeltype))
        self.latent_dim = latent_dim
        self.njoints = njoints

        # If values change checkpoint may not load properly
        ff_size = 1024
        nlayers = 8
        nheads = 4

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        tlayer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=nheads,
                dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(tlayer, num_layers=nlayers)

        for p in self.seqTransDecoder.parameters():
            p.requires_grad = False

        if self.modeltype.startswith('LastPose'):
            self.finallayer = nn.Linear(latent_dim, 150)
            if self.modeltype != 'LastPose_FineTune':
                for p in self.finallayer.parameters():
                    p.requires_grad = False
        else:
            if stabilizer_arch == 'ShallowMLP':
                self.finallayer = nn.Sequential(
                                nn.Linear(latent_dim, 256),
                                nn.GELU(),
                                nn.Linear(256, njoints * 6))
            elif stabilizer_arch == 'RNN': # Use with LastPenultimateLayerConcat
                raise NotImplementedError

    def forward(self, d):
        z = d['z']
        nb = z.shape[0]
        assert z.shape[1] == self.latent_dim
        z = z[None]  # nb x d -> 1 x nb x d

        timequeries = torch.zeros(d['nframes'], nb, self.latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        tout = self.seqTransDecoder(tgt=timequeries, memory=z) # nf x nb x d
        tlast = tout[-1]  # nb x d

        if self.modeltype == 'LastPenultimateLayerConcat':
            #concat = torch.concat([
            #out = self.finallayer(
            raise NotImplementedError

        out = self.finallayer(tlast) # nb x (150 or 6nj)
        assert len(out.shape) == 2

        if self.modeltype.startswith('LastPose'):
            rot6d_mc = out.reshape(nb, 25, 6)[:, :self.njoints]
            rotmat = self.__motionclip_r6d2rmat(rot6d_mc)
            rot6d = rotmat[..., :3, :2]
            pe_out_1f = rot6d.reshape(nb, self.njoints * 6)
        elif self.modeltype == 'LastPenultimateLayerStabilizer':
            pe_out_1f = out

        d['pe_out'] = pe_out_1f[:, None] # nb x 1 x 6nj

        return d

    def __motionclip_r6d2rmat(self, r6d):
        """
         MotionCLIP Rot6D -> Mat
         
         from MotionCLIP/src/utils/rotation_conversions/rotation_6d_to_matrix
         """
        a1, a2 = r6d[..., :3], r6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def verify_state_dict_keys(self, missing_keys, unexpected_keys, decoder_name="decoder"):
        def _cond_mk(k):
            if self.modeltype == 'LastPose':
                return k.startswith("encoder.")
            else:
                return k.startswith("encoder.") or \
                    k.startswith("{}.finallyer.".format(decoder_name))

        def _cond_uk(k):
            AB = "{}.actionbiases".format(decoder_name)
            if self.modeltype.startswith('LastPose'):
                return k == AB
            else:
                return k in (AB, "{}.finallayer.weight".format(decoder_name),
                    "{}.finallayer.bias".format(decoder_name))
                
        if not all(map(_cond_mk, missing_keys)) and all(map(_cond_uk, unexpected_keys)):
            raise ValueError(
                "Invalid state dict:\n\tmissing keys: {}\n\t unexpected keys: {}".format(
                    missing_keys, unexpected_keys))



class MotionSpacePoseEstimator(PoseEstimatorBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.encoder = MotionSpaceEncoder(
                input_dim=self.mc.input_dim - self.mc.get_motion_embedding_dim(),
                latent_dim=512, modeltype=self.mc['msp_encoder'])
        self.decoder = MotionSpaceDecoder(latent_dim=512,
                njoints=self.mc['output_njoints'],
                modeltype=self.mc['msp_decoder'],
                stabilizer_arch=self.mc['msp_decoder_stabilizer'])

        # Load pretrained MotionCLIP decoder
        state_dict = torch.load(self.mc['motionclip_checkpoint_path_for_guidance'],
                map_location=get_device())
        for k in list(state_dict.keys()):
            if k.startswith("encoder."):
                del state_dict[k]
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        self.decoder.verify_state_dict_keys(missing_keys, unexpected_keys)
        self.train_sequentially = self.mc['msp_train_sequentially']

        # ME is used for loss
        if not self.mc['input_motion_embeddings']:
            raise ValueError("'input_motion_embeddings' must be set")
        if self.mc['input_motion_embeddings__reduced_dim'] != 0:
            raise ValueError("'input_motion_embeddings__reduced_dim' must be 0")
        if self.mc['window_size'] < 60:
            raise ValueError(
                "Window size ({}) must be gt or eq to MotionCLIP input length (60)".format(
                    self.ws))
        self.ws = self.mc['window_size']

        self.concat_me_to_inp = False
        self.me_loss_coeff = self.mc['loss_multiplier_MOTION_EMBEDDING']

    def forward(self, x):
        d = self.encoder(x)
        movec = self.__movec_from_encoder_out(d)
        d['z'] = movec
        d = self.decoder(d)
        return d

    def __movec_from_encoder_out(self, d):
        std = torch.exp(0.5 * d['logvar'])
        e = torch.randn_like(std)
        return d['mu'] + e * std

    def _training_init_data(self):
        hst = self.mc['hidden_state_type']
        return {
            'hst': hst,
        }

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        nf = inp.shape[1]
        if self.ws > nf:
            raise ValueError(
                    "Input window size is {}, but only {} frames were given".format(
                        self.ws, nf))

        if self.train_sequentially:
            preds = []
            pred_zs = []
            st, ed = (self.ws - 1, nf - 1)
            me_start_frm = st
            me_end_frm = ed
            meta['frames_range'] = (st, ed)
            for end_frm in range(st, ed+1):
                start_frm = end_frm - self.ws + 1
                inp_win = inp[:, start_frm:end_frm+1]
                pred_win_d = self(inp_win)
                preds.append(pred_win_d['pe_out'])
                pred_zs.append(pred_win_d['z'][:, None])

            pred_e = torch.cat(preds, axis=1) # nb x nf x (22*6)
            pred_z = torch.cat(pred_zs, axis=1)
            outp_e = outp[:, -(nf - self.ws + 1):]
            assert len(pred_e.shape) == len(outp_e.shape)
        else:
            rand_st = random.randint(0, nf - self.ws)
            start_frm = rand_st
            ed = rand_st + self.ws - 1
            me_start_frm = ed
            me_end_frm = ed
            inp_e = inp[:, rand_st:ed+1]

            pred_e_d = self(inp_e)
            pred_e = pred_e_d['pe_out']
            pred_z = pred_e_d['z'][:, None]
            outp_e = outp[:, [ed]]

        window_sz = meta['window_sz']

        t0 = time.time()

        pred_cvt = self.convert_pred(pred_e)
        outp_cvt = self.convert_outp(outp_e)

        self.lg.debug("MSPoseEstimator conversion time: %f", time.time() - t0)

        t0 = time.time()

        loss, loss_view, _, loss_detail_view = self.criterion(pred_cvt, outp_cvt)
        if self.me_loss_coeff > 0:
            me_inp_e = meta['data']['me'][:, me_start_frm:me_end_frm+1]
            me_loss = F.mse_loss(pred_z, me_inp_e)
            me_loss = self.me_loss_coeff * me_loss
            loss += me_loss
            loss_view += me_loss.item()
            loss_detail_view['gtme_loss'] = me_loss.item()

        self.lg.debug("MSPoseEstimator loss computation time: %f", time.time() - t0)

        t0 = time.time()

        if is_training:
            loss.backward()

            if self.mc['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.mc['clip_grad_norm'])

        self.lg.debug("MSPoseEstimator BP time: %f", time.time() - t0)
        
        return ((loss_view, loss_detail_view), {})

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        nf = inp.shape[1]
        pred = []
        st, ed = (self.ws - 1, nf - 1)
        meta['frames_range'] = (st, ed)
        for end_frm in range(st, ed+1):
            start_frm = end_frm - self.ws + 1
            inp_win = inp[:, start_frm:end_frm+1]
            pred_win_d = self(inp_win)
            pred.append(pred_win_d['pe_out'])

        pred_e = torch.cat(pred, axis=1) # nb x nf x (22*6)
        outp_e = outp[:, -(nf - self.ws + 1):]
        assert len(pred_e.shape) == len(outp_e.shape)

        pred_e.detach_()
        return ((pred_e, outp_e), {})


class AvatarPoser(PoseEstimatorBase):
    def __init__(self, config):
        super().__init__(config=config)
        input_dim = 54
        output_dim = 132
        embed_dim = 256
        num_layer = 3
        nhead = 8

        self.winsize = self.mc['ap_winsize']
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.stabilizer = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 6)
            )
        self.joint_rotation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 126)
            )
        #self.ckpt_path = self.mc['checkpoint_avatarposer']
        #if os.path.isfile(self.ckpt_path):
        #    state_d = torch.load(self.ckpt_path, map_location=self.device)
        #    self.load_state_dict(state_d)
        #else:
        #    raise ValueError("You must provide a valid checkpoint for trained "
        #            "AvatarPoser model: {}".format(self.ckpt_path))

    def forward(self, input_tensor):
        x = self.linear_embedding(input_tensor)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)[:, -1]

        global_orientation = self.stabilizer(x)
        joint_rotation = self.joint_rotation_decoder(x)
        return global_orientation, joint_rotation

    def _training_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get, is_training=True):
        raise NotImplementedError()

    def _validation_step_callback(self, inp, outp, meta, prev_step_custom_data,
            train_init_data, train_get):
        return self._training_step_callback(inp, outp, meta, prev_step_custom_data,
                train_init_data, train_get, is_training=False)

    def _testing_step_callback(self, inp, outp, meta, prev_step_custom_data,
            testing_init_data, test_get, is_inference):
        assert len(inp.shape) == 3  # nb x nf x (nj*nf)
        assert not self.me_inp
        nb, nf = inp.shape[:2]
        rootr, r21 = self(inp)

        # outp: nb x nf x (22*6)
        pred = []
        #for end_frm in tqdm(range(0, nf)):
        for end_frm in range(0, nf):
            start_frm = max(0, end_frm - self.winsize + 1)
            inp_win = inp[:, start_frm:end_frm+1]
            p_rootr_win, p_r21_win = self(inp_win)  # rootr: nb x 6, r21: nb x (21*6)
            p_win = torch.cat([p_rootr_win, p_r21_win], dim=1)
            assert len(p_win.shape) == 2 # nb x (22*6)
            pred.append(p_win)

        pred = torch.cat([w[:, None] for w in pred], axis=1) # nb x nf x (22*6)
        assert len(pred.shape) == len(outp.shape)
        assert all(s1 == s2 for (s1, s2) in zip(pred.shape, outp.shape))

        return ((pred.detach(), outp), {})


def get_nn(config):
    model_config = config['model_config']
    nn_arch = model_config['nn_architecture']
    if nn_arch == 'RNN':
        return RNNBasedNN(config=config)
    elif nn_arch == 'VAE':
        return VAEBasedNN(config=config)
    elif nn_arch == 'VAE_REF':
        return RefVAENN(config=config)
    elif nn_arch == 'FLAG':
        return FlowBasedNN(config=config)
    elif nn_arch == 'AvatarPoser':
        return AvatarPoser(config=config)
    elif nn_arch == 'MSP':
        return MotionSpacePoseEstimator(config=config)
    elif nn_arch == 'ActionRecognizer':
        return ActionRecognizer(config=config)
    else:
        raise ValueError("Unknown NN architecture: {}".format(nn_arch))


def init_model(config, model_config_path=None, append=False, cuda_index=0,
        tensorboard_logdir=None):
    if model_config_path:
        glg().info("Loading model config from %s%s...",
                model_config_path, " (a)" if append else "")
        if not append:
            mc = ModelConfig(None)
        else:
            mc = ModelConfig(config.get('model_config', None))
        mc.load_from_file(model_config_path, append=append)
    else:
        mc = ModelConfig(config.get('model_config', None))

    if not mc.get('name', None):
        if model_config_path:
            mc['name'] = os.path.splitext(os.path.basename(model_config_path))[0]
        else:
            mc['name'] = "#"

    global get_device
    get_device = functools.partial(get_device, cuda_index=cuda_index)
    set_get_device_func(get_device)

    if not tensorboard_logdir:
        tensorboard_logdir = "./allruns/{}".format(
                datetime.datetime.now().strftime("%Y%m%dT%_H%M%S"))
    mc['tensorboard_logdir'] = tensorboard_logdir

    config['model_config'] = mc


class Node:
    def __init__(self, parent=None):
        self.parent = parent

    def __repr__(self):
        return "Node : {}".format(self.parent)


cvt_nodes = {
    'rot_6d': Node(),
    'rot_mats': Node('rot_6d'),
    'rot_mats_g': Node('rot_mats'),
    'joints': Node('rot_mats'),
    'vel': Node('joints'),
    'jitter': Node('joints'),
    'acc': Node('vel'),
    'joints_rel': Node('rot_mats'),
    'vel_rel': Node('joints_rel'),
    'avel': Node('rot_mats'),
    'aacc': Node('avel'),
    'davel': Node('rot_mats')
}


def find_ancestor_cvt_nodes(node, include_self=False):
    if not isinstance(node, str):
        raise ValueError
    ancestors = [node] if include_self else []
    cn = cvt_nodes[node].parent
    while cn is not None:
        ancestors.append(cn)
        cn = cvt_nodes[cn].parent

    return ancestors


def nn_out_converter(config, targets, training, **kw):
    """
        It is assumed that the output of the NN are 3d or 6d rotations for each joint.
        Expected shape: nb x nf x (outnj*6)
    """

    mc = config['model_config']
    #if 'model' in kw:
    #    model = kw['model']
    #else:
    #    model_path = get_model_path(config, 'smpl', 'male')
    #    model = load_smpl_model(model_path,
    #            as_class=kw.get('load_smpl_model_as_class', False))

    if mc.out_rot_d != 6:
        raise NotImplementedError

    r2fc = kw.get(
            'recompute_first_two_cols_of_6d_rot', True)

    lg = glg()
    comp_targets = [targets[0]]
    ct_ancestors = [set(find_ancestor_cvt_nodes(targets[0], include_self=True))]
    lg.debug("initial comp targets: %s", comp_targets)
    for t in targets[1:]:
        ta = set(find_ancestor_cvt_nodes(t, include_self=True))
        lg.debug("iter for '%s' (anc=%s)", t, ta)
        for ci, (ct, cta) in enumerate(zip(comp_targets, ct_ancestors)):
            if ta.issubset(cta):
                break
            elif cta.issubset(ta):
                comp_targets[ci] = t
                ct_ancestors[ci] = ta
                break
        else:
            comp_targets.append(t)
            ct_ancestors.append(ta)
    lg.debug("enditer comp targets: %s", comp_targets)
    #breakpoint()
    assert all('rot_6d' in cta.union({ct}) for ct, cta in zip(comp_targets, ct_ancestors))

    converters = []
    for ct in comp_targets:
        if ct == 'rot_6d':
            continue
        converters.append(get_converter(config, 'rot_6d', ct, return_intermediates=True,
                    model=kw.get('model', None), normalise_velocities=training))

    lg = glg()

    def closure(data, ctxt=None):
        if len(data.shape) != 3:
            raise ValueError("Invalid data shape for data: {}".format(data.shape))

        batch_n_fr = tuple(data.shape[:2])
        reshp = batch_n_fr + (mc['output_njoints'], 3, 2)

        data_r6d = data.reshape(reshp)  # nb x nf x nj x 3 x 2
        conv_results = {
            'rot_6d': data_r6d
        }
        lg.debug("#conversions needed: %d", len(comp_targets))
        for cvt, ct in zip(converters, comp_targets):
            t0 = time.time()
            c, intm = cvt(data_r6d, ctxt=ctxt)
            conv_results[ct] = c
            conv_results = {**conv_results, **intm}
            lg.debug("Conversion from input to '%s' took %fs (also converted to: %s)",
                    ct, time.time() - t0, intm.keys())
        return conv_results

    return closure


def train(config, training_data_loader, validation_data_loader=None,
        checkpoints_save_dir=None, checkpoint_path=None, nepochs=None, window_sz=None):
    model_config = config['model_config']

    device = get_device()
    glg().info("Using device: %s", device)
    model = get_nn(config=config).to(device)
    glg().info("Model Summary:\n%s", str(model))
    glg().info("Start training...")
    model.start_training(training_data_loader,
            validation_data_loader=validation_data_loader,
            checkpoints_save_dir=checkpoints_save_dir,
            checkpoint_path=checkpoint_path,
            nepochs=nepochs,
            window_sz=window_sz)


def check_torch_model_keys(missing_keys, unexpected_keys, checkpoint_path=None):
    for k in missing_keys:
        #if not k.startswith('mo_embedder'):
        #    raise ValueError("Error loading checkpoint: '{}'".format(checkpoint_path))
        if 'clip_model.' not in k:
            raise ValueError(
                    "Error loading checkpoint: '{}'\nMissing key: '{}'".format(
                        checkpoint_path, k))


def test(config, test_data_loader, checkpoint_path, results_save_dir=None, window_sz=None,
        inference_callback=None, **kw):
    mc = config['model_config']
    device = get_device()
    glg().info("Using device: %s", device)
    model = get_nn(config=config).to(device)
    glg().info("Model Summary:\n%s", str(model))
    glg().info("Start testing...")
    if checkpoint_path:
        glg().info("Loading checkpoint: %s", checkpoint_path)
        missing_keys, unexpected_keys = load_torch_model(model, checkpoint_path, strict=False)
        #load_torch_model(model, checkpoint_path)
        check_torch_model_keys(
                missing_keys=missing_keys, unexpected_keys=unexpected_keys, checkpoint_path=checkpoint_path)
    else:
        glg().info("!!! No checkpoint provided !!!")
    model.start_testing(test_data_loader, results_save_dir=results_save_dir,
            window_sz=window_sz, inference_callback=inference_callback, **kw)


def analyse_losses(losses, meta, save_dir, model_config=None):
    def fmtdf(_df):
        return _df.to_html()

    losses_cols = list(losses.keys())
    meta_cols = list(filter(lambda k: k != 'windows', meta.keys()))
    losses_datas = [losses[c] for c in losses_cols]
    meta_datas = [meta[c] for c in meta_cols]
    all_datas = losses_datas + meta_datas
    l_maxlen = max(map(len, all_datas))
    for il, l in enumerate(all_datas):
        if len(l) == 0 or l_maxlen % len(l) == 0:
            l_new = []
            nw = l_maxlen // len(l) if len(l) > 0 else 0
            for litem in l:
                for _ in range(nw):
                    l_new.append(litem)
            all_datas[il] = l_new
        else:
            raise AssertionError
    assert all(len(all_datas[i]) == len(all_datas[i+1]) for i in range(len(all_datas)-1))

    df = pd.DataFrame(list(zip(*all_datas)),
            columns=losses_cols + meta_cols)

    os.makedirs(save_dir, exist_ok=True)
    analysis_html_fp = os.path.join(save_dir, "analysis.html")
    analysis_res_dir_rel = "res/"
    analysis_res_dir = os.path.join(save_dir, analysis_res_dir_rel)
    os.makedirs(analysis_res_dir, exist_ok=True)

    figw = 12

    with open(analysis_html_fp, 'w') as text_writer:
        text_writer.write("Analysis date and time: {} <br>".format(
                    datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        text_writer.write("Model Config: {} <br>".format(model_config))

        text_writer.write("<h1>Summary</h1>")
        text_writer.write(fmtdf(df.describe()))

        fig = plt.figure(figsize=(figw, round(figw * 2/3)))
        ax = fig.add_subplot()
        ax.boxplot(df[losses_cols], showfliers=False)
        ax.set_xticks(range(1, len(losses_cols)+1), losses_cols, rotation=10)
        fig_fp = os.path.join(analysis_res_dir, "summary.jpg")
        fig.savefig(fig_fp)
        
        text_writer.write("<img src=\"{}\"> <br>".format(
                    os.path.join(analysis_res_dir_rel, os.path.basename(fig_fp))))

        for m in tqdm(meta_cols):
            glg().info("Meta = %s", m)
            text_writer.write("<h2>Group by: \"{}\"</h2>".format(m))
            gb = df.groupby(m)
            text_writer.write(fmtdf(gb.mean()))
            if len(gb) < 15:
                g_desc = gb.describe()
            else:
                g_desc = None
            
            m_vals = df[m].unique()
            grps = [gb.get_group(v) for v in m_vals]

            #plots_ncols = 4
            #plots_nrows = int(math.ceil(len(losses_cols) / plots_ncols))
            #fig, axes = plt.subplots(plots_nrows, plots_ncols,
            #        figsize=(figw, figw * (plots_nrows / plots_ncols)))
            #
            #for i_l, ll in tqdm(enumerate(losses_cols), desc="Plots"):
            #    if plots_ncols == 1:
            #        ax = axes[i_l]
            #    else:
            #        ax = axes[i_l // plots_ncols, i_l % plots_ncols]
            #    ax.boxplot([g[ll] for g in grps], showfliers=False)
            #    ax.set_xticks(range(1, len(m_vals)+1), m_vals, rotation=90)
            #    ax.set_xlabel(m)
            #    ax.set_ylabel(ll)
            ##fig.tight_layout()
            #fig_fp = os.path.join(analysis_res_dir, "meta={}_tables.jpg".format(m))
            #fig.savefig(fig_fp)#, bbox_inches='tight')mc['output_njoints'])

            #text_writer.write("<img src=\"{}\"> <br>".format(
            #            os.path.join(analysis_res_dir_rel, os.path.basename(fig_fp))))

            for ll in tqdm(losses_cols, desc="Tables"):
                text_writer.write("<h4>{}</h4>".format(ll))
                if g_desc is not None:
                    if ll in g_desc:
                        text_writer.write(fmtdf(g_desc[ll]))
                text_writer.write("<br>")

            text_writer.write("<br><br>")


def analyse_test_results(test_result_paths, save_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    if not save_dir:
        raise ValueError("You must provide a valid directory in which to save results")
    else:
        os.makedirs(save_dir, exist_ok=True)

    all_losses = {}
    loss_types = OrderedDict()

    # Load all losses
    for itr, trfp in enumerate(test_result_paths):
        with open(trfp, 'rb') as f:
            test_result = pickle.load(f)

        result_lists = test_result['lists']
        #name = test_result.get('model_config', {}).get('name',
        #        "TestResult_{}".format(itr+1))
        name = trfp.replace('\\', '/').split('/')[-2]

        all_losses[name] = result_lists['losses']
        for loss_name in result_lists['losses'].keys():
            if loss_name not in loss_types:
                loss_types[loss_name] = loss_name

    # Analyse and write results to files
    html_fp = os.path.join(save_dir, "test_results_analysis.html")
    res_fn = "res/"
    res_fp = os.path.join(save_dir, res_fn)
    os.makedirs(res_fp, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()

    dataframes = {}
    with open(html_fp, 'w') as text_writer:
        for i_l, l_type in enumerate(loss_types):
            text_writer.write("<h2>{}</h2>".format(l_type))

            boxplt_inp = []
            boxplt_lbls = []
            for name, ll in all_losses.items():
                if l_type in ll:
                    boxplt_inp.append(ll[l_type])
                    boxplt_lbls.append(name)

            df = pd.DataFrame(list(zip(*boxplt_inp)), columns=boxplt_lbls)
            df_desc = df.describe()
            text_writer.write(df_desc.to_html())

            dataframes[l_type] = df
            dataframes[l_type + "_desc"] = df_desc

            ax.boxplot(boxplt_inp, showfliers=False)
            ax.set_xticks(range(1, len(boxplt_lbls)+1), boxplt_lbls, rotation=20)
            ax.set_xlabel("Test Results")
            ax.set_ylabel("Loss: {}".format(l_type))
            #ax.set_ylim([0, 0.015])

            graph_fn = "loss_{}.jpg".format(i_l+1)
            fig.savefig(os.path.join(res_fp, graph_fn))

            ax.clear()

            text_writer.write("<img src=\"{}\">".format(os.path.join(res_fn, graph_fn)))

    dfs_pkl_fp = os.path.join(save_dir, "test_results_analysis.pkl")
    with open(dfs_pkl_fp, 'wb') as f:
        pickle.dump(dataframes, f)


def model_output_joints_to_smpl_joints(outp_joints):
    if isinstance(outp_joints, torch.Tensor):
        p = torch.zeros(*outp_joints.shape[:-2], 24, 3).to(outp_joints.device)
    else:
        p = np.zeros(tuple(outp_joints.shape[:-2]) + (24, 3), dtype=dtype_np)
    # Hands' orientations are the same as those of wrists
    p[..., :22, :] = outp_joints[..., :22, :]
    p[..., [22, 23], :] = p[..., [20, 21], :]
    return p


def model_output_to_smpl_poses(outp, recompute_first_two_cols_of_6d_rot=True):
    p = np.zeros(tuple(outp.shape[:-1]) + (24, 3), dtype=dtype_np)
    if outp.shape[-1] == 22*6:  # R[:, :2]
        r6d = outp.reshape((-1, 22, 3, 2))
        p[..., :22, :] = rot_mat_to_vec(
                rot_6d_to_mat(r6d, recompute_first_two_cols=recompute_first_two_cols_of_6d_rot))
    elif outp.shape[-1] == 22*3:  # Axis-angle repr
        p[..., :22, :] = outp.reshape((-1, 22, 3))
    else:
        raise NotImplementedError

    # Hands' orientations are the same as those of wrists
    p[..., [22, 23], :] = p[..., [20, 21], :]
    return p


def model_output_to_rot_mats(outp, recompute_first_two_cols_of_6d_rot=False):
    p = np.zeros(tuple(outp.shape[:-1]) + (24, 3, 3), dtype=dtype_np)
    if outp.shape[-1] == 22*6:  # R[:, :2]
        r6d = outp.reshape((-1, 22, 3, 2))
        p[..., :22, :, :] = rot_6d_to_mat(r6d, recompute_first_two_cols=recompute_first_two_cols_of_6d_rot)
    else:
        raise NotImplementedError

    # Hands' orientations are the same as those of wrists
    p[..., [22, 23], :, :] = p[..., [20, 21], :, :]
    return p
