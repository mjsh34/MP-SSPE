from util import ConfigBase, get_device, glg, get_root_dir, dtype_np, dtype_torch
from data_wrangling import totorch, rot_mat_to_vec, rot_6d_to_mat, get_model_path, \
     compute_joints, compute_foot_slides, get_converter, set_get_device_func, \
     recover_root_transform, \
     SMPL_UPPER_BODY_JOINTS, SMPL_LOWER_BODY_JOINTS, SMPL_LEG_JOINTS, SMPL_JOINT_INDICES
from model import BaseNN

import torch
from torch import nn
from tqdm import tqdm

import os
import math
import functools


class MotionPriorBase(BaseNN):
    def __init__(self, config, cvt_kws=None):
        super().__init__()

        self.name = "MP_{}".format(self.mc['name'])
