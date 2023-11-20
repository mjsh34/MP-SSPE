from lbs import lbs
from util import get_model_path, glg, get_device, load_smpl_model, dtype_torch, dtype_np

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import copy
import functools
import time
from collections.abc import Iterable


# https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
SMPL_JOINT_LABELS = {
        0: "Pelvis",
        1: "L_Hip",
        2: "R_Hip",
        3: "Spine1",
        4: "L_Knee",
        5: "R_Knee",
        6: "Spine2",
        7: "L_Ankle",
        8: "R_Ankle",
        9: "Spine3",
        10: "L_Foot",
        11: "R_Foot",
        12: "Neck",
        13: "L_Collar",
        14: "R_Collar",
        15: "Head",
        16: "L_Shoulder",
        17: "R_Shoulder",
        18: "L_Elbow",
        19: "R_Elbow",
        20: "L_Wrist",
        21: "R_Wrist",
        22: "L_Hand",
        23: "R_Hand",
}
SMPL_JOINT_INDICES = {v:k for k, v in SMPL_JOINT_LABELS.items()}

SMPL_LEG_JOINTS = (4, 5, 6, 7)
SMPL_FOOT_JOINTS = (10, 11)
SMPL_HEAD_JOINT = 15
SMPL_UPPER_EXTREMITY_JOINTS = (20, 21, 22, 23)
SMPL_LOWER_BODY_JOINTS = SMPL_LEG_JOINTS + (10, 11)
SMPL_UPPER_BODY_JOINTS = tuple(set(range(24)).difference((1, 2) + SMPL_LOWER_BODY_JOINTS))

# See notebooks/analyse_windows.ipynb 
ACTIONS_TOP50 = ('walk',
 'transition',
 'hand movements',
 'stand',
 'arm movements',
 'interact with/use object',
 'sit',
 't pose',
 'circular movement',
 'turn',
 'raising body part',
 'leg movements',
 'backwards movement',
 'step',
 'look',
 'stretch',
 'forward movement',
 'dance',
 'bend',
 'gesture',
 'clean something',
 'touching body part',
 'touch object',
 'sideways movement',
 'head movements',
 'perform',
 'foot movements',
 'move up/down incline',
 'jump',
 'take/pick something up',
 'knee movement',
 'swing body part',
 'place something',
 'lowering body part',
 'move something',
 'animal behavior',
 'grasp object',
 'waist movements',
 'kick',
 'stand up',
 'wave',
 'touching face',
 'exercise/training',
 'throw',
 'lean',
 'action with ball',
 'run',
 'lift something',
 'poses',
 'play sport')

IDX2ACT = ('transition', 'stand', 'walk', 'hand movements', 'turn', 'arm movements',
 'interact with/use object', 't pose', 'step', 'raising body part',
 'backwards movement', 'look', 'touch object', 'leg movements',
 'circular movement', 'forward movement', 'stretch', 'sit',
 'take/pick something up', 'place something', 'touching body part', 'bend',
 'foot movements', 'a pose', 'stand up', 'lowering body part', 'jump',
 'sideways movement', 'throw', 'move up/down incline', 'head movements',
 'action with ball', 'kick', 'gesture', 'grasp object', 'run',
 'waist movements', 'knee movement', 'move something', 'lift something',
 'lean', 'catch', 'swing body part', 'touching face', 'wave', 'poses', 'dance',
 'sports move', 'clean something', 'punch', 'jog', 'face direction',
 'exercise/training', 'greet', 'squat', 'shake', 'stumble', 'scratch',
 'play sport', 'spin', 'balance', 'crossing limbs', 'hop', 'martial art',
 'upper body movements', 'grab body part', 'kneel', 'crouch', 'knock', 'hit',
 'move back to original position', 'perform', 'animal behavior', 'crawl',
 'adjust', 'touch ground', 'bow', 'shoulder movements', 'rolling movement',
 'twist', 'clap', 'tap', 'grab person', 'spread', 'stop', 'telephone call', 'lie',
 'evade', 'play instrument', 'play catch', 'press something',
 'side to side movement', 'jump rope', 'wrist movements', 'stances', 'skip',
 'drink', 'support', 'point', 'sway', 'sneak', 'wait', 'rub', 'cartwheel',
 'misc. activities', 'dribble', 'protect', 'limp', 'to lower a body part',
 'shuffle', 'lunge', 'swim', 'flap', 'communicate (vocalise)', 'misc. action',
 'duck', 'salute', 'open something', 'misc. abstract action',
 'sudden movement', 'slide', 'shrug', 'leap', 'move misc. body part', 'trip',
 'get injured', 'relax', 'style hair', 'flip', 'jumping jacks', 'tie', 'swipe',
 'march', 'interact with rope', 'yoga', 'mix', 'stagger', 'mime', 'draw', 'eat',
 'pray', 'wiggle', 'fall', 'hug', 'bump', 'shivering', 'clasp hands',
 'inward motion', 'stroke', 'golf', 'zombie', 'stomp', 'search',
 'give something', 'check', 'release', 'wash', 'rocking movement', 'pose',
 'listen', 'strafe', 'dip', 'wobble', 'close something', 'aim', 'cry',
 'drunken behavior', 'prepare', 'chicken dance', 'yawn', 'operate interface',
 'flail arms', 'cough', 'hurry', 'fish', 'juggle', 'fight', 'shave', 'despair',
 'handstand', 'write', 'gain', 'sign', 'hang', 'drive', 'waddle', 'celebrate',
 'remove', 'unknown', 'sneeze', 'fire gun', 'cut', 'headstand', 'worry', 'chop',
 'pat', 'skate', 'hiccup', 'uncross', 'lick', 'moonwalk', 'feet movements',
 'steady', 'laugh', 'backwards', 'disagree', 'think', 'charge', 'navigate',
 'smell', 'stick', 'fidget', 'noisy labels', 'zip/unzip', 'vomit', 'dive',
 'cower', 'read', 'endure', 'shout', 'lead', 'curtsy',
 'bartender behavior series', 'glide', 'play', 'reveal', 'admire',
 'tentative movements', 'slash gesture series', 'plant feet', 'plead',
 'excite', 'try', 'conduct', 'tiptoe')


def totorch(ndarray, do_copy=False, device=None):
    if do_copy:
        ndarray = copy.deepcopy(ndarray)
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    else:
        return torch.from_numpy(ndarray).to(dtype_torch).to(device or get_device())


def set_get_device_func(func):
    global get_device
    get_device = func

def get_k(modeltype):
    try:
        return {'smpl': 23, 'smplh': 51, 'smplx': 53}[modeltype]
    except KeyError:
        raise ValueError


def model_type_from_njoints(njoints):
    try:
        return {24: 'smpl', 52: 'smplh', 54: 'smplx'}[njoints]
    except KeyError:
        raise ValueError


def correct_amass_rotation(poses, tp='axisangles', uncorrect=False):
    r = Rotation.from_euler('zx', (-90, 270), degrees=True)
    if uncorrect:
        r = r.inv()
    if tp == 'axisangles':
        poses[..., 0, :] = (r * Rotation.from_rotvec(poses[..., 0, :])).as_rotvec()
    elif tp == 'translations':
        transls_flatb = poses.reshape((-1, 3))
        transls_flatb_rotated = r.apply(transls_flatb)
        poses[:] = transls_flatb_rotated.reshape(poses.shape)
    else:
        raise ValueError

def rotate_y_smpl(poses, deg, tp='axisangles'):
    r = Rotation.from_euler('y', deg, degrees=True)
    if tp == 'axisangles':
        poses[..., 0, :] = (r * Rotation.from_rotvec(poses[..., 0, :])).as_rotvec()
    elif tp == 'translations':
        transls_flatb = poses.reshape((-1, 3))
        transls_flatb_rotated = r.apply(transls_flatb)
        poses[:] = transls_flatb_rotated.reshape(poses.shape)

def adjust_framerate(og_framerate, target_framerate, seq):
    if target_framerate > og_framerate:
        raise NotImplementedError("Upsampling is not supported")

    downsample_factor = round(og_framerate / target_framerate)
    return seq[::downsample_factor], og_framerate / downsample_factor


def compute_joints(config, pose=None, transl=None, beta=None, beta_size=None,
        rot_mats=None, device=None, model=None, model_path=None, model_type=None, gender=None,
        njoints_cutoff=-1, joints_rest=None, pose_reshape_ok=True,
        relative=False, return_dict=False, return_torch_objects=False):
    """
    If model is not provided, it will be loaded from model_path,
    which will be determined by model_type and gender if it is not provided.
    pose must have shape (njoints x 3) or (nbatch x njoints x 3)

    # For batch only
    If joints_rest is provided, joint positions in rest pose
    will not have to be computed from model['v_template'] and model['J_regressor'],
    speeding up computation.
    """
    if rot_mats is None:
        if len(pose.shape) == 2:
            is_batch = False
        elif len(pose.shape) == 3:
            is_batch = True
        else:
            raise ValueError
    else:
        if len(rot_mats.shape) == 3:
            is_batch = False
        elif len(rot_mats.shape) == 4:
            is_batch = True
        else:
            raise ValueError

    if njoints_cutoff > 0:
        if njoints_cutoff > (pose if pose is not None else rot_mats).shape[1]:
            raise ValueError
        njoints = njoints_cutoff
        pose = pose[:, :njoints]
    else:
        njoints = (pose if pose is not None else rot_mats).shape[1]

    beta_size = beta_size or config['model_config']['beta_size']

    if model is None and model_path is None:
        if njoints >= 1 + get_k('smpl'):
            model_type = model_type_from_njoints(njoints)
        else:
            model_type = 'smpl'
        model_path = get_model_path(config, model_type, gender or 'male')

    if not is_batch:
        if pose is None:
            raise NotImplementedError("SMPL pose is required for non-batch joint computation")

        glg().debug("Computing joints via SMPLModel...")

        from smpl_np import SMPLModel
        if model is None:
            model = load_smpl_model(model_path, as_class=True, k=njoints-1)
        if beta is None:
            beta = np.zeros(beta_size)
        model.set_params(pose=pose, beta=beta, trans=transl, skip_verts_update=True)
        joints = model.pose_joints()
        if not return_dict:
            return joints
        else:
            return {
                'joints': joints,
                'model': model
            }
    else:  # batch
        if model is None:
            model = load_smpl_model(model_path)

        nbatch = (pose if pose is not None else rot_mats).shape[0]
        if beta is None:
            beta = np.zeros((nbatch, beta_size), dtype=dtype_np)
                    #dtype=np.float64 if 
                    #(pose if pose is not None else rot_mats).dtype in [torch.float64, np.float64]
                    #else np.float32)

        shapedirs = np.array(model['shapedirs'][:, :, :beta_size])
        posedirs = model['posedirs'].reshape([model['posedirs'].shape[0] * 3, -1]).T
        parents = model['kintree_table'].astype(np.int32)[0]
        weights = model['weights']
        if joints_rest is not None:
            v_template = None
            J_regressor = None
        else:
            v_template = model['v_template'][np.newaxis]
            J_regressor = model['J_regressor']
            if not isinstance(J_regressor, np.ndarray):
                J_regressor = J_regressor.todense()

        parents = parents[:njoints]
        weights = weights[:, :njoints]
        if J_regressor is not None:
            J_regressor = J_regressor[:njoints]

        glg().debug("Computing joints via lbs(...)...")
        tt = functools.partial(totorch, device=get_device())
        _verts, J, J_transformed, rot_mats, rot_mats_g = \
                lbs(
            betas=tt(beta), pose=tt(pose if rot_mats is None else rot_mats),
            pose2rot=rot_mats is None,
            shapedirs=tt(shapedirs), posedirs=tt(posedirs),
            parents=tt(parents).long(),
            lbs_weights=tt(weights),
            
            v_template=tt(v_template) if v_template is not None else None,
            J_regressor=tt(J_regressor) if J_regressor is not None else None,
            joints_rest=tt(joints_rest) if joints_rest is not None else None,
            pose_reshape_ok=pose_reshape_ok, compute_joints_only=True)

        if return_torch_objects:
            joints = J_transformed
        else:
            joints = J_transformed.detach().cpu().numpy()
            rot_mats = rot_mats.detach().cpu().numpy()
            rot_mats_g = rot_mats_g.detach().cpu().numpy()

        if relative:
            parents1 = parents.copy()
            parents1[parents1 < 0] = 0
            par_joints = joints[..., parents1, :]
            joints = joints - par_joints

        if transl is not None:
            raise NotImplementedError("TODO")

        if not return_dict:
            return joints
        else:
            return {
                'joints': joints,
                'rot_mats': rot_mats,
                'rot_mats_g': rot_mats_g,
                'joints_rest': J.detach(),
                'model': model,
                'kintree': parents
            }


def recover_root_transform(start_pos, start_index, rel_rmats, kintree, *,
        joints=None,
        start_rmat=None, joints_rest=None):
    """
        start_pos: ... x 3
        start_index: index of the starting joint in the kinematic tree
        rel_rmats: RMAT OF THE ROOT JOINT IS ASSUMED TO BE GLOBAL; ... x nj x 3 x 3
        joints: posed joint positions; ... x nj x 3
        If joints is provided, only the starting POSITION will be used to find
        root transformation.

        Otherwise, if start_rmat, rel_rmats, & joints_rest are provided,
        IK will be performed starting from start_pose rotated by start_rmat.
        start_pos: ... x 3
        start_rmat: ... x 3 x 3
    """
    is_torch = isinstance(rel_rmats, torch.Tensor)
    batch_dims = tuple(rel_rmats.shape[:-3])

    if joints is not None:
        if not is_torch:
            rel_rmats = torch.FloatTensor(rel_rmats)
            joints = torch.FloatTensor(joints)
        joints_anchored = joints.clone()
        joints_anchored -= joints_anchored[..., [start_index], :]
        joints_anchored += start_pos[..., None, :]
        root_tfm = torch.zeros(batch_dims + (4, 4))
        root_tfm[..., :3, :3] = rel_rmats[..., 0, :, :]
        root_tfm[..., :3, 3] = joints_anchored[..., 0, :]
        root_tfm[..., 3, 3] = 1.0
    else:
        if start_rmat is None or rel_rmats is None or joints_rest is None:
            raise ValueError
        #if joints is None or kintree is None:
        #    joints_d = compute_joints(config, rot_mats=rel_rmats)
        #    joints = joints_d['joints']
        #    kintree = joints_d['kintree']
        if not is_torch:
            rel_rmats = torch.FloatTensor(rel_rmats)
            joints_rest = torch.FloatTensor(joints_rest)
            start_pos = torch.FloatTensor(start_pos)
            start_rmat = torch.FloatTensor(start_rmat)

        joints_rel = joints_rest.clone()
        joints_rel[..., 1:, :] -= joints_rest[..., kintree[1:], :]

        i = start_index
        root2e = torch.zeros(batch_dims + (4, 4))
        root2e[..., :, :] = torch.eye(4)

        while i > 0:
            R = torch.zeros(batch_dims + (4, 4))
            R[..., :3, :3] = rel_rmats[..., i, :, :]
            R[..., :3, 3] = joints_rel[..., i, :]
            R[..., 3, 3] = 1.0
            root2e = R @ root2e
            i = kintree[i]
        e2root = torch.zeros(batch_dims + (4, 4))
        e2root[..., 3, 3] = 1.0
        e2root_rot = torch.transpose(root2e[..., :3, :3], -1, -2)
        e2root[..., :3, :3] = e2root_rot
        e2root[..., :3, 3] = (-e2root_rot @ root2e[..., :3, [3]])[..., 0]
        #assert torch.isclose(e2root, torch.linalg.inv(root2e)).all()

        start_tfm = torch.zeros(batch_dims + (4, 4))
        start_tfm[..., :3, :3] = start_rmat
        start_tfm[..., :3, 3] = start_pos
        start_tfm[..., 3, 3] = 1.0
        root_tfm = start_tfm @ e2root

    return root_tfm if is_torch else root_tfm.cpu().numpy()


def normalise(v):
    """
    Normalise vector(s) in last axis
    """
    if isinstance(v, torch.Tensor):
        t = torch.div(v, torch.linalg.vector_norm(v, dim=-1)[..., None])
        t[t != t] = 0  # remove NaN
        return t
    else:
        return v / np.expand_dims(np.linalg.norm(v, axis=-1), axis=-1)


def rot_6d_to_mat(r6d, recompute_first_two_cols=True):
    """
    r6d expected to have shape (... x 3 x 2)
    """
    if isinstance(r6d, torch.Tensor):
        rotmats = torch.zeros(tuple(r6d.shape[:-2]) + (3, 3),
                dtype=dtype_torch).to(r6d.device)
        if recompute_first_two_cols:
            rotmats[..., 0] = normalise(r6d[..., 0].clone())
            r6d1 = r6d[..., 1].clone()
            rm0 = rotmats[..., 0].clone()
            rotmats[..., 1] = normalise(
                    r6d1 - (torch.sum(rm0 * r6d1, dim=-1)[..., None] * rm0))
            #rotmats[..., 1] = normalise(
            #        r6d[...,1].clone() - (torch.sum(rotmats[...,0].clone() * r6d[...,1].clone(), dim=-1)[..., None] * rotmats[...,0].clone()))
        else:
            rotmats[..., [0, 1]] = r6d[..., [0, 1]]
        rotmats_d = rotmats.detach()
        rotmats[..., 2] = torch.linalg.cross(rotmats_d[..., 0], rotmats_d[..., 1], dim=-1)
    else:
        rotmats = np.zeros(tuple(r6d.shape[:-2]) + (3, 3), dtype=r6d.dtype)
        if recompute_first_two_cols:
            rotmats[..., 0] = normalise(r6d[..., 0])
            rotmats[..., 1] = normalise(
                r6d[..., 1] - 
                np.expand_dims(np.sum(rotmats[..., 0] * r6d[..., 1], axis=-1), axis=-1) * rotmats[..., 0])
        else:
            rotmats[..., [0, 1]] = r6d[..., [0, 1]]
        rotmats[..., 2] = np.cross(rotmats[..., 0], rotmats[..., 1])
    return rotmats


def rot_mat_to_vec(rotmats):
    og_shape = None
    if len(rotmats.shape) > 3:
        og_shape = rotmats.shape
        rotmats = rotmats.reshape((-1, 3, 3))

    rvec = Rotation.from_matrix(rotmats).as_rotvec()
    if og_shape is not None:
        rvec = rvec.reshape(tuple(og_shape[:-2]) + (3,))
    return rvec


def joints_to_vel(data, fps=1, initial='skip'):
    """
        Accepts inputs of form (nb x)* nf x nj x 3
    """
    if isinstance(data, torch.Tensor):
        if initial == 'skip':
            return fps * torch.diff(data, 1, dim=-3)
        elif initial == 'rest':
            return fps * torch.diff(data, 1, dim=-3, prepend=data[..., [0], :, :])
    else:
        if initial == 'skip':
            return fps * np.diff(data, 1, axis=-3)
        elif initial == 'rest':
            return fps * np.diff(data, 1, axis=-3, prepend=data[..., [0], :, :])
    raise ValueError


def vel_to_acc(data, fps=1):
    return joints_to_vel(data, fps)


def joints_to_jitter(j, fps=1):
    """ Modified from https://github.com/Xinyu-Yi/PIP 
        MAY BE SUBJECT TO GPLv3
        m/s^3
    """
    return ((j[..., 3:, :, :] - 3 * j[..., 2:-1, :, :] + 3 * j[..., 1:-2, :, :]
                - j[..., :-3, :, :]) * (fps ** 3)).norm(dim=-1)


def compute_foot_slides(j, thresh=None, fps=1):
    fj_lr = [SMPL_JOINT_INDICES['L_Foot'], SMPL_JOINT_INDICES['R_Foot']]
    ft_horiz = j[..., fj_lr, :][..., [0, 2]] # ... x nf x (l, r) x (x, z)
    ft_vertical = j[..., fj_lr, :][..., [1]] # ... x nf x (l, r) x (y)
    ft_vel_horiz = joints_to_vel(ft_horiz, fps=fps, initial='skip') # ... x nf-1 x (l, r) x (x, z)
    # ... x nf-1 x (l, r)
    if isinstance(ft_vel_horiz, torch.Tensor):
        ft_spd_horiz = ft_vel_horiz.norm(dim=-1)
    else:
        ft_spd_horiz = np.linalg.norm(ft_vel_horiz, axis=-1)

    thresharg_isiter = isinstance(thresh, Iterable)
    thresh = thresh if thresharg_isiter else [thresh]
    fs_all = []
    for th in thresh:
        fs_I = ft_vertical[..., 1:, :, 0] < th # ... x nf-1 x (l, r)
        fs_freq_lr = fs_I.sum(-2) / fs_I.shape[-2] # ... x (l, r)
        fs_freq = fs_freq_lr.sum(-1) # ...
        fs_spd = ft_spd_horiz * fs_I # ... x nf-1 x (l, r)
        fs_spd_avg = fs_spd.mean(-2).sum(-1) # ...
        fs_spd_exp = (2 - (2 ** (ft_vertical[..., 1:, :, 0] / th))) * fs_I # ... x nf-1 x (l, r)
        fs_spd_exp_avg = fs_spd_exp.mean(-2).sum(-1) # ...
        fs_all.append({
                'freq_l': fs_freq_lr[..., 0], # ... x 1
                'freq_r': fs_freq_lr[..., 1], # ... x 1
                'freq': fs_freq, # ... x 1
                'speed_all': fs_spd, # ... x nf-1 x 2
                'speed': fs_spd_avg, # ... x 1
                'speed_exp_all': fs_spd_exp, # ... x nf-1 x 2
                'speed_exp': fs_spd_exp_avg # ... x 1
                })

    if thresharg_isiter:
        return fs_all
    else:
        return fs_all[0]


def rot_mats_to_davel(data, fps):
    if isinstance(data, torch.Tensor):
        return fps * (data - torch.roll(data, 1, dims=-4))[..., 1:, :, :, :]
    else:
        return fps * (data - axis.roll(data, 1, axis=-4))[..., 1:, :, :, :]


def estimate_velocity_humor(data_seq, h):
    '''
    Modified davrempe/humor/humor/scripts/process_amass_data.py

    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq


def estimate_angular_velocity_humor(rot_seq, h):
    '''
    Modified davrempe/humor/humor/scripts/process_amass_data.py

    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    istorch = isinstance(rotmats, torch.Tensor)

    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity_humor(rot_seq, h)
    R = rot_seq[1:-1]
    RT = torch.transpose(R, -1, -2) if istorch else np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = dRdt @ RT

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w


def angular_velocity_from_joints_rot_mats(rotmats, fps=1, avel_type='delta', initial='skip'):
    """
        Accepts inputs of form (nb x)* nf x nj x 3 x 3 
    """
    if isinstance(rotmats, torch.Tensor):
        if avel_type == 'delta':
            rm_s = torch.roll(rotmats, 1, dims=-4)
            # Initial rest
            rm_s[..., 0, :, :, :] = rotmats[..., 0, :, :, :]
            avel = rotmats @ torch.transpose(rm_s, -1, -2)
        elif avel_type == 'humor':
            avel = torch.swapaxes(
              estimate_angular_velocity_humor(torch.swapaxes(rotmats, 0, -4), 1 / fps), -4, 0)
        else:
            raise ValueError
    else:
        if avel_type == 'delta':
            rm_s = np.roll(rotmats, 1, axis=-4)
            # Initial rest
            rm_s[..., 0, :, :, :] = rotmats[..., 0, :, :, :]
            avel = rotmats @ np.moveaxis(rm_s, -1, -2)
        elif avel_type  == 'humor':
            avel = np.swapaxes(
              estimate_angular_velocity_humor(np.swapaxes(rotmats, 0, -4), 1 / fps), -4, 0)
        else:
            raise ValueError

    if initial == 'skip':
        return fps * avel[..., 1:, :, :, :]
    elif initial == 'rest':
        return fps * avel
    else:
        raise ValueError


def get_converter(config, src_type, dest_type, return_intermediates=False, model=None, **kw):
    """
        data expected to have shape (nb*) x nf * nj * ...

    """

    if kw.get('normalise_velocities', False):
        fps = 1
    else:
        fps = config['model_config']['target_framerate']
    rf2c = kw.get('recompute_first_two_cols', True)

    if model is None:
        print("get_converter - no SMPL model provided; loading one...")

        model_path = get_model_path(config, 'smpl', 'male')
        model = load_smpl_model(model_path, as_class=kw.get('load_smpl_model_as_class', False))

    lg = glg()

    def _convert(data, st, dt, rdepth=0, intm=None, ctxt=None):
        def _cerr():
            raise NotImplementedError("Unknown conversion {} -> {}".format(st, dt))

        if rdepth > 10:
            raise RuntimeError("Invalid conversion from {} -> {} (recursion_depth={})"
                    .format(src_type, dest_type, recursion_depth))
        if intm is None:
            intm = {}

        out = None

        lg.debug("get_converter-_convert %s (%s) ->%s (d=%d, r2fc=%s, kw=%s)",
                st, data.shape, dt, rdepth, rf2c, kw)
        t0 = time.time()

        if st == 'rot_6d':
            rmats = rot_6d_to_mat(data, recompute_first_two_cols=rf2c)
            if dt == 'rot_mats':
                out = rmats
            else:
                intm['rot_mats'] = rmats
                out, intm = _convert(rmats, 'rot_mats', dt, rdepth=rdepth+1, intm=intm,
                        ctxt=ctxt)
        elif st == 'rot_mats':
            if dt in ('rot_mats_g', 'joints', 'vel', 'acc', 'jitter') or \
                dt in ('joints_rel', 'vel_rel'):
                isrel = dt.endswith("_rel")
                joints_lbl = "joints" + ("_rel" if isrel else "")
                t1 = time.time()
                if len(data.shape) == 4: # nf x nj x 3 x 3
                    batch_n_fr = None
                    reshp = None
                elif len(data.shape) == 5: # nb x nf x nj x 3 x 3
                    batch_n_fr = tuple(data.shape[:2])
                    reshp = (-1,) + tuple(data.shape[2:])
                else:
                    raise ValueError("Invalid rot_mat shape: {}".format(data.shape))

                if reshp is not None:
                    data = data.reshape(reshp)

                joints_d = compute_joints(config, rot_mats=data, model=model,
                        relative=isrel, beta=ctxt.get('beta', None) if ctxt else None,
                        return_dict=True,
                        return_torch_objects=isinstance(data, torch.Tensor))
                joints = joints_d['joints']
                lg.debug("get_converter-_convert rmats (%s)->%s (%s) took %fs",
                        data.shape, joints_lbl, joints.shape, time.time() - t1)
                intm['kintree'] = joints_d['kintree']

                if dt != 'rot_mats_g' and reshp is not None:
                    joints = joints.reshape(batch_n_fr + joints.shape[1:])

                if dt == 'rot_mats_g':
                    out = joints_d['rot_mats_g']
                elif dt == joints_lbl:
                    out = joints
                else:
                    intm[joints_lbl] = joints
                    out, intm = _convert(joints, joints_lbl, dt, rdepth=rdepth+1, intm=intm)
            elif dt in ('avel', 'aacc'):
                avel = angular_velocity_from_joints_rot_mats(data, fps)
                if dt == 'avel':
                    out = avel
                else:
                    intm['avel'] = avel
                    out, intm = _convert(avel, 'avel', dt, rdepth=rdepth+1, intm=intm,
                            ctxt=ctxt)
            elif dt == 'davel':
                out = rot_mats_to_davel(data, fps)
            else:
                _cerr()
        elif st in ('joints', 'joints_rel'):
            isrel = st.endswith('_rel')
            if dt == 'jitter':
                out = joints_to_jitter(data, fps)
            else:
                vel_lbl = "vel" + ("_rel" if isrel else "")
                vel = joints_to_vel(data, fps)
                if dt == vel_lbl:
                    out = vel
                else:
                    intm[vel_lbl] = vel
                    out, intm = _convert(vel, vel_lbl, dt, rdepth=rdepth+1, intm=intm,
                            ctxt=ctxt)
        elif st == 'vel':
            acc = vel_to_acc(data, fps)
            if dt == 'acc':
                out = acc
            else:
                _cerr()
        elif st == 'avel':
            aacc = angular_velocity_from_joints_rot_mats(data, fps)
            if dt == 'aacc':
                out = aacc
            else:
                _cerr()
        else:
            _cerr()

        lg.debug("get_converter-_convert conversion (%s->%s) took %fs",
                st, dt, time.time() - t0)

        return out, intm

    def wrapper_f(data, ctxt=None):
        c, intm = _convert(data, src_type, dest_type, ctxt=ctxt)
        if return_intermediates:
            return c, intm
        else:
            return c

    return wrapper_f

