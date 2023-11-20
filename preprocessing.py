from util import glg, get_root_dir, get_model_path, dtype_np, dtype_torch
from data_wrangling import totorch, lbs, correct_amass_rotation, rotate_y_smpl, \
     compute_joints, adjust_framerate, get_k, \
     joints_to_vel, angular_velocity_from_joints_rot_mats
from data_loading import load_body_datas

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

import os
import os.path as osp
import datetime
import traceback
from bdb import BdbQuit
import math
import gc
import json
from collections import defaultdict, Counter
from operator import itemgetter


def prepare(config, amass_fp, **kw):
    """
    Preprocess one AMASS animation file (not batch) such that the result is fit to be fed to a dataloader
    This function must perform preprocess(...) 'on-the-fly'
    """
    model_config = config['model_config']
    with np.load(amass_fp) as bdata:
        poses = bdata['poses']
        transl = bdata['trans']
        fr = int(bdata['mocap_framerate'])
        target_fr = model_config['target_framerate'] or -1
        nfrms = poses.shape[0]
        if target_fr > 0 and fr > target_fr:
            poses, _ = adjust_framerate(fr, target_fr, poses)
            transl, _  = adjust_framerate(fr, target_fr, transl)
            nfrms = poses.shape[0]
        poses = poses.reshape((nfrms, -1, 3))

        if model_config['do_correct_amass_rotation']:
            correct_amass_rotation(poses)
            correct_amass_rotation(transl, 'translations')

        if kw.get('rotate_y', model_config.get('rotate_y', None)):
            rotangle_deg = kw.get('rotate_y', model_config.get('rotate_y'))
            assert rotangle_deg is not None
            glg().info("Model will be rotated {} degrees about the y axis.".format(
                        rotangle_deg))
            rotate_y_smpl(poses, rotangle_deg)
            rotate_y_smpl(transl, rotangle_deg, 'translations')

        if model_config['normalise_shape']:
            betas = np.zeros((nfrms, model_config['beta_size']), dtype=poses.dtype)
        else:
            assert len(bdata['betas'].shape) == 1
            betas = np.repeat(bdata['betas'][:model_config['beta_size']][np.newaxis],
                    nfrms, axis=0)

        joints_d = compute_joints(config, poses, beta=betas,
            njoints_cutoff=model_config['njoints_cutoff'], return_dict=True)

        dtransls = np.diff(transl, axis=0, prepend=0)

        joints3 = joints_d['joints'][... , model_config['input_joints'], :]
        joints22 = joints_d['joints'][..., :22, :]

        joints3_g = np.repeat(transl[..., np.newaxis, :], 3, axis=-2) + joints3
        joints22_g = np.repeat(transl[..., np.newaxis, :], 22, axis=-2) + joints22

        FEET_JOINTS = [10, 11]
        foot_min_y = np.min(joints22_g[:, FEET_JOINTS, 1])
        joints22_g_grounded = joints22_g.copy()
        joints22_g_grounded[:, :, 1] -= foot_min_y
        assert math.isclose(0.0, joints22_g_grounded[:, FEET_JOINTS, 1].min())

        rot_mats = joints_d['rot_mats']
        rot_mats_g = joints_d['rot_mats_g']

        return {
            'joints22': joints22,
            'joints22_g': joints22_g,
            'joints22_g_grounded': joints22_g_grounded,
            'rot_mats': rot_mats,
            'rot_mats_g': rot_mats_g,
            'dtransls': dtransls,
            'path': amass_fp,
            'gender': parse_gender(bdata['gender']),
            'nframes': nfrms
        }


def preprocess(config, body_datas=None, batch_processing=True,
        n_body_datas_per_batch=64, save_in_batch=False,
        sort_body_datas_by_length=False, debug=False, save_dir=None):
    """
    If n_body_datas_per_batch > 1 SMPL models for bodies in each body data batch
    must be of same type (e.g., SMPL+H).
    """
    model_config = config['model_config']

    njoints_cutoff = model_config['njoints_cutoff']
    do_correct_amass_rotation = model_config['do_correct_amass_rotation']
    do_rotate_y = model_config.get('rotate_y', None)
    y_rot_deg = model_config.get('rotate_y', None)
    normalise_shape = model_config['normalise_shape']
    normalise_framerates = (model_config['target_framerate'] or -1) > 0

    body_datas_root_dir = config['body_data_root_directory']
    if body_datas is None:
        glg().info("Loading body data infos...")
        body_datas = load_body_datas(body_datas_root_dir)
        glg().info("Loaded %d body data\n", len(body_datas))

    if sort_body_datas_by_length:
        body_datas = sorted(body_datas, key=lambda i: i['n_frms'] / i['framerate'])

    preprocessing_root_dir = get_root_dir(config, 'preprocessing')
    if not save_dir:
        glg().info("Preprocessed files will be saved to: %s", preprocessing_root_dir)
        os.makedirs(preprocessing_root_dir, exist_ok=True)
    else:
        #if os.path.isdir(save_dir):
        #    print("Removing save_dir: {} because it exists!".format(save_dir))
        #    shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    glg().info("Begin writing joints data")
    nerrs = 0
    models_cached = {}
    batch_idx = 0
    # (mini) batches
    cur_bdatas_batch = []
    cur_bdatas_batch_processed = False
    joints_rest = None
    start_dt = datetime.datetime.now()
    lg = glg()
    BETA_SZ = model_config['beta_size']
    target_framerate = model_config['target_framerate']
    INPUT_JOINTS = model_config['input_joints']
    FEET_JOINTS = [10, 11]
    nbdatas_total = len(body_datas)
    for ibdata, bdata_info in enumerate(body_datas):
        bdata_rel_path = os.path.relpath(bdata_info['path'], body_datas_root_dir)
        if not save_in_batch:
            save_path = \
                    os.path.splitext(os.path.join(preprocessing_root_dir if not save_dir else save_dir, bdata_rel_path))[0] \
                    + ".npz"
        else:
            batch_fname = "batch_{:04}-{:04}.npz".format(n_body_datas_per_batch * batch_idx,
                        n_body_datas_per_batch * (batch_idx+1))
            if not save_dir:
                batches_dirname = "batches_{}".format(start_dt.strftime("%Y%m%dT%H%M%S"))
                save_path = os.path.join(preprocessing_root_dir, batches_dirname, batch_fname)
            else:
                save_path = os.path.join(save_dir, batch_fname)

        lg.info("[{}/{} ({:2.02f}%; {} err(s))] {}: {} ({})\n\t{}\n--> {}".format(
            ibdata+1, len(body_datas), 100 * ibdata / nbdatas_total, nerrs,
            bdata_info['dataset_shortname'],  bdata_info['subject'],
            bdata_info['dataset_type'], bdata_rel_path, save_path))

        if not save_in_batch and os.path.isfile(save_path):
            lg.info("File exists; skipping")
        else:
            try:
                with np.load(bdata_info['path']) as bdata:
                    gender = parse_gender(bdata['gender'])
                    framerate = framerate_og = int(bdata.get('mocap_framerate', -1))
                    model_path = get_model_path(config, bdata_info['dataset_type'], gender)
                    n_frms = n_frms_og = bdata['poses'].shape[0]
                    k = k_og = get_k(bdata_info['dataset_type'])
                    if njoints_cutoff:
                        k = min(k, njoints_cutoff - 1)

                    rot_mats = None
                    rot_mats_g = None

                    poses = bdata['poses'].reshape((n_frms, -1, 3))[:, :k + 1]
                    translations = bdata['trans']
                    if do_correct_amass_rotation:
                        correct_amass_rotation(poses)
                        correct_amass_rotation(translations, 'translations')
                    if do_rotate_y:
                        rotate_y_smpl(poses, rotate_y_deg)
                        rotate_y_smpl(translations, rotate_y_deg, 'translations')
                    beta = bdata['betas'][:BETA_SZ]

                    if normalise_framerates and framerate > target_framerate:
                        poses, _ = adjust_framerate(framerate, target_framerate, poses)
                        translations, framerate = adjust_framerate(framerate, target_framerate, translations)
                        framerate = int(round(framerate))
                        n_frms = poses.shape[0]
                        lg.info("%d (-> %d) frames downsampled; fps: %d -> %d",
                                n_frms_og, n_frms, framerate_og, framerate)

                    # Compute joint positions
                    if batch_processing:
                        if normalise_shape:
                            betas = np.zeros((1, BETA_SZ), dtype=poses.dtype)
                        else:
                            betas = beta[np.newaxis]

                        cur_bdatas_batch.append({
                            'betas': betas,
                            'poses': poses,
                            'n_frms': n_frms,
                            'n_frms_og': n_frms_og,
                            'translations': translations,
                            'path': bdata_info['path'],
                            'gender': gender,
                            'framerate': framerate,
                            'framerate_og': framerate_og
                        })

                        if len(cur_bdatas_batch) == n_body_datas_per_batch or ibdata == nbdatas_total - 1:
                            # aggregate
                            poses = np.concatenate(
                                    [e['poses'] for e in cur_bdatas_batch], axis=0)
                            betas = []
                            for e in cur_bdatas_batch:
                                nf = e['poses'].shape[0]
                                bb = e['betas']
                                assert len(bb.shape) == 2 and bb.shape[0] == 1
                                for _ in range(nf):
                                    betas.append(bb)
                            betas = np.concatenate(betas, axis=0)

                            if joints_rest is not None:
                                joints_rest = joints_rest[0].repeat(poses.shape[0], 1, 1)

                            try:
                                joints_d = compute_joints(config, poses, beta=betas,
                                    model=models_cached.get(model_path, None), model_path=model_path,
                                    njoints_cutoff=njoints_cutoff, joints_rest=joints_rest,
                                    return_dict=True)
                            except:
                                traceback.print_exc()
                                breakpoint()
                                # This error is too easy to miss...
                                raise RuntimeError("Failed to compute joints ({})".format(save_dir))

                            if normalise_shape:  # Reuse resting joints location
                                joints_rest = joints_d['joints_rest']
                            if model_path not in models_cached:
                                models_cached[model_path] = joints_d['model']
                            joints = joints_d['joints']
                            rot_mats = joints_d['rot_mats']
                            rot_mats_g = joints_d['rot_mats_g']
                            cur_bdatas_batch_processed = True
                        else:
                            joints = None
                        beta = None
                    else:  # not batch_processing
                        if normalise_shape:
                            beta = np.zeros_like(beta)

                        bdata_joints = []
                        for frm in tqdm(range(n_frms)):
                            pose = bdata['poses'][frm]

                            joints_d = compute_joints(config, pose, transl=(0, 0, 0),
                                    model=models_cached.get(model_path, None), model_path=model_path,
                                    return_dict=True)

                            joints = joints_d['joints']
                            if model_path not in models_cached:
                                models_cached[model_path] = joints_d['model']

                            bdata_joints.append(joints)
                        joints = np.array(bdata_joints)
                        betas = beta

                    data_d = None
                    # Save
                    if not save_in_batch:
                        data_d = {
                            'type': bdata_info['dataset_type'],
                            'dataset_name': bdata_info['dataset_shortname'],
                            'path': bdata_rel_path,
                            'subject': bdata_info['subject'],
                            'framerate': framerate,
                            'nframes': n_frms,
                            'nframes_og': n_frms_og,
                            'njoints': k + 1,
                            'joints': joints,
                            'poses': poses,
                            'translations': translations,
                            'shape': beta,
                            'gender': gender
                        }

                    elif cur_bdatas_batch_processed:  # save_in_batch == True
                        nfrms = [e['n_frms'] for e in cur_bdatas_batch]
                        max_frms = max(nfrms)
                        lg.info("#frames min: {}, #frames max: {}".format(min(nfrms), max_frms))
                        nf_cum = 0
                        #joints_b = []
                        poses_b = []
                        joints3_b = []
                        joints22_b = []
                        joints22_g_b = []
                        rot_mats_b = []
                        rot_mats_g_b = []
                        joints22_g_grounded_b = []
                        dtransls_b = []
                        assert model_config['do_correct_amass_rotation']
                        # Iterate through animations
                        # Pad data with zeros so they all span the same number of frames
                        for bd, nf in zip(cur_bdatas_batch, nfrms):
                            pad_amt = max_frms - nf

                            transls_cur = bd['translations']
                            joints22_cur = joints[nf_cum:nf_cum+nf, :22]

                            joints22_global_cur = \
                                np.repeat(transls_cur[..., np.newaxis, :], 22, axis=-2) \
                                + joints22_cur

                            foot_min_y = np.min(joints22_global_cur[:, FEET_JOINTS, 1])
                            joints22_global_grounded_cur = joints22_global_cur.copy()
                            joints22_global_grounded_cur[:, :, 1] -= foot_min_y
                            assert math.isclose(0.0, joints22_global_grounded_cur[:, FEET_JOINTS, 1].min())

                            poses_b.append(
                                    np.pad(bd['poses'],
                                        ((0, pad_amt), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            joints22_b.append(np.pad(joints22_cur,
                                        ((0, pad_amt), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            joints22_g_b.append(np.pad(joints22_global_cur,
                                        ((0, pad_amt), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            joints22_g_grounded_b.append(np.pad(joints22_global_grounded_cur,
                                        ((0, pad_amt), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            rot_mats_b.append(np.pad(rot_mats[nf_cum:nf_cum+nf],
                                        ((0, pad_amt), (0, 0), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            rot_mats_g_b.append(np.pad(rot_mats_g[nf_cum:nf_cum+nf],
                                        ((0, pad_amt), (0, 0), (0, 0), (0, 0)),
                                        'constant', constant_values=0.0))
                            dtransls_b.append(np.pad(
                                          np.diff(transls_cur, axis=0, prepend=0),
                                        ((0, pad_amt), (0, 0)),
                                        'constant', constant_values=0.0))
                            nf_cum += nf
                        poses_b = np.array(poses_b)
                        rot_mats_b = np.array(rot_mats_b)
                        rot_mats_g_b = np.array(rot_mats_g_b)
                        dtransls_b = np.array(dtransls_b)
                        joints22_b = np.array(joints22_b)
                        joints22_g_b = np.array(joints22_g_b)
                        joints22_g_grounded_b = np.array(joints22_g_grounded_b)

                        data_d = {
                            'paths': [e['path'] for e in cur_bdatas_batch],
                            'nframes': nfrms,
                            'genders': [e['gender'] for e in cur_bdatas_batch],
                            'poses': poses_b,       # NB x NmaxFrm x J x 3
                            'rot_mats': rot_mats_b, # NB x NmaxFrm x J x 3 x 3
                            'rot_mats_g': rot_mats_g_b, # NB x NmaxFrm x J x 3 x 3
                            'dtransls': dtransls_b, # NB x NmaxFrm x 3
                            'joints22': joints22_b,   # NB x NmaxFrm x 22 x 3
                            'joints22_g': joints22_g_b,   # NB x NmaxFrm x 22 x 3
                            'joints22_g_grounded':  joints22_g_grounded_b, # NB x NmaxFrm x 22 x 3
                            'betas': np.array([e['betas'][0] for e in cur_bdatas_batch]),
                            'framerate': framerate
                        }

                        cur_bdatas_batch.clear()
                        gc.collect()
                        cur_bdatas_batch_processed = False
                        batch_idx += 1

                    if data_d:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        np.savez(save_path, **data_d)
                        lg.info("Saved preprocessed data to: {}".format(save_path))
            except (KeyboardInterrupt, NotImplementedError, BdbQuit) as e:
                traceback.print_exc()
                if os.path.isfile(save_path):
                    os.remove(save_path)
                exit(1)
            except:
                traceback.print_exc()
                breakpoint()
                if debug:
                    breakpoint()
                nerrs += 1
                if os.path.isfile(save_path):
                    os.remove(save_path)
                lg.exception("Failed to parse body data")
    lg.info("Done")


def parse_gender(bdata_gender):
    s = str(bdata_gender)
    if 'male' in s:
        return 'male'
    elif 'female' in s:
        return 'female'
    elif 'neutral' in s:
        return 'neutral'
    else:
        raise ValueError("Unknown gender: {}".format(bdata_gender))


def preprocess_babel(batches_base_dir, output_base_dir, babel_data_dir):
    os.makedirs(output_base_dir, exist_ok=True)

    babeld = {}
    all_act_cats = []
    all_act_cats_w = defaultdict(lambda: 0.0)

    for ibabelfile, babeld_fn in enumerate(os.listdir(babel_data_dir)):
        babeld_fp = osp.join(babel_data_dir, babeld_fn)
        print("[{}/{}] BABEL file: '{}'".format(
                    ibabelfile+1, len(os.listdir(babel_data_dir)), babeld_fp))
        with open(babeld_fp) as f:
            babeld_js = json.load(f)
            for d in babeld_js.values():
                pth = d['feat_p']
                if d.get('frame_ann', None):
                    assert pth not in babeld
                    babeld[pth] = d
                    for ac_d in (d['frame_ann'].get('labels') or []):
                        ac_dur = ac_d['end_t'] - ac_d['start_t']
                        for ac in ac_d.get('act_cat') or []:
                            all_act_cats.append(ac)
                            all_act_cats_w[ac] += ac_dur
    print("BABEL motions count: {}".format(len(babeld)))
    all_act_cats = Counter(all_act_cats)
    nacts = len(list(all_act_cats))
    print("BABEL unique action categories ({}):\n{}".format(
                len(all_act_cats), all_act_cats.most_common()))
    print("BABEL unique action categories (weighted) ({}):\n{}".format(
                len(all_act_cats),
                sorted([(k, v) for k, v in all_act_cats_w.items()],
                    key=itemgetter(1), reverse=True)))
    act2idx = {}
    idx2act = []
    for i, (act, _) in enumerate(all_act_cats.most_common()):
        act2idx[act] = i
        idx2act.append(act)

    C0 = {
        'DFaust': 'DFause67',
        'SSM': 'SSMsynced',
        'Transitions': 'Transitionsmocap',
        'HDM05': 'MPIHDM05',
        'TCDHands': 'TCDhandMocap',
        'MoSh': 'MPImosh',
        'PosePrior': 'MPILimits',
    }
    print("Start preprocessing BABEL for: '{}'".format(batches_base_dir))
    print("Output dir: '{}'".format(output_base_dir))
    for split in ['train', 'validation', 'test']:
        batches_dir = osp.join(batches_base_dir, split)
        output_dir = osp.join(output_base_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        for ibatch, batch_fn in enumerate(tqdm(os.listdir(batches_dir), desc="Dataset: " + split)):
            batch_fp = osp.join(batches_dir, batch_fn)
            output_batch_fp = osp.join(output_dir, 'babel_' + batch_fn)
            with np.load(batch_fp) as b:
                batchsz = b['poses'].shape[0]
                maxnfrms = b['poses'].shape[1]
                paths_b = b['paths']
                nframes_b = b['nframes']
                assert len(paths_b) == batchsz and len(nframes_b) == batchsz
                if len(b['framerate'].shape) == 0:
                    fps_b = batchsz * [b['framerate'].item()]
                elif len(b['framerate'].shape) == 1 and len(b['framerate']) == batchsz:
                    fps_b = b['framerate']
                else:
                    raise AssertionError("Invalid framerate '{}' in batch '{}'".format(
                                b['framerate'], batch_fp))

                act_cat_b = np.zeros((batchsz, maxnfrms, nacts), dtype=int)
                for ib in range(batchsz):
                    path = paths_b[ib]
                    fps_p = fps_b[ib]
                    nfrms_p = nframes_b[ib]
                    cpath = path.replace('\\', '/').split('/')[-5:]
                    del cpath[1]
                    if cpath[0] in C0:
                        cpath[0] = C0[cpath[0]]
                    assert len(cpath) == 4
                    cpath = '/'.join(cpath)

                    act_cats_curmotion = []
                    st_frms_curmotion = []
                    ed_frms_curmotion = []
                    if cpath in babeld:
                        babel_data = babeld[cpath]
                        dur = babel_data['dur']
                        for seg in babel_data['frame_ann']['labels']:
                            st = min(nfrms_p - 1, round(nfrms_p * (seg['start_t'] / dur)))
                            ed = min(nfrms_p - 1, round(nfrms_p * (seg['end_t'] / dur)))
                            act_cat = seg['act_cat'] or []
                            st_frms_curmotion.append(st)
                            ed_frms_curmotion.append(ed)
                            act_cats_curmotion.append(act_cat)

                    for ac_list, st, ed in \
                        zip(act_cats_curmotion, st_frms_curmotion, ed_frms_curmotion):
                        for ac in ac_list:
                            act_cat_b[ib, st:ed+1, act2idx[ac]] = 1

                # Write output batch
                with open(output_batch_fp, 'wb') as f:
                    d = {
                        'act_labels': np.array(idx2act, dtype=str),
                        'act_counts': np.array([all_act_cats[a] for a in idx2act]),
                        'act_counts_w': np.array([all_act_cats_w[a] for a in idx2act]),
                        'act_cat': act_cat_b,
                    }
                    np.savez(f, **d)


def preprocessed_to_model_inout(preprocessed_data, model_config, train_or_test,
        return_intermediate_values=False, batchify=True,
        test_run=False, test_run_params=None, **kw):
    if test_run:
        glg().info("preprocessed_to_model_inout - test run")
        assert not preprocessed_data
        params = test_run_params or {}
        nb = params.get('batch_size', 1)
        nf = params.get('nframes', 3)
        in_j = model_config['input_njoints']
        out_j = model_config['output_njoints']

        preprocessed_data = {
            'rot_mats': np.zeros((nb, nf, out_j, 3, 3), dtype=dtype_np),
            'rot_mats_g': np.zeros((nb, nf, out_j, 3, 3), dtype=dtype_np),
            'joints22': np.zeros((nb, nf, 22, 3), dtype=dtype_np),
            'joints22_g': np.zeros((nb, nf, 22, 3), dtype=dtype_np),
            'joints22_g_grounded': np.zeros((nb, nf, 22, 3), dtype=dtype_np),
            'dtransls': np.zeros((nb, nf, 3)),
        }

    # (NB x) NmaxFrms x J x (3 x 3)
    rmats = preprocessed_data['rot_mats'].astype(dtype_np) 
    # (NB x) NmaxFrms x #(inpJ) x (3 x 3)
    r3 = preprocessed_data['rot_mats_g'].astype(dtype_np)[..., model_config.input_joints, :, :]

    # (NB x) NmaxFrms x #(inpJ) x 3
    if model_config['input_global_joint_positions']:
        if model_config['global_joints_grounded']:
            j3 = preprocessed_data['joints22_g_grounded'][..., model_config.input_joints, :].astype(dtype_np)
        else:
            j3 = preprocessed_data['joints22_g'][..., model_config.input_joints, :].astype(dtype_np)
    else:
        j3 = preprocessed_data['joints22'][..., model_config.input_joints, :].astype(dtype_np)
    
    if True:#kw.get('viz_global_pose', False):
        if model_config['global_joints_grounded']:
            j22 = preprocessed_data['joints22_g_grounded'].astype(dtype_np)
        else:
            j22 = preprocessed_data['joints22_g'].astype(dtype_np)
    else:
        j22 = None

    # Use Transl
    use_transl = model_config['input_delta_translation']
    if use_transl:
        # (NB x) NmaxFrms x 3
        dtransls = preprocessed_data['dtransls'].astype(dtype_np)
    else:
        dtransls = None

    intm_vals = {}

    # Batchify
    remove_batch_axis = False
    if len(j3.shape) == 3:
        remove_batch_axis = not batchify
        rmats = rmats[np.newaxis]
        #rmats_g = rmats_g[np.newaxis]
        j3 = j3[np.newaxis]
        if j22 is not None:
            j22 = j22[np.newaxis]
        r3 = r3[np.newaxis]
        if use_transl:
            dtransls = dtransls[np.newaxis]

    if len(j3.shape) != 4:
        raise ValueError("Unknown preprocessed data format (joints3.shape: {})".format(j3.shape))

    # Shape: NB x NmaxFrms x ...

    batch_sz = rmats.shape[0]
    nfrms_max = rmats.shape[1]

    inp_types = []
    intm_vals = {
        'batch_size': batch_sz,
        'max_frames': nfrms_max,
        'batch': {}
    }

    # Randomise root rotation
    if train_or_test == 'train' and model_config['random_root_rotation']:
        randangles = (2 * math.pi) * np.random.random((batch_sz, 1))
        rand_rmat = Rotation.from_euler('y', randangles, degrees=False).as_matrix().astype(dtype_np)
        # nb x nf x (3 x 3)
        rand_rmat = np.repeat(rand_rmat[:, np.newaxis], nfrms_max, axis=1)

        rmats[..., 0, :, :] = rand_rmat @ rmats[..., 0, :, :]

        # nb x nf x 3 x (3 x 3)
        rand_rmat_j3 = np.repeat(rand_rmat[..., np.newaxis, :, :], model_config.input_njoints, axis=-3)
        r3 = rand_rmat_j3 @ r3

        j3 = (rand_rmat_j3 @ j3[..., np.newaxis])[..., 0]
        if j22 is not None:
            rand_rmat_j22 = np.repeat(rand_rmat[..., np.newaxis, :, :], 22, axis=-3)
            j22 = (rand_rmat_j22 @ j22[..., np.newaxis])[..., 0]
        if use_transl:
            # RHS: nb x nf x 3
            # LHS: ((nb x nf x (3 x 3)) @ (nb x nf x 3 x 1))[..., 0]
            dtransls = (rand_rmat @ transls[..., np.newaxis])[..., 0]

    rot_6d = rmats[..., :2]
    r3_6d = r3[..., :2].reshape(*r3.shape[:-2], 6)

    if return_intermediate_values:
        j3c = j3.copy()
        r3c = r3.copy()
        intm_vals['batch']['j3'] = j3c if not remove_batch_axis else j3c[0]
        intm_vals['batch']['r3'] = r3c if not remove_batch_axis else r3c[0]
        if j22 is not None:
            j22c = j22.copy()
            intm_vals['batch']['j22'] = j22c if not remove_batch_axis else j22c[0]

    j3_me = j3.copy()  # for ME do not apply any normalisation
    # Global pos normalisation
    if model_config['input_global_joint_positions'] and \
        model_config['normalise_global_joint_positions']:
        if model_config['normalise_global_joint_positions_y']:
            norm_axes = [0, 1, 2]
        else:
            norm_axes = [0, 2]  # normalise x, z
        j3_mean = np.mean(j3[..., norm_axes], axis=-2, keepdims=True)
        j3[..., norm_axes] -= j3_mean
        if model_config['normalise_global_joint_positions_divbystd']:
            j3_std = np.std(j3[..., norm_axes], axis=-2, keepdims=True)
            j3[..., norm_axes] /= np.maximum(0.00000001, j3_std)

    # Input base: NB x NmaxFrms x 3#(inpJ)
    if model_config['input_positions']:
        inp_base = j3.reshape((batch_sz, nfrms_max, -1))
        inp = inp_base
        inp_types.append(((0, inp.shape[-1]), 'j3'))
    else:
        inp = np.zeros((batch_sz, nfrms_max, 0), dtype=j3.dtype)

    me_inp_base = j3_me
    me_inp = me_inp_base

    # TODO if using convolutional layer organise them more sensibly
    def concat2inp(a_, name_):
        a_flat_ = a_.reshape((batch_sz, nfrms_max, -1))
        inp_types.append(((inp.shape[-1], inp.shape[-1] + a_flat_.shape[-1]), name_))
        return np.concatenate([inp, a_flat_], axis=-1)

    def concat2meinp(a_):
        assert a_.shape[-2] == model_config.input_njoints and len(a_.shape) == 4 # nb x nf x 3 x nfeats
        return np.concatenate([me_inp, a_], axis=-1)  # concat along features dim

    if model_config['input_rotations']:
        inp = concat2inp(r3_6d, 'rotations')
        me_inp = concat2meinp(r3_6d)

    if model_config['input_velocities']:
        v3 = joints_to_vel(j3, initial='rest')
        inp = concat2inp(v3, 'velocities')
        me_inp = concat2meinp(v3)

    if model_config['input_angular_velocities']:
        av3 = angular_velocity_from_joints_rot_mats(r3, initial='rest')[..., :2]
        inp = concat2inp(av3, 'angular_velocities')
        me_inp = concat2meinp(av3.reshape(*av3.shape[:-2], 6))

    if model_config['input_prev_frame_pose']:
        #if model_config['input_prev_frame_pose__use_gt']:
        #    prevfrmo = np.pad(rot_6d[..., :model_config['output_njoints'], :, :].reshape((batch_sz, nfrms_max, -1)),
        #             ((0, 0), (1, 0), (0, 0)), 'edge')[:, :-1]
        #else:
        #    prevfrmo = np.zeros((batch_sz, nfrms_max, model_config.output_dim),
        #            dtype=inp.dtype)

        #intm_vals['prev_frame_pose_idx_begin'] = inp.shape[-1]
        #inp = np.concatenate([inp, prevfrmo], axis=-1)
        #intm_vals['prev_frame_pose_idx_end'] = inp.shape[-1] - 1
        raise NotImplementedError
    if use_transl:
        inp = concat2inp(dtransls, 'dtransls')

    outp = rot_6d.reshape((batch_sz, nfrms_max, -1)) # NB x NmaxFrms x 6J

    if remove_batch_axis:
        inp = inp[0]
        outp = outp[0]

    intm_vals['input_types'] = inp_types
    intm_vals['me_inp'] = me_inp

    if not return_intermediate_values:
        return (inp, outp)
    else:
        return (inp, outp, intm_vals)
