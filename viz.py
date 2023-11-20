from util import get_model_path, get_device, get_root_dir, glg, load_smpl_model
from data_wrangling import rot_6d_to_mat, rot_mat_to_vec, correct_amass_rotation, totorch, compute_joints
from data_loading import load_torch_model
from model import get_nn, model_output_to_smpl_poses, model_output_to_rot_mats
from smpl_np import joint_labels

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image

import os
import os.path as osp
import subprocess
import shutil
import datetime
import time


def init_viz(config):
    if config['use_tkagg']:
        matplotlib.use('TkAgg')


def smpl_to_mpl_axes(xyz):
    x, y, z = xyz
    return x, -np.array(z), y


def viz_model(model, ax=None, posed=True, joints=None, display_labels=False,
        plt_title='', plt_show=True, caption='', caption_pos=None, **kw):
    clear = kw.get('clear', False)

    if ax is None:
        ax = plt.axes()
    if joints is None:
        joints = []
        for j in range(model.k + 1):
            if posed:
                coords = model.G[j, :3, 3]
            else:
                coords = model.J[j, :]
            joints.append(coords)

    if clear:
        ax.set_axis_off()

    # Draw labels
    if display_labels:
        # TODO wrong labels for SMPL-X
        for j in range(kw.get('k', kw.get('njoints', model.k + 1) - 1) + 1):
            ax.text(*smpl_to_mpl_axes(joints[j]),
                    "{}: {}".format(j, joint_labels[j]) if j in joint_labels else "")

    # Draw caption
    if caption:
        if not caption_pos:
            caption_pos = (0, 0)
        ax.text2D(caption_pos[0], caption_pos[1], caption)

    # Draw joints
    if kw.get('draw_joints', False):
        joint_coords = []
        for i in range(3):
            joint_coords.append([joints[j][i] for j in range(len(joints))])
        ax.scatter(*smpl_to_mpl_axes(joint_coords))

    # Draw bones
    bdepths = {}
    for bone_id in model.kintree_table[1, :]:
        if bone_id == 0:
            continue
        l = 1
        b = bone_id
        while True:
            p = model.parent[b]
            if p == 0:
                break
            b = p
            l += 1
        bdepths[bone_id] = l

    maxbdepth = max(bdepths.values())

    r = 0.45
    nj = model.kintree_table[1, :].max()
    for bone_id in model.kintree_table[1, :]:
        if bone_id != 0:
            par_bone_id = model.parent[bone_id]
            ax.plot(*smpl_to_mpl_axes(
                [[joints[par_bone_id][i], joints[bone_id][i]] for i in range(3)]),
                    linewidth=6)
                #color=3*[(max(0, 1 - r - (1 - r) * bdepths[bone_id]/maxbdepth))])

    # Try to make axes equally spaced
    #if not posed:
    #    ax.set_ylim([-0.8, 0.8])

    if 'xlim' in kw:
        ax.set_xlim(kw['xlim'])
    if 'ylim' in kw:
        ax.set_ylim(kw['ylim'])
    if 'zlim' in kw:
        ax.set_zlim(kw['zlim'])

    if not clear:
        ax.set_title(plt_title)
    if plt_show:
        plt.show()


def play_animation(config, data_path, fig, ax, data=None, do_play=True, save_as_images=False,
        preprocessed=True, max_frames=None,
        correct_root_rotation=True, zero_root_orientation=False,
        fix_graph_boundaries=True, **kw):
    from smpl_np import SMPLModel

    data_root_dir = get_root_dir(config, 'preprocessing' if preprocessed else 'body_data')
    clear = kw.get('clear', False)

    data_fn_noext = os.path.splitext(os.path.basename(data_path))[0]
    anim_save_root_dir = get_root_dir(config, 'animation')
    anim_save_dir = os.path.join(anim_save_root_dir,
            os.path.relpath(os.path.dirname(data_path), data_root_dir),
            data_fn_noext)
    print("Animation frames will be saved to: {}".format(anim_save_dir))
    if os.path.isdir(anim_save_dir):
        shutil.rmtree(anim_save_dir)
    os.makedirs(anim_save_dir)

    if data is None:
        data = np.load(data_path, encoding='latin1')

    model_path = get_model_path(config, str(data['type']), str(data.get('gender', 'male')))
    model = load_smpl_model(model_path=model_path, as_class=True, njoints=data['njoints'])

    n_frms = min(data.get('poses', data.get('rot_mats', data.get('joints'))).shape[0],
            max_frames or data['nframes'], data['nframes'])

    captions = kw.get('captions', [])

    joints_all = []
    if 'joints' in data:
        joints_all = np.array(data['joints'])
    elif 'rot_mats' in data:
        joints_all = compute_joints(config, rot_mats=data['rot_mats'], model_path=model_path)
    else:
        poses = data['poses'].copy()
        for frm in tqdm(range(n_frms), desc="Computing joints"):
            pose = poses[frm]

            pose[0] = kw.get('root_rotation', pose[0])
            if correct_root_rotation:
                correct_amass_rotation(pose)

            if zero_root_orientation:
                pose[0] = [0, 0, 0]

            model.set_params(pose=pose, beta=data['shape'],
                    trans=data['translations'][frm])
            joints = model.pose_joints()
            joints_all.append(joints)
        joints_all = np.array(joints_all)

    xlim, ylim, zlim = None, None, None
    if fix_graph_boundaries:
        min_x, min_y, min_z = smpl_to_mpl_axes(np.min(joints_all.reshape((-1, 3)), axis=0))
        max_x, max_y, max_z = smpl_to_mpl_axes(np.max(joints_all.reshape((-1, 3)), axis=0))
        min_x, max_x = min(min_x, max_x), max(min_x, max_x)
        min_y, max_y = min(min_y, max_y), max(min_y, max_y)
        min_z, max_z = min(min_z, max_z), max(min_z, max_z)
        padding_pct = 0.05
        xlim = (min_x - padding_pct * (max_x - min_x),
                max_x + padding_pct * (max_x - min_x))
        ylim = (min_y - padding_pct * (max_y - min_y),
                max_y + padding_pct * (max_y - min_y))
        zlim = (min_z - padding_pct * (max_z - min_z),
                max_z + padding_pct * (max_z - min_z))

        #max_rng = max(ll[1] - ll[0] for ll in [xlim, ylim, zlim])
        avg_rng = np.mean([ll[1] - ll[0] for ll in [xlim, ylim, zlim]])
        rng = avg_rng
        xlim, ylim, zlim = [((ll[1]+ll[0])/2 - rng/2, (ll[1]+ll[0])/2 + rng/2)
                for ll in [xlim, ylim, zlim]]

    if clear:
        xlim = (-0.5, 0.5)
        ylim = (-0.5, 0.5)
        zlim = (-1.0, 0.2)

    def update(index):
        ax.clear()
        viz_model(model, ax=ax, posed=ylim[0] > 0.1, joints=joints_all[index],
                display_labels=kw.get('display_labels', True),
                plt_title=kw.get('plt_title', data_fn_noext), plt_show=False,
                xlim=xlim, ylim=ylim, zlim=zlim,
                caption=captions[index] if captions and index <= len(captions) - 1 else '',
                clear=clear)

    if save_as_images:
        for frm in tqdm(range(n_frms), desc="Processing frames"):
            update(frm)
            fig.savefig(os.path.join(anim_save_dir, "{}_{:04}.jpg".format(data_fn_noext, frm)))
        if do_play:
            subprocess.run(['sxiv', anim_save_dir, '-S', '0.1'])
        return anim_save_dir
    else:
        fps = config['model_config']['target_framerate']
        anim_writer = 'ffmpeg' # 'pillow'
        t0 = time.time()
        anim = FuncAnimation(fig, update, frames=n_frms, interval=1000/fps,
                repeat=False, init_func=lambda: None)
        glg().info("FuncAnimation took %.2fs", time.time() - t0)
        t0 = time.time()
        #plt.tight_layout()  # causes plot title to be cropped out
        anim_path = osp.join(anim_save_dir, "{}.gif".format(data_fn_noext))
        anim.save(anim_path, writer=anim_writer, fps=fps)
        glg().info("Saving anim took %.2fs using writer backend '%s'",
                time.time() - t0, anim_writer)
        plt.close()
        return anim_path


def test_one(config, checkpoint_path=None, test_data=None,
        pred_n_outp=None, pred_n_outp_pose=None, pred_n_outp_joints=None,
        nframes=None, **kw):
    """
    One of (checkpoint_path, test_data), pred_n_outp = (pred, outp), pred_n_outp_poses = (pred_pose, outp_pose), pred_n_outp_joints = (pred_joints, outp_joints) must be provided.
    If test_data provided, prediction(s) will be run through the model loaded with checkpoint.
    """
    model_config = config['model_config']
    use_rot_6d = kw.get('use_rot_6d', False)

    batch_idx = kw.get('batch_idx', 0)
    if test_data is not None:
        glg().info("Running prediction from test data...")
        # Init device & load model
        device = get_device()
        model = get_nn(config=config).to(device)
        load_torch_model(model, checkpoint_path)

        if len(test_data['input'].shape) == 2:
            inp = totorch(test_data['input'], device=device)
            outp = totorch(test_data['output'], device=device)
        elif len(test_data['input'].shape) == 3:
            inp = totorch(test_data['input'][batch_idx], device=device)
            outp = totorch(test_data['output'][batch_idx], device=device)
        else:
            raise ValueError

        if model_config['input_prev_frame_pose'] and \
            not model_config['input_prev_frame_pose__use_gt']:
            raise NotImplementedError
            #i1 = test_data['intm_vals']['prev_frame_pose_idx_begin']
            #i2 = test_data['intm_vals']['prev_frame_pose_idx_end']
            #hs = None
            #pred = []
            #for ifrm in tqdm(range(inp.shape[0]), desc="Autorgression"):
            #    if ifrm > 0:
            #        inp[ifrm, i1:i2+1] = pred[ifrm - 1] # outp[ifrm - 1]  # sanity check
            #    pred_frm, hs = model(inp[[ifrm]], hs)
            #    pred.append(pred_frm.detach())
            #pred = torch.concat(pred, dim=0)
        else:
            pred, _ = model(inp, None)

        pred = pred.detach().cpu().numpy()
        outp = outp.cpu().numpy()
    elif pred_n_outp is not None:
        pred, outp = pred_n_outp
        if len(outp.shape) == 3:
            outp = outp[batch_idx]
            pred = pred[batch_idx]
    elif pred_n_outp_pose is not None:
        if len(pred_n_outp_pose[0].shape) == 4:
            raise NotImplementedError
        pred = None
        outp = None
    elif pred_n_outp_joints is not None:
        if len(pred_n_outp_joints[0].shape) == 4:
            raise NotImplementedError
        pred = None
        outp = None
    else:
        raise ValueError(
                "One of test_data+checkpoint_path, pred_n_outp, pred_n_outp_pose, or pred_n_outp_joints must be provided.\n"
                "See function documentation.")

    # If param nframes given, only extract that amount (from the back) for evaluation
    if pred is not None:
        n_allframes = pred.shape[0]
    elif pred_n_outp_pose is not None:
        n_allframes = pred_n_outp_pose[0].shape[0]
    else:
        n_allframes = pred_n_outp_joints[0].shape[0]

    if nframes is not None and nframes > 0:
        nframes = min(nframes, n_allframes)
        if pred is not None:
            outp = outp[n_allframes - nframes:]
            pred = pred[n_allframes - nframes:]
    else:
        nframes = n_allframes

    glg().info("Using %d out of %d frames", nframes, n_allframes)

    zero_shape = np.zeros(10)
    transl = np.zeros((nframes, 3))

    pred_pose = outp_pose = None
    pred_rotmats = outp_rotmats = None
    pred_joints = outp_joints = None
    if pred_n_outp_pose is not None:
        pred_pose, outp_pose = pred_n_outp_pose
        pred_pose = pred_pose[n_allframes - nframes:]
        outp_pose = outp_pose[n_allframes - nframes:]
    elif pred_n_outp_joints is not None:
        pred_joints, outp_joints = pred_n_outp_joints
        pred_joints = pred_joints[n_allframes - nframes:]
        outp_joints = outp_joints[n_allframes - nframes:]
    else:
        rf2c = kw.get('recompute_first_two_cols_of_6d_rot', True)
        # Compute pose OR rot_mats from (pred, outp)
        if use_rot_6d:
            glg().info("Using 6D rotation (rf2c=%s)", rf2c)
            pred_rotmats = model_output_to_rot_mats(pred, recompute_first_two_cols_of_6d_rot=rf2c)
            outp_rotmats = model_output_to_rot_mats(outp, recompute_first_two_cols_of_6d_rot=rf2c)
        else:
            glg().info("Using SMPL poses")
            pred_pose = model_output_to_smpl_poses(pred, recompute_first_two_cols_of_6d_rot=rf2c)
            outp_pose = model_output_to_smpl_poses(outp, recompute_first_two_cols_of_6d_rot=rf2c)

    gt_smpl = {
        'type': 'smpl',
        'shape': zero_shape,
        'njoints': 24,
        'nframes': nframes,
        'translations': transl,
        'gender': 'male'
    }
    if outp_pose is not None:
        gt_smpl['poses'] = outp_pose
    elif outp_rotmats is not None:
        gt_smpl['rot_mats'] = outp_rotmats
    else:
        gt_smpl['joints'] = outp_joints

    pred_smpl = {
        'type': 'smpl',
        'shape': zero_shape,
        'njoints': 24,
        'nframes': nframes,
        'translations': transl,
        'gender': 'male'
    }
    if pred_pose is not None:
        pred_smpl['poses'] = pred_pose
    elif pred_rotmats is not None:
        pred_smpl['rot_mats'] = pred_rotmats
    else:
        pred_smpl['joints'] = pred_joints

    pred_captions = []
    if kw.get('viz_losses') or []:
        losses_to_viz = kw['viz_losses']
        losses_ds = {}
        if kw.get('perframe_losses') or []:
            losses_d_key = kw.get('perframe_losses_desc') or 'perframe_losses'
            losses_ds[losses_d_key] = kw['perframe_losses']
        if kw.get('ma_losses') or []:
            losses_d_key = kw.get('ma_losses_desc') or 'ma_losses'
            losses_ds[losses_d_key] = kw['ma_losses']

        for fr in range(nframes):
            txt = ""
            for k, l_d in losses_ds.items():
                txt += "== {} ==\n".format(k)
                ll = l_d[fr]
                for lk in losses_to_viz:
                    txt += "{}: ".format(lk)
                    if lk in ll:
                        txt += "{:.4f}".format(ll[lk])
                    else:
                        txt += "-"
                    txt += "\n"
                txt += "\n"
            pred_captions.append(txt)
        assert len(pred_captions) == nframes

    if test_data is not None:
        test_data_path = test_data['path'] if isinstance(test_data['path'], str) else test_data['path'][batch_i]
        data_name = os.path.splitext(os.path.basename(test_data_path))[0]
    else:
        if 'animation_dirname' in kw:
            data_name = kw['animation_dirname']
        else:
            data_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    anim_base_dir = os.path.join(get_root_dir(config, 'animation'), "test_anim", data_name)

    # Render GT Anim
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    gt_anim_path = play_animation(config, os.path.join(anim_base_dir, "gt"),
            save_as_images=False, preprocessed=True,
            do_play=False, data=gt_smpl, ax=ax, fig=fig, display_labels=False,
            correct_root_rotation=False, plt_title="GT", clear=kw.get('clear', False))

    # Render Pred Anim
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    pred_anim_path = play_animation(config, os.path.join(anim_base_dir, "pred"),
            save_as_images=False, preprocessed=True,
            do_play=False, data=pred_smpl, ax=ax, fig=fig, display_labels=False,
            correct_root_rotation=False, plt_title=kw.get('name', "Pred"),
            captions=pred_captions, clear=kw.get('clear', False))

    # Merge videos
    if kw.get('video_path'):
        vid_save_path = kw['video_path']
    else:
        vid_save_path = os.path.join(os.path.dirname(gt_anim_path), "merged_vid.mp4")
    if os.path.isfile(vid_save_path):
        os.remove(vid_save_path)
    glg().info("Video will be saved to: %s", vid_save_path)

    #ffmpeg -i input0 -i input1 -filter_complex vstack=inputs=2 output
    subprocess.run(["ffmpeg", "-y", "-i", gt_anim_path, "-i", pred_anim_path,
            "-filter_complex", "hstack=inputs=2", vid_save_path])

    return {
        'gt_anim_path': gt_anim_path,
        'pred_anim_path': pred_anim_path
    }
