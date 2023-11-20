from util import Config, init_logger, printerr
import data_loading
import preprocessing
import model

import argparse
import sys
import os
import os.path as osp
import json
import functools
import subprocess
import random
import time
import shutil


DEFAULT_CONFIG_PATH = "./config.json"


def load_preprocessed_dataloader(train_or_test, preprocessed_dir, *, model_config,
        motion_embeddings_preprocessed_dir=None, babel_preprocessed_dir=None, **kw):
    if train_or_test == 'train':
        training = True
    elif train_or_test == 'test':
        training = False
    else:
        raise ValueError

    p2m = functools.partial(preprocessing.preprocessed_to_model_inout,
            model_config=model_config, train_or_test=train_or_test)

    d2 = data_loading.Data2(model_config=model_config,
            preprocessed_to_model_inout_func=p2m,
            training_data_dir=preprocessed_dir if training else None,
            test_data_dir=preprocessed_dir if not training else None,
            load_motion_embeddings=motion_embeddings_preprocessed_dir is not None,
            training_motion_embeddings_data_dir=motion_embeddings_preprocessed_dir if training else None,
            test_motion_embeddings_data_dir=motion_embeddings_preprocessed_dir if not training else None,
            load_babel_data=babel_preprocessed_dir is not None,
            training_babel_data_dir=babel_preprocessed_dir if training else None,
            test_babel_data_dir=babel_preprocessed_dir if not training else None
            )

    dlkw = {}
    if kw.get('window_sz', None):
        dlkw['window_sz'] = kw['window_sz']
    if kw.get('batch_sz', None):
        dlkw['batch_sz'] = kw['batch_sz']
    if kw.get('batch_pruning_thresh', None):
        dlkw['batch_pruning_thresh'] = kw['batch_pruning_thresh']
    if kw.get('exhaustive', False):
        dlkw['exhaustive'] = kw['exhaustive']

    return d2.dataloader(train_or_test,
            shuffle_opts=kw.get('shuffle_opts', None), **dlkw)


def load_amass_dataloader(train_or_test, amass_paths, config, **kw):
    from util import glg

    datas = []
    lg = glg()
    lg.info("Preprocessing {} files...".format(len(amass_paths)))
    for i_amass, pth in enumerate(amass_paths):
        lg.info("[{}/{}] {}".format(i_amass+1, len(amass_paths), pth))
        datas.append(preprocessing.prepare(config, pth, **kw))
    lg.info("Preprocessing done")

    p2m = functools.partial(preprocessing.preprocessed_to_model_inout,
            model_config=config['model_config'], train_or_test=train_or_test,
            viz_global_pose=kw.get('viz_global_pose', False))

    d2 = data_loading.Data2(config['model_config'],
            preprocessed_to_model_inout_func=p2m,
            test_datas=datas)

    return d2.dataloader(train_or_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['train', 'test', 'preprocess', 'ops'])

    ap.add_argument('--config', '-C', default=DEFAULT_CONFIG_PATH, dest='config_path')
    ap.add_argument('--model_config', '-M', default=None, dest='model_config_path')

    ap.add_argument('--cuda_index', '-i', default=0)

    ap.add_argument('--log_directory', default=None)
    ap.add_argument('--log_path', default=None)

    ap.add_argument('--random_seed', default=None)

    # Train/Test
    ap.add_argument('--window_size', '-w', type=int, default=-1)
    ap.add_argument('--loading_window_size', '-l', type=int, default=-1)
    ap.add_argument('--batch_size', type=int, default=-1)
    ap.add_argument('--test_batch_size', type=int, default=-1)
    ap.add_argument('--epochs', type=int, default=-1)

    ap.add_argument('--preprocessed', '-p', dest='preprocessed_path',
            help="Path to preprocessed files directory")
    ap.add_argument('--validation', '-v', dest='validation_preprocessed_path',
            help="Path to preprocessed files directory for validation")

    ap.add_argument('--preprocessed_motion_embeddings', '-E', dest='preproc_mes_path')
    ap.add_argument('--validation_preprocessed_motion_embeddings',
            dest='val_preproc_mes_path')
    ap.add_argument('--preprocessed_babel_base', dest='preproc_babel_path_base',
            help="Must contain subdirectories: train/, test/, validation/")
    ap.add_argument('--checkpoints_save_directory', default=None)
    ap.add_argument('--checkpoint', '-c', default=None)

    ap.add_argument('--test_results_directory', '-R', default=None,
            help="Directory in which to save normal test results, unless --compare_test_results is set, "
            "in which case this may either be a directory containing the raw files at depth 1 or "
            "a directory of subdirectories where subdirectories are the normal test results directory which should "
            "contain a raw test results file under normal testing.")
    ap.add_argument('--amass', '-A', dest='amass_paths', nargs='+',
            help="Path(s) to AMASS motion file(s), or directories containing them.")
    ap.add_argument('--test_quantitatively', action='store_true',
            help="Set if wish to run quantitative tests regardless of whether "
            "preprocessed path (via --preprocessed) or "
            "amass data path(s) was/were passed.")
    ap.add_argument('--viz_clear', action='store_true')
    ap.add_argument('--save_as_video', nargs='+', default=None,
            help="Test result will be saved to video. You must provide the path to "
            "the base directory under which the video files will be saved."
            " Optionally you can specify the depth of the directory structure that will transfer"
            " from the input file path to the output file path. Default value is 0."
            " For example, if given values are 'vids/ 2' and given input path is "
            "'d1/d2/d3/amassmotion1.npz', the video will be saved to: 'vids/d2/d3/amassmotion1.mp4'")
    ap.add_argument('--gifs_dir', nargs='+', default=None)
    ap.add_argument('--dump_npz', action='store_true',
            help="If set, arguments to --save_as_video will be treated as arguments for "
            "dumping npz files instead")
    ap.add_argument('--test_result_files', nargs='*', default=[],
            help="Path(s) to test result (raw) files. Used for comparing multiple test results only"
            "(set --compare_test_results). Alternatively pass a directory to --test_results_directory")
    ap.add_argument('--compare_test_results', action='store_true',
            help="Set to compare multiple test results. The raw test result files (.pkl) produced through "
            "testing can be passed via --test_results_directory or --test_result_files."
            " See respective documentations.")
    ap.add_argument('--comparison_test_results_directory', default=None)
    ap.add_argument('--dset_path_index', type=int, nargs='?')
    ap.add_argument('--frames', '-f', nargs='*', type=int,
            help="Number(s) of frames to render.\n"
            "If no input provided, all the frames will be used for all data.\n"
            "If one input provided, its value will be used for all data.\n"
            "Otherwise the number of inputs must be the same as that of data.")
    ap.add_argument('--viz_global_pose', action='store_true')
    ap.add_argument('--qualitative_oui_disable', action='store_true')
    ap.add_argument('--qualitative_oui_ma_window_size', type=int, default=-1)

    ap.add_argument('--export_windows', help="If set, windows will be exported to this path")

    # Motion Embedding
    ap.add_argument('--motion_embedding_test_noise_scale', default=0.0, type=float,
            help="TODO implement for qualitative testing")

    # Data loading
    ap.add_argument('--body_data', default=None, dest='body_data_path')

    ap.add_argument('--prune_batch', dest='batch_pruning_thresh', type=float, default=0.92,
            help="--shuffle is not set to 'all', a batch will be 'pruned', so that "
            "at least 100 * {val} % of motions in the batch are 'alive' for all windows."
            " This is due to the fact that in a batch all motions are zero padded to have "
            "the same number of frames.\nTHIS IS FOR TRAIN/TEST depending on action.")
    ap.add_argument('--prune_batch_validation',
            dest='batch_pruning_thresh_validation', type=float, default=0.86,
            help="Same as --prune_batch but for validation dataset")
    ap.add_argument('--every_possible_window', action='store_true',
            help="Experimental. Currently available for quantitative testing only.")

    ap.add_argument('--shuffle', default='none',
            choices=['none', 'batches', 'windows', 'all'])
    ap.add_argument('--shuffle_n_windows_cutoff', default=-1, type=int)
    ap.add_argument('--shuffle_n_windows_random', action='store_true')
    ap.add_argument('--shuffle_validation', action='store_true')
    ap.add_argument('--shuffle_test', action='store_true')
    ap.add_argument('--shuffle_windows_offsets', default=-1, type=int)
    ap.add_argument('--rotate_y', default=None,
            help="If running qualitative test, the model will be preprocessed to have the root joint rotated about "
            "the y axis by value (degrees) before if given")


    ap.add_argument('--operation', '-O', default=None,
            help="For ops; specify function name in tests.py to execute.")


    # Preprocessing
    ap.add_argument('--babel_data', default=None, help="Path to BABEL data."
            " BABEL (https://babel.is.tue.mpg.de/) -> ./data/babel_v1.0_release")

    ap.add_argument('--dont_sort_by_length', action='store_true',
            help="If NOT set, batches will be stored in ascending order of length during preprocessing.")

    ap.add_argument('--start_tensorboard', '-T', default=None)
    ap.add_argument('--tensorboard_logdir', default=None)
    ap.add_argument('--tensorboard_host', default="0.0.0.0")
    ap.add_argument('--tensorboard_port', default=6006)
    ap.add_argument('--tensorboard_bin', default=".venv/bin/tensorboard")

    ap.add_argument('--debug', '-D', action='store_true')
    loglevels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
    ap.add_argument('--stdout_loglevel', default='INFO', choices=loglevels)
    ap.add_argument('--file_loglevel', default='INFO', choices=loglevels)

    aa = ap.parse_args()

    # Logger
    lg = init_logger(save_to_file=True, save_dir=aa.log_directory, save_path=aa.log_path,
            stream_handler_logging_level=aa.stdout_loglevel,
            file_handler_logging_level=aa.file_loglevel)

    lg.debug("Program arguments: %s", aa)

    # Random Seed
    if aa.random_seed is not None:
        sd = int(aa.random_seed)
    else:
        sd = int(time.time())
    random.seed(sd)
    lg.info("Random seed: %d", sd)

    # Tensorboard
    if aa.start_tensorboard:
        if not aa.tensorboard_logdir:
            tensorboard_logdir = "./allruns/{}".format(
                    datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        lg.info("Starting Tensorboard on {}:{} with logdir: {}".format(
                    aa.tensorboard_host, aa.tensorboard_port, tensorboard_logdir))
        subprocess.run([aa.tensorboard_bin, "--logdir", tensorboard_logdir,
            "--host", aa.tensorboard_host, "--port", tensorboard_port])
    else:
        if not aa.tensorboard_logdir:
            tensorboard_logdir = "./runs"

    # Configs
    config = Config(aa.config_path)

    data_loading.init(cuda_index=aa.cuda_index)
    model.init_model(config, model_config_path=aa.model_config_path, append=True,
            cuda_index=aa.cuda_index, tensorboard_logdir=tensorboard_logdir)
    model_config = config['model_config']
    name = "" if not aa.model_config_path else os.path.basename(aa.model_config_path)

    if aa.export_windows:
        ew_path = aa.export_windows
        ew_path = osp.splitext(ew_path)[0] + ".pkl"
        lg.info("Windows will be exported to: '%s'", ew_path)
        config['export_windows'] = True
        config['exported_windows_path'] = ew_path

    lg.debug("Config:\n%s\n", json.dumps(config, indent=4))
    lg.info("Model Config:\n%s\n\n", model_config)

    # Model Args
    checkpoint = aa.checkpoint or model_config['checkpoint_path']

    nepochs = aa.epochs if aa.epochs > 0 else None
    batch_size = aa.batch_size if aa.batch_size > 0 else model_config['batch_size']
    if aa.action == 'test':
        if aa.test_batch_size > 0:
            batch_size = aa.test_batch_size
        elif model_config['test_batch_size'] > 0:
            batch_size = model_config['test_batch_size']
    window_sz = aa.window_size if aa.window_size > 0 else model_config['window_size']
    loading_window_sz = aa.loading_window_size if aa.loading_window_size > 0 else model_config['loading_window_size']
    if aa.action == 'test' and model_config['test_loading_window_size'] > 0:
        lg.info("test_loading_window_size: {}".format(model_config['test_loading_window_size']))
        loading_window_sz = model_config['test_loading_window_size']
    inp_me = model_config['input_motion_embeddings']
    use_me = bool(inp_me or
            model_config['motion_embedding_guidance'] or
            model_config['motion_embedding_evaluation'])
    preproc_mes_path = aa.preproc_mes_path
    val_preproc_mes_path = aa.val_preproc_mes_path
    lg.info("Batch size: {}".format(batch_size))
    lg.info("Window size: {}".format(window_sz))
    lg.info("Loading window size: {}".format(loading_window_sz))
    if aa.preprocessed_path and inp_me and not preproc_mes_path:
        preproc_mes_path = \
            model_config['{}_split_preprocessed_motion_embeddings_path'.format(
                    aa.action)]
        if not preproc_mes_path:
            preproc_mes_path = os.path.join(
                    model_config['preprocessed_motion_embeddings_base_path'],
                    'train' if aa.action == 'train' else 'test')
        if not preproc_mes_path:
            print("Preprocessed Motion Embeddings directory (for training) not provided.")
            exit(1)
    if aa.validation_preprocessed_path and inp_me and not val_preproc_mes_path:
        val_preproc_mes_path = None
        val_split_path = model_config['validation_split_preprocessed_motion_embeddings_path']
        if val_split_path:
            val_preproc_mes_path = val_split_path
        else:
            val_base_path = model_config['validation_preprocessed_motion_embeddings_base_path']
            if not val_base_path:
                val_base_path = model_config['preprocessed_motion_embeddings_base_path']

            if val_base_path:
                val_preproc_mes_path = os.path.join(
                    model_config['validation_preprocessed_motion_embeddings_base_path'],
                        'validation')
        if not val_preproc_mes_path:
            print("Preprocessed Motion Embeddings directory (for validation) not provided.")
            exit(1)

    if aa.preproc_babel_path_base:
        preproc_babel_path = osp.join(aa.preproc_babel_path_base, aa.action)
    else:
        preproc_babel_path = None

    if aa.preprocessed_path and inp_me:
        lg.info("Loading preprocessed MEs from '%s'", preproc_mes_path)
    if aa.validation_preprocessed_path and inp_me:
        lg.info("Loading preprocessed MEs (validation) from '%s'", val_preproc_mes_path)
    if aa.action == 'test' and preproc_babel_path:
        lg.info("Loading preprocessed BABEL data from '%s'", preproc_babel_path)

    # Select Program Action
    if aa.action == 'train':
        if aa.preprocessed_path:
            shuffle_opts = model_config['shuffle_opts']
            if aa.shuffle != 'none':
                if aa.shuffle == 'batches':
                    shuffle_opts['shuffle_batches'] = True
                elif aa.shuffle == 'windows':
                    if aa.shuffle_n_windows_cutoff > 0:
                        shuffle_opts['n_windows'] = aa.shuffle_n_windows_cutoff
                    shuffle_opts['shuffle_windows'] = True
                elif aa.shuffle == 'all':
                    shuffle_opts['shuffle_everything'] = True

                if aa.shuffle_n_windows_cutoff > 0:
                    shuffle_opts['n_windows'] = aa.shuffle_n_windows_cutoff

                if aa.shuffle_n_windows_random:
                    shuffle_opts['sample_windows_randomly'] = True

                shuffle_opts['shuffle_windows_offsets_every_n_epochs'] = \
                    aa.shuffle_windows_offsets

            train_dataloader = load_preprocessed_dataloader('train', aa.preprocessed_path,
                    model_config=model_config,
                    shuffle_opts=shuffle_opts, window_sz=loading_window_sz, batch_sz=batch_size,
                    batch_pruning_thresh=aa.batch_pruning_thresh,
                    motion_embeddings_preprocessed_dir=preproc_mes_path if inp_me else None,
                    babel_preprocessed_dir=preproc_babel_path)
            val_dataloader = None
            if aa.validation_preprocessed_path:
                shuffle_opts_val = {}
                if aa.shuffle_validation:
                    shuffle_opts_val['shuffle_everything'] = True
                val_dataloader = load_preprocessed_dataloader('test',
                        aa.validation_preprocessed_path,
                        model_config=model_config,
                        shuffle_opts=shuffle_opts, window_sz=loading_window_sz, batch_sz=batch_size,
                        batch_pruning_thresh=aa.batch_pruning_thresh_validation,
                        motion_embeddings_preprocessed_dir=val_preproc_mes_path if inp_me else None,
                        babel_preprocessed_dir=osp.join(aa.preproc_babel_path_base, 'validation')
                            if preproc_babel_path else None)
            model.train(config, train_dataloader,
                    validation_data_loader=val_dataloader,
                    checkpoints_save_dir=aa.checkpoints_save_directory,
                    checkpoint_path=checkpoint, nepochs=nepochs, window_sz=window_sz)
        else:
            printerr("Path to preprocessed files directory (via --preprocessed) not provided!")
            exit(1)
    elif aa.action == 'test':
        def find_amass_files():
            amass_fps = []
            for amass_path in aa.amass_paths:
                if os.path.isfile(amass_path):
                    amass_fps.append(amass_path)
                else: # isdir
                    for root, _dirnames, fns in os.walk(amass_path, followlinks=True):
                        for fn in fns:
                            if os.path.splitext(fn)[1] == ".npz":
                                amass_fps.append(os.path.join(root, fn))
            return amass_fps

        if aa.test_quantitatively or aa.preprocessed_path:
            if aa.preprocessed_path:
                shuffle_opts_test = model_config['shuffle_opts_test']
                if aa.shuffle_test:
                    shuffle_opts_test['shuffle_everything'] = True
                dataloader = load_preprocessed_dataloader('test', aa.preprocessed_path,
                        model_config=model_config, window_sz=loading_window_sz, batch_sz=batch_size,
                        batch_pruning_thresh=aa.batch_pruning_thresh,
                        exhaustive=aa.every_possible_window,
                        shuffle_opts=shuffle_opts_test,
                        motion_embeddings_preprocessed_dir=
                            preproc_mes_path if use_me else None,
                        babel_preprocessed_dir=preproc_babel_path)
                dset_path_index = -5
            else:
                amass_fps = find_amass_files()
                dataloader = load_amass_dataloader('test', amass_fps, config=config,
                        rotate_y=False)
                dset_path_index = -3

            if aa.dset_path_index is not None:
                dset_path_index = aa.dset_path_index

            model.test(config, dataloader, checkpoint,
                    results_save_dir=aa.test_results_directory, window_sz=window_sz,
                    dset_path_index=dset_path_index,
                    # TODO implement for qualitative testing
                    motion_embedding_test_noise_scale=aa.motion_embedding_test_noise_scale)
        elif aa.amass_paths is not None and len(aa.amass_paths) > 0:
            import viz
            import numpy as np

            amass_fps = find_amass_files()
            lg.info("Found %d test AMASS motion files:\n%s", len(amass_fps),
                    "- " + "\n- ".join(amass_fps))

            viz.init_viz(config)
            dataloader = load_amass_dataloader('test', amass_fps, config=config,
                    rotate_y=False, viz_global_pose=aa.viz_global_pose)

            if model_config['loss'] == 'MPJPE':
                # This actually does not fully solve the problem described in the
                # comments in data_wrangling.mpjpe
                rf2c = False
            else:
                rf2c = True

            do_save_as_video = aa.save_as_video is not None
            do_save_gifs = aa.gifs_dir is not None
            vids_dir = None
            vid_path = None
            vids_pathdepth_inp = None
            if do_save_as_video:
                vids_dir = aa.save_as_video[0]
                if osp.splitext(vids_dir)[1]: # has ext, i.e., likely not a dir
                    vid_path = vids_dir
                    vids_dir = None
                else:
                    vids_pathdepth_inp = int(aa.save_as_video[1]) if len(aa.save_as_video) > 1 else 0
                    os.makedirs(vids_dir, exist_ok=True)
            gifs_dir = None
            gifs_pathdepth_inp = None
            if do_save_gifs:
                gifs_dir = aa.gifs_dir[0]
                gifs_pathdepth_inp = int(aa.gifs_dir[1])

            def callback(pred_j, outp_j, meta):
                lg.info("[{}/{}] {}".format(
                    meta['index']+1, len(amass_fps), amass_fps[meta['index']]))

                assert pred_j.shape[0] == 1 and outp_j.shape[0] == 1
                pred_j, outp_j = pred_j[0], outp_j[0]

                # Determine where to save video
                vid_outpath = None
                gifs_outpath = None
                if vid_path is not None:
                    vid_outpath = vid_path
                else:
                    if do_save_as_video:
                        if vids_pathdepth_inp == 0:
                            vid_dir = vids_dir
                        else:  # vids_pathdepth_inp > 0
                            inp_dir_split = osp.normpath(
                                    osp.dirname(meta['data']['path'])).split(os.sep)
                            if vids_pathdepth_inp < 0 or vids_pathdepth_inp > len(inp_dir_split):
                                raise ValueError("Invalid video path depth: {}".format(vids_pathdepth_inp))
                            vid_dir = os.sep.join(inp_dir_split[-vids_pathdepth_inp:])
                            vid_dir = os.path.join(vids_dir, vid_dir)
                        os.makedirs(vid_dir, exist_ok=True)
                        vid_outpath = os.path.join(vid_dir,
                                os.path.splitext(os.path.basename(meta['data']['path']))[0] + ".mp4")
                if do_save_gifs:
                    inp_dir_split = osp.normpath(
                            osp.dirname(meta['data']['path'])).split(os.sep)
                    gif_dir = os.sep.join(inp_dir_split[-gifs_pathdepth_inp:])
                    gif_dir = os.path.join(gifs_dir, gif_dir)
                    os.makedirs(gif_dir, exist_ok=True)
                    gif_outpath = os.path.join(gif_dir,
                            os.path.splitext(os.path.basename(meta['data']['path']))[0] + ".gif")

                if aa.dump_npz:
                    npz_outpath = osp.splitext(vid_outpath)[0] + ".npz"
                    nframes_og = meta['data']['nframes']
                    pred_rmats = meta['pred_rot_mats']
                    outp_rmats = meta['outp_rot_mats']
                    assert len(pred_rmats.shape) == len(outp_rmats.shape) and \
                        all(map(lambda i: pred_rmats.shape[i] == outp_rmats.shape[i],
                                    range(len(pred_rmats.shape))))
                    # (nb x)? nf x nj x 3 x 3
                    nframes_out = pred_rmats.shape[-4]
                    assert nframes_out <= nframes_out

                    # Recover transl
                    if not aa.viz_global_pose:
                        raise ValueError("Cannot recover transl because global pose is not calculated")

                    j22_local = meta['outp_joints']
                    # nb=1 x nf_og x nj=22 x 3
                    assert len(j22_local.shape) == 4
                    rj_local = j22_local[0, :, 0, :]
                    transl = outp_j[..., 0, :] - rj_local

                    dump_data = {
                        'pred_rot_mats': pred_rmats,
                        'outp_rot_mats': outp_rmats,
                        'transl': transl,

                        'path': meta['data']['path'],
                        'start_frame': nframes_og - nframes_out,
                    }
                    if 'raw_me_unperturbed_torch' in meta['data']:
                        dump_data['me'] = meta['data']['raw_me_unperturbed_torch'].detach().cpu().numpy()
                    lg.info("Exporting %s to '%s'...", dump_data.keys(), npz_outpath)
                    with open(npz_outpath, 'wb') as f:
                        np.savez(f, **dump_data)
                    return

                if not aa.qualitative_oui_disable and not aa.viz_clear:
                    losses_to_viz = ['me_err', 'mpjpe', 'mpjpe_upper_body', 'mpjpe_legs', 'mpjpe_global']
                    pf_losses = meta['perframe_losses']
                    ma_losses = meta['perframe_ma_losses']
                    # TODO
                    #ma_losses_ws = aa.qualitative_oui_ma_window_size
                    #if ma_losses_ws <= 0:
                    #    ma_losses_ws = loading_window_sz
                    #ma_losses = {}
                    #for lossname, pfl in pf_losses.items():
                    #    if lossname not in losses_to_viz: continue #TODO lossname is actually current frame
                    #    mal = np.cumsum(pfl)
                    #    mal[ma_losses_ws:] = mal[ma_losses_ws:] - mal[:-ma_losses_ws]
                    #    mal /= ma_losses_ws
                    #    ma_losses[lossname] = mal
                else:
                    losses_to_viz = []
                    pf_losses = None
                    ma_losses = None

                viz_meta = viz.test_one(config, checkpoint_path=checkpoint,
                        name=name, pred_n_outp_joints=(pred_j, outp_j),
                        nframes=None if not aa.frames else
                            (aa.frames[0] if len(aa.frames) == 1 else aa.frames[i]),
                        save_as_video=do_save_as_video, video_path=vid_outpath,
                        viz_losses=losses_to_viz,
                        perframe_losses=pf_losses,
                        ma_losses=ma_losses, ma_losses_desc="MA",
                        clear=aa.viz_clear)

                if do_save_gifs:
                    gif_outpath_noext = osp.splitext(gif_outpath)[0]
                    gif_outpath_gt = gif_outpath_noext + "_gt.gif"
                    gif_outpath_pred = gif_outpath_noext + "_pred.gif"
                    shutil.copy(viz_meta['gt_anim_path'], gif_outpath_gt)
                    shutil.copy(viz_meta['pred_anim_path'], gif_outpath_pred)
                    lg.info("GIFs saved in: '%s'", osp.dirname(gif_outpath))

            model.test(config, dataloader, checkpoint,
                    results_save_dir=aa.test_results_directory, window_sz=window_sz,
                    inference_callback=callback, infer_global_pose=aa.viz_global_pose,
                    dset_path_index=aa.dset_path_index if aa.dset_path_index is not None else -3)
        elif aa.compare_test_results:
            if not aa.comparison_test_results_directory:
                lg.error("Please provide --comparison_test_results_directory in which "
                        "the comparison test results could be saved.")
                exit(1)

            result_files = []
            for rf in aa.test_result_files or []:
                if not os.path.isfile(rf):
                    lg.error("\"%s\" is not a valid file!", rf)
                else:
                    result_files.append(rf)

            if aa.test_results_directory:
                if not os.path.isdir(aa.test_results_directory):
                    lg.error("\"%s\" is no a valid directory!", aa.test_results_directory)
                else:
                    for fn in os.listdir(aa.test_results_directory):
                        fp = os.path.join(aa.test_results_directory, fn)
                        if os.path.isfile(fp):
                            if os.path.splitext(fp)[1] == ".pkl":
                                result_files.append(fp)
                        else:  # isdir
                            inner_trfp = os.path.join(fp, model_config['test_results_raw_fn'])
                            if os.path.isfile(inner_trfp):
                                result_files.append(inner_trfp)

            lg.info("Found %d test results to compare:\n%s", len(result_files),
                    "- " + "\n- ".join(result_files))

            model.analyse_test_results(result_files, save_dir=aa.comparison_test_results_directory)
        else:
            printerr("You must provide either a path to preprocessed files (via --preprocessed) or path(s) to AMASS motion file(s) (via --amass) or set --compare_test_results.")
            exit(1)
    elif aa.action == 'preprocess':
        from preprocessing import preprocess, preprocess_babel

        if aa.babel_data:  # Preprocess BABEL data
            print("Preprocessing BABEL data")
            print("Preprocessed batches dir: '{}'".format(aa.preprocessed_path))
            print("Destination preprocessed BABEL batches dir: '{}".format(
                    aa.preproc_babel_path_base))
            if not osp.isdir(aa.preprocessed_path):
                printerr("Preprocessed batches directory does not exist.")
                exit(1)
            if not osp.isdir(aa.babel_data):
                printerr("Please provide valid BABEL data path: '{}'".format(aa.babel_data))
                exit(1)

            preprocess_babel(batches_base_dir=aa.preprocessed_path,
                    output_base_dir=aa.preproc_babel_path_base,
                    babel_data_dir=aa.babel_data)
        else:
            if aa.body_data_path:
                with open(aa.body_data_path) as f:
                    body_datas = json.load(f)
            else:
                body_datas = data_loading.load_body_datas(config, load_additional_datas=True)

            preprocess(config, body_datas=body_datas, n_body_datas_per_batch=batch_size,
                    save_in_batch=True,
                    sort_body_datas_by_length=not aa.dont_sort_by_length,
                    debug=aa.debug, save_dir=aa.preprocessed_path)
    elif aa.action == 'ops':
        import tests

        if aa.debug:
            breakpoint()

        getattr(tests, aa.operation)(config=config)

        if aa.debug:
            breakpoint()
    else:
        exit(1)


if __name__ == '__main__':
    main()
