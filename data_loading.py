from util import glg, get_root_dir, load_smpl_model, dtype_np, dtype_torch, get_device

import numpy as np
import torch
from tqdm import tqdm

import json
import os
import random
import math
import functools
from collections import namedtuple, OrderedDict


babel_global_d = {}


def init(cuda_index):
    global get_device
    get_device = functools.partial(get_device, cuda_index=cuda_index)


def load_me_batches(me_fp):
    me_batch_d = np.load(me_fp, allow_pickle=True)
    if isinstance(me_batch_d, np.ndarray):
        me_batch = me_batch_d
        me_mu_batch = me_batch
        me_logvar_batch = np.full_like(me_batch, float('-inf'))
    else:
        #me_batch = me_batch_d['me']  # too much RAM needed
        me_mu_batch = me_batch_d['mu']
        me_batch = me_mu_batch
        me_logvar_batch = me_batch_d['logvar']
    return me_batch, me_mu_batch, me_logvar_batch


def update_babel_global_d(np_obj, force_update=False):
    if force_update or 'act_labels' not in babel_global_d:
        babel_global_d['act_labels'] = np_obj['act_labels']
        babel_global_d['act_counts'] = np_obj['act_counts']
        babel_global_d['act_counts_w'] = np_obj['act_counts_w']


class Data2:
    """
    Data loading class that loads batches of continuous preprocessed data
    that is meant as a more efficient alternative to Data
    """

    def __init__(self, model_config, preprocessed_to_model_inout_func,
            training_data_dir=None, test_data_dir=None,
            training_datas=None, test_datas=None, *,
            load_motion_embeddings=False,
            training_motion_embeddings_data_dir=None,
            test_motion_embeddings_data_dir=None,
            load_babel_data=False,
            training_babel_data_dir=None,
            test_babel_data_dir=None):
        self.train_pct = model_config['training_data_percentage']
        self.model_config = model_config
        self.preprocessed_to_model_inout_func = preprocessed_to_model_inout_func
        self.training_data_dir = training_data_dir
        self.test_data_dir = test_data_dir
        self.training_datas = training_datas or []
        self.test_datas = test_datas or []

        self.load_me = load_motion_embeddings
        self.training_me_data_dir = training_motion_embeddings_data_dir
        self.test_me_data_dir = test_motion_embeddings_data_dir

        self.load_babel = load_babel_data
        self.training_babel_data_dir = training_babel_data_dir
        self.test_babel_data_dir = test_babel_data_dir

        self.train_load_counter = 0

    def dataloader(self, train_or_test, shuffle_opts=None, **kw):
        is_training = train_or_test == 'train'
        if train_or_test == 'train':
            is_training = True
            data_dir = self.training_data_dir
            data_list = self.training_datas
            me_data_dir = self.training_me_data_dir
            babel_data_dir = self.training_babel_data_dir
        elif train_or_test == 'test':
            is_traning = False
            data_dir = self.test_data_dir
            data_list = self.test_datas
            me_data_dir = self.test_me_data_dir
            babel_data_dir = self.test_babel_data_dir
        else:
            raise ValueError()

        shuffle_opts_default = {
            'shuffle_batches': False,
            'shuffle_windows': False,
            'n_windows': None,
            'sample_windows_randomly': False,
            # This option will load all the batches into RAM once.
            # Use at own risk.
            'shuffle_everything': False,
            'shuffle_windows_offsets_every_n_epochs': 8 if train_or_test == 'train' else 0,
        }
        shuffle_opts = shuffle_opts or {}
        for k, v in shuffle_opts_default.items():
            if k not in shuffle_opts:
                shuffle_opts[k] = v

        window_sz = kw.get('window_sz', self.model_config['loading_window_size'])
        me_len = 0
        if 'motion_embedding_length' in kw:
            me_len = kw['motion_embedding_len']
        elif self.load_me:
            me_len = self.model_config['motion_embedding_length']

        if shuffle_opts['shuffle_everything']:
            glg().info("Loading dataset...")
            torch_dset = TorchDataset(data_dir,
              window_sz=window_sz,
              preprocessed_to_model_inout_func=self.preprocessed_to_model_inout_func,
              nwindows_cutoff_per_file=shuffle_opts['n_windows'],
              nwindows_cutoff_randomise=shuffle_opts['sample_windows_randomly'],
              exhaustive=kw.get('exhaustive', False),
              load_motion_embeddings=self.load_me,
              motion_embedding_length=me_len,
              motion_embedding_batches_dir=me_data_dir,
              load_babel_data=self.load_babel,
              babel_batches_dir=babel_data_dir)

            if not is_training:
                random.seed(1)
                torch.manual_seed(1)
                torch.cuda.manual_seed(1)
                np.random.seed(1)

            def collate(d):
                return list(zip(*d))
            torch_dloader = torch.utils.data.DataLoader(torch_dset,
                    batch_size=kw.get('batch_sz', self.model_config['batch_size']),
                        collate_fn=collate, shuffle=True)
        else:
            torch_dset = None
            torch_dloader = None


        def load_data_closure(**kw2):
            """
                batch_pruning_thresh: value in [0, 1]. 1.0 to keep all (ie no pruning).
                                e.g., if value is 0.95, when the # of motions that have
                                ended exceeds (1 - 0.95) = 5%, the remaining frames for all
                                motions will be culled from the batch.
            """
            nonlocal window_sz
            window_sz = kw2.get('window_sz') or window_sz
            if is_training:
                self.train_load_counter += 1
            # Load from data dir
            if data_dir:
                if not shuffle_opts['shuffle_everything']:
                    data_dir_ls = os.listdir(data_dir)
                    if shuffle_opts['shuffle_batches']:
                        random.shuffle(data_dir_ls)
                    for i_file, batch_fn in enumerate(data_dir_ls):
                        if not batch_fn.startswith('batch'):
                            continue
                        batch_fp = os.path.join(data_dir, batch_fn)
                        glg().info("[%d/%d] loading batch: %s (ws=%d)",
                                i_file+1, len(data_dir_ls), batch_fp, window_sz or -1)
                        with np.load(batch_fp) as batch:
                            inp, outp, intm_vals = self.preprocessed_to_model_inout_func(batch,
                                    return_intermediate_values=True, train_or_test=train_or_test)
                            nfrms_max = intm_vals['max_frames']

                            if self.load_me:
                                me_fn = "me_" + batch_fn
                                me_fp = os.path.join(me_data_dir, me_fn)
                                me_batch, me_mu_batch, me_logvar_batch = load_me_batches(me_fp)
                                me_dim = me_batch.shape[-1]
                            else:
                                me_batch = None
                                me_mu_batch = None
                                me_logvar_batch = None

                            if self.load_babel:
                                babel_fn = "babel_" + batch_fn
                                babel_fp = os.path.join(babel_data_dir, babel_fn)
                                babel_d = np.load(babel_fp)
                                act_cat = babel_d['act_cat']
                                update_babel_global_d(babel_d)
                            else:
                                act_cat = None

                            batch_pruning_thresh = kw.get('batch_pruning_thresh', 0.95) or 0.95
                            nfrms_max_old = nfrms_max
                            nb = batch['poses'].shape[0]
                            batch_nframes = batch['nframes']
                            if batch_pruning_thresh < 1.0:
                                # Assumption: batch elements are sorted by length (shortest to longest)
                                #assert batch_nframes[0] <= batch_nframes[-1]  # Error may exist from framerate normalisation
                                cutoff_idx = math.floor((nb - 1) * (1 - batch_pruning_thresh))
                                nfrms_max_new = batch_nframes[cutoff_idx]
                                glg().debug(("Batch pruned by {:.00f}%: at index {}/{}, "
                                            "{} -> {} ({:.01f}% retrained)").format(
                                            100 * (1 - batch_pruning_thresh),
                                            cutoff_idx, nb, nfrms_max, nfrms_max_new,
                                            100 * nfrms_max_new / nfrms_max))
                                glg().debug(("nwindows (size={}): {} -> {} "
                                            "({:.01f}% windows retained)").format(
                                            window_sz, nfrms_max // window_sz,
                                            nfrms_max_new // window_sz,
                                            100 * ((nfrms_max_new // window_sz) / (nfrms_max // window_sz))))
                                nfrms_max = nfrms_max_new

                            nfrms_max = min(nfrms_max_old, max(nfrms_max, window_sz))
                            nwindows = nfrms_max // window_sz

                            if nwindows <= 0:
                                glg().warning("No windows generated from batch: %s"
                                        " maxframes: %d, window_sz: %d",
                                        batch_fp, nfrms_max, window_sz)

                            if is_training:
                                windows_inds = list(range(nwindows))
                                if shuffle_opts['shuffle_windows']:
                                    windows_inds = random.shuffle(window_inds)
                                if shuffle_opts['n_windows'] is not None:
                                    windows_inds = windows_inds[:shuffle_opts['n_windows']]
                            else:
                                windows_inds = range(nfrms_max - window_sz)

                            for i in windows_inds:
                                if is_training:
                                    start_frm = i * window_sz
                                else:
                                    start_frm = i
                                end_frm_p1 = start_frm + window_sz
                                if end_frm_p1 - 1 > nfrms_max - 1:
                                    continue

                                intm = {}
                                for k, v in intm_vals['batch'].items():
                                    intm[k] = v[:, start_frm:end_frm_p1]

                                yield {
                                    'path': batch['paths'],
                                    'gender': batch['genders'],
                                    'nframes': batch_nframes,
                                    'start_frame': np.full_like(batch_nframes, start_frm),
                                    'end_frame': np.full_like(batch_nframes, end_frm_p1 - 1),
                                    'input': inp[:, start_frm:end_frm_p1],
                                    'output': outp[:, start_frm:end_frm_p1],
                                    'me': me_batch[:, start_frm:end_frm_p1]
                                        if me_batch is not None else None,
                                    'me_mu': me_mu_batch[:, start_frm:end_frm_p1]
                                        if me_mu_batch is not None else None,
                                    'me_logvar': me_logvar_batch[:, start_frm:end_frm_p1]
                                        if me_logvar_batch is not None else None,
                                    'act_cat': act_cat[:, start_frm:end_frm_p1]
                                        if act_cat is not None else None,
                                    # TODO this might be problematic if windows are shuffled
                                    'file_idx': i_file,
                                    'intm': intm,
                                    'intm_vals': intm_vals
                                }
                else:  # shuffle everything; using torch.nn.DataLoader
                    if is_training:
                        shufoffsets = shuffle_opts['shuffle_windows_offsets_every_n_epochs']
                        if (shufoffsets or 0) > 0 and (self.train_load_counter > 0 and
                                self.train_load_counter % shufoffsets == 0):
                            glg().info("shifting offsets... (shufoffsets={}, train_load_counter={})".format(shufoffsets, self.train_load_counter))
                            torch_dset.reload_shifted()
                    
                    batchsz = torch_dloader.batch_size
                    nbatches = len(torch_dloader)
                    ndata = batchsz * nbatches
                    glg().info(("({}) Start loading data from torch dataloader: "
                                "{}bx{}={} (ws={})").format(
                                train_or_test, batchsz, nbatches, ndata, window_sz))

                    for ibatch, tup in enumerate(tqdm(torch_dloader,
                                desc="{}bx{}={}".format(batchsz, nbatches, ndata))):
                        d = {}
                        for i in range(len(tup)):
                            d[torch_dset.keys[i]] = tup[i]
                        d['input'] = np.array(d['input'])
                        d['output'] = np.array(d['output'])
                        if self.load_me:
                            d['me'] = np.array(d['me'])
                            if 'me_mu' in d:
                                d['me_mu'] = np.array(d['me_mu'])
                            if 'me_logvar' in d:
                                d['me_logvar'] = np.array(d['me_logvar'])
                        if self.load_babel:
                            d['act_cat'] = np.array(d['act_cat'])
                        # Basically, there is no feasible way to keep track of files,
                        # so we just assign a unique value each time.
                        d['file_idx'] = ibatch
                        # TODO Optimise this
                        intm_stacked = {}
                        if 'intm' in d:
                            intm = d['intm']
                            for intm_k in d['intm'][0].keys():
                                intm_stacked[intm_k] = \
                                    np.array([intm_e[intm_k] for intm_e in intm])
                        d['intm'] = intm_stacked
                        yield d
            # Load from data list
            for i_data, data in enumerate(data_list):
                inp, outp, intm_vals = self.preprocessed_to_model_inout_func(data,
                        return_intermediate_values=True, train_or_test=train_or_test)
                d = {
                    'path': data.get('path', "#{}".format(i_data+1)),
                    'gender': data.get('gender', 'male'),
                    'input': inp,
                    'output': outp,
                    'intm_vals': intm_vals,
                    'intm': intm_vals['batch']
                }
                if 'max_frames' in intm_vals:
                    d['nframes'] = intm_vals['max_frames']
                yield d

        return load_data_closure


class TorchDataset(torch.utils.data.Dataset):
    keys = ['path', 'gender', 'nframes', 'start_frame', 'end_frame',
        'input', 'output', 'file_idx', 'intm']
    def __init__(self, batches_dir, window_sz, preprocessed_to_model_inout_func, *,
            exhaustive=False,
            load_motion_embeddings=False,
            nwindows_cutoff_per_file=None, nwindows_cutoff_randomise=True,
            motion_embedding_length=64, motion_embedding_batches_dir=None,
            incomplete_motion_embeddings_policy='randomise',
            load_babel_data=False, babel_batches_dir=None, **kw):
        # Create one large list
        self.batches_dir = batches_dir
        self.preprocessed_to_model_inout_func = preprocessed_to_model_inout_func
        self.window_sz = window_sz
        self.exhaustive = exhaustive
        self.nwindows_cutoff_per_file = nwindows_cutoff_per_file
        self.nwindows_cutoff_randomise = nwindows_cutoff_randomise
        self.slices = []

        self.use_me = load_motion_embeddings
        self.me_len = motion_embedding_length
        self.me_batches_dir = motion_embedding_batches_dir
        self.incomplete_me_policy = incomplete_motion_embeddings_policy
        if self.use_me and 'me' not in self.keys:
            i = self.keys.index('output') + 1
            self.keys.insert(i, 'me')
            self.keys.insert(i+1, 'me_mu')
            self.keys.insert(i+2, 'me_logvar')

        self.use_babel = load_babel_data
        self.babel_batches_dir = babel_batches_dir
        if self.use_babel and 'act_cat' not in self.keys:
            i = self.keys.index('output' if not self.use_me else 'me_logvar') + 1
            self.keys.insert(i, 'act_cat')

        self.load_more_windows = kw.get('load_more_windows', False)

        self.reload_shifted(shift=0)
        self.random_shifts = None


    def reload_shifted(self, shift=None):
        glg().info("Reloading all batches with window offsets %s",
                "randomly shifted" if shift is None else "shifted by {}".format(shift))
        self.slices.clear()
        filecounter = 0
        batch_fns = os.listdir(self.batches_dir)
        shiftval = shift
        window_sz = self.window_sz
        for ibfile, fn in enumerate(batch_fns):
            if not fn.startswith("batch"):
                continue
            fp = os.path.join(self.batches_dir, fn)
            glg().info("Processing batch %d/%d:\n\t%s", ibfile+1, len(batch_fns), fp)
            with np.load(fp) as batch:
                inp_b, outp_b, intm_vals = self.preprocessed_to_model_inout_func(batch,
                        return_intermediate_values=True)
                nb = len(batch['poses'])
                frs_b = batch['nframes']
                paths = batch['paths']
                genders = batch['genders']
                slices_curbatch = []
                intm_batch = OrderedDict()

                if self.use_me:
                    me_fn = "me_" + fn
                    me_fp = os.path.join(self.me_batches_dir, me_fn)
                    me_batch, me_mu_batch, me_logvar_batch = load_me_batches(me_fp)
                    me_dim = me_batch.shape[-1]
                else:
                    me_batch = None
                    me_mu_batch = None
                    me_logvar_batch = None
                    me_dim = -1

                if self.use_babel:
                    babel_fn = "babel_" + fn
                    babel_fp = os.path.join(self.babel_batches_dir, babel_fn)
                    babel_d = np.load(babel_fp)
                    act_cat_batch = babel_d['act_cat']
                    update_babel_global_d(babel_d)
                else:
                    act_cat_batch = None

                # Iterate within batch
                for ib in tqdm(list(range(nb)),
                        desc="B{}/{}".format(ibfile+1, len(batch_fns))):
                    nfrms = frs_b[ib]
                    pth = paths[ib]
                    gen = genders[ib]
                    if not self.exhaustive:
                        nwinds = nfrms // window_sz
                    else:
                        nwinds = nfrms - window_sz + 1
                    inp = inp_b[ib]
                    outp = outp_b[ib]
                    me = None
                    me_mu = None
                    me_logvar = None
                    ac = None
                    if me_batch is not None:
                        me = me_batch[ib]
                        me_mu = me_mu_batch[ib]
                        me_logvar = me_logvar_batch[ib]
                    if act_cat_batch is not None:
                        ac = act_cat_batch[ib]

                    for k, v in intm_vals['batch'].items():
                        intm_batch[k] = v[ib]
                    if 'me_inp' in intm_vals:
                        intm_batch['me_inp'] = intm_vals['me_inp'][ib]

                    sls = []
                    cutoff = nwinds
                    if (self.nwindows_cutoff_per_file is not None and
                            not self.nwindows_cutoff_randomise):
                        cutoff = min(nwinds, self.nwindows_cutoff_per_file)
                    if shift is None:
                        shiftval = random.randint(0, window_sz - 1)
                    for iw in range(cutoff):
                        if not self.exhaustive:
                            st = iw * window_sz + shiftval
                        else:
                            st = iw
                        edp1 = st + window_sz
                        if self.exhaustive or self.load_more_windows:
                            if edp1 - 1 > nfrms - 1:
                                continue
                        else: # Kept for backward compatibility
                            if edp1 >= nfrms:
                                continue

                        # Handle incomplete ME
                        if me is not None and st < self.me_len:
                            if self.incomplete_me_policy == 'randomise':
                                me[:self.me_len-st] = np.random.normal(size=me_dim).astype(dtype_np)
                            elif self.incomplete_me_policy == 'zero':
                                me[:self.me_len-st] = np.zeros(me_dim, dtype=dtype_np)

                            me_mu[:self.me_len-st] = me[:self.me_len-st]
                            me_logvar[:self.me_len-st] = \
                                np.full_like(me[:self.me_len-st], float('-inf'))

                        # Construct slice
                        curslice = [pth, gen, nfrms, st, edp1-1, inp[st:edp1], outp[st:edp1]]
                        if self.use_me:
                            curslice.extend([me[st:edp1], me_mu[st:edp1], me_logvar[st:edp1]])
                        if self.use_babel:
                            curslice.append(ac[st:edp1])
                        curslice.extend([filecounter,
                                {k: v[st:edp1] for k, v in intm_batch.items()}])
                        sls.append(curslice)
                    if (self.nwindows_cutoff_per_file is not None
                            and self.nwindows_cutoff_randomise):
                        random.shuffle(sls)
                        sls = sls[:self.nwindows_cutoff_per_file]
                    slices_curbatch.extend(sls)
                    filecounter += 1
                glg().info("%d slices collected", len(slices_curbatch))
                self.slices.extend(slices_curbatch)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return self.slices[idx]


def load_body_datas(config, root_dir=None, body_data_filter=None, load_additional_datas=False,
        json_save_path=None, prefer_cached=False):
    """
    body_data_filter: filter func that accepts bdata dict
    evaluated inside initial loop.
    load_additional_datas: if set to True, actual AMASS data files will be open for more metadata
    json_save_path: if set to a valid path, data will be saved to it,
    unless it already exists AND prefered_cached is True.
    prefer_cached: if json_save_path is specified and points to an existing file and
    this flag is set, the file will be loaded and returned instead.
    """
    if prefer_cached and json_save_path and os.path.isfile(json_save_path):
        glg().info("Loading cached body data from \"%s\"...", json_save_path)
        with open(json_save_path) as f:
            return json.load(f)

    root_dir = root_dir or get_root_dir(config, 'body_data')
    body_datas = []
    glg().info("Loading body data from: %s", root_dir)
    for dset_shortname in os.listdir(root_dir):
        glg().info(dset_shortname)
        dset_path = os.path.join(root_dir, dset_shortname)
        if not os.path.isdir(dset_path):
            continue

        for dset_type_dirname in os.listdir(dset_path):
            if dset_type_dirname.startswith("SMPLpH"):
                dset_type = 'smplh'
            elif dset_type_dirname.startswith("SMPL-X"):
                dset_type = 'smplx'
            else:
                raise ValueError("Unknown dataset type \"{}\" ({})".format(
                    dset_type_dirname, dset_path))
            dset_type_path = os.path.join(dset_path, dset_type_dirname)
            for subjects_dirs_cont_dirname in os.listdir(dset_type_path):
                subjects_dirs_cont_path = os.path.join(dset_type_path, subjects_dirs_cont_dirname)
                if not os.path.isdir(subjects_dirs_cont_path):
                    continue
                for subject in os.listdir(subjects_dirs_cont_path):
                    subject_dirpath = os.path.join(subjects_dirs_cont_path, subject)
                    if not os.path.isdir(subject_dirpath):
                        continue
                    for bdata_fn in os.listdir(subject_dirpath):
                        bdata_name, ext = os.path.splitext(bdata_fn)
                        if ext.lower() != '.npz':
                            continue

                        bdata_path = os.path.join(subject_dirpath, bdata_fn)

                        bdata = {
                            'dataset_type': dset_type,
                            'dataset_shortname': dset_shortname,
                            'subject': subject,
                            'path': bdata_path
                        }
                        if body_data_filter is None or body_data_filter(bdata):
                            body_datas.append(bdata)

    if load_additional_datas:
        glg().info("Loading additional data...")
        cull_list = []
        for ibdata_info, bdata_info in enumerate(tqdm(body_datas)):
            with np.load(bdata_info['path']) as bdata:
                if 'poses' in bdata:
                    bdata_info['n_frms'] = bdata['poses'].shape[0]
                    bdata_info['gender'] = str(bdata['gender'])
                    bdata_info['framerate'] = int(
                            bdata.get('mocap_framerate', bdata.get('mocap_frame_rate', None)))
                else:
                    cull_list.append(ibdata_info)
        body_datas = [e for i, e in enumerate(body_datas) if i not in cull_list]

    if json_save_path:
        with open(json_save_path, 'w') as f:
            json.dump(body_datas, f)

    glg().info("Done")

    return body_datas


def load_torch_model(model, checkpoint_path, strict=True, filter_prefix=None):
    if checkpoint_path:
        glg().info("Loading checkpoint: {}".format(checkpoint_path))
        state_d = torch.load(checkpoint_path, map_location=get_device())
        items = list(state_d.items())
        for k, v in list(state_d.items()):
            if k.startswith("lstm."):
                state_d["rnn." + k[len("lstm."):]] = v
                del state_d[k]
            elif k.startswith("linear."):
                state_d["linear_out." + k[len("linear."):]] = v
                del state_d[k]

            for pfx in filter_prefix or []:
                if k.startswith(pfx):
                    del state_d[k]
        out = model.load_state_dict(state_d, strict=strict)
        if not strict:
            return out[0], out[1] # missing_keys, unexpected_keys
    else:
        glg().info("No checkpoint given!")
