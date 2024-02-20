#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import datetime
from builtins import range
from builtins import str
import jax
import torch
from past.utils import old_div
from typing import *
from jax.typing import ArrayLike
from jnormcorre.utils.lazy_array import lazy_data_loader

logging.basicConfig(level=logging.ERROR)
import numpy as np
import tifffile
from typing import List
from jnormcorre.onephotonmethods import get_kernel, high_pass_filter_cv, high_pass_batch
from jnormcorre.utils.lazy_array import lazy_data_loader
from tqdm import tqdm
import math
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import random


class frame_corrector():
    def __init__(self, template: np.ndarray,
                 max_shifts: Tuple[int, int], strides: Tuple[int, int],
                 overlaps: Tuple[int, int], max_deviation_rigid: int,
                 min_mov: Optional[float] = None,
                 batching: int = 100) -> None:
        """
        Standalone motion correction object, allowing users to register frames via rigid or piecewise
        rigid motion correction to a given template.

        Args:
            template (np.ndarray): Shape (d1, d2) where d1 and d2 are the FOV dimensions
            max_shifts (Tuple): Two integers, specifying maximum shift in the two FOV dimensions (height, width)
            strides (Tuple): Two integers, used to specify patch dimensions for pwrigid registration
            overlaps (Tuple): Overlap b/w patches. strides[i] + overlaps[i] are the patch size dimensions.
            max_deviation_rigid (int): Specifies max number of pixels a patch can deviate from the rigid shifts.
            min_mov (float): The minimum value of the movie, if known.
            batching (int): Specifies how many frames we register at a time. Toggle this to avoid GPU OOM errors.
        """
        self.template = template
        self.max_shifts = max_shifts
        self.upsample_factor_fft = 10

        if min_mov is not None:
            self.add_to_movie = -min_mov
        else:
            self.add_to_movie = 0

        self.strides = strides
        self.overlaps = overlaps
        self.max_deviation_rigid = max_deviation_rigid
        self.batching = batching

        # Set the pwrigid function
        self.pw_registration_method = jit(
            vmap(_register_to_template_pwrigid, in_axes=(0, None, None, None, None, None, None, None, None, None)),
            static_argnums=(2, 3, 4, 5, 7))

        def simplified_registration_func_pw(frames: np.ndarray) -> ArrayLike:
            return self.pw_registration_method(frames, self.template, self.strides[0], self.strides[1],
                                               self.overlaps[0], self.overlaps[1], self.max_shifts,
                                               self.upsample_factor_fft, self.max_deviation_rigid, self.add_to_movie)[0]

        self.jitted_pwrigid_method = simplified_registration_func_pw

        # Set the rigid function
        self.rigid_registration_method = jit(vmap(_register_to_template_rigid, in_axes=(0, None, None, None)))

        def simplified_registration_func_rig(frames: np.ndarray) -> ArrayLike:
            return self.rigid_registration_method(frames, self.template, self.max_shifts, self.add_to_movie)[0]

        self.jitted_rigid_method = simplified_registration_func_rig

    def register_frames(self, frames: np.ndarray, pw_rigid: bool = False) -> np.ndarray:
        """
        Function to register a set of frames to this object's template.

        Args:
            frames (np.ndarray): dimensions (T, d1, d2), where T is the number of frames and d1, d2 are FOV dims
            pwrigid (bool): Indicates whether we do piecewise rigid or rigid registration. Defaults to rigid.

        Returns:
            corrected_frames (np.array): Dimensions (T, d1, d2). The registered output from the input (frames)
        """
        if pw_rigid:
            output = self.jitted_pwrigid_method(frames)
        else:
            output = self.jitted_rigid_method(frames)
        return np.array(output)

    @property
    def rigid_function(self) -> Callable[[np.ndarray], ArrayLike]:
        """
        The rigid registration function of this frame correction object
        """
        return self.jitted_rigid_method

    @property
    def pwrigid_function(self) -> Callable[[np.ndarray], ArrayLike]:
        """
        The piecewise rigid registration function of this frame correction object
         """
        return self.jitted_pwrigid_method


def verify_strides_and_overlaps(dim: int, stride: int, overlap: int) -> None:
    if not stride > 0:
        raise ValueError(
            "Stride value needs to be positive. Right now it is {}. See documentation for more details.".format(
                stride))
    if not overlap > 0:
        raise ValueError("Overlap value needs to be positive. Right now it is {}. See documentation".format(overlap))
    if not dim > 0:
        raise ValueError(
            "Dim needs to be positive. Right now the length along this FOV axis is {}. See documentation".format(
                dim))
    if not stride < dim:
        raise ValueError(
            "Stride must be less than the field of view dimension, otherwise this parameter is not meaningful for piecewise-rigid registration. Right now the value of stride is {} and the length of this axis of the FOV is {}. See documentation for more details.".format(
                stride, dim))
    if not overlap < stride:
        raise ValueError(
            "The degree of overlap must be less than the stride for piecewise-rigid registration. Right now, the value of overlap is {} and stride is {}. See documentation for more details".format(
                overlap, stride))
    if not stride + overlap < dim:
        raise ValueError(
            "The stride + overlap (i.e. overall patch size) should be less the length of this axis of the FOV. Right now, stride is {} and overlap is {} and the FOV axis length is {}. See documentation for more details.".format(
                stride, overlap, dim))


class MotionCorrect(object):
    """
    class implementing motion correction operations
    """

    def __init__(self, lazy_dataset: lazy_data_loader, max_shifts: tuple[int, int] = (6, 6),
                 frames_per_split: int = 1000, num_splits_to_process_rig: Optional[int] = None, niter_rig: int = 1,
                 pw_rigid: bool = False, strides: tuple[int, int] = (96, 96), overlaps: tuple[int, int] = (32, 32),
                 max_deviation_rigid: int = 3, num_splits_to_process_els: Optional[int] = None,
                 niter_els: int = 1, min_mov: float = None, upsample_factor_grid: int = 4,
                 gSig_filt: Optional[list[int]] = None, bigtiff: bool = False) -> None:

        """
        Constructor class for motion correction operations

        Args:
            lazy_dataset (lazy_data_loader): Lazy data loader for loading frames of the data
            max_shifts (Tuple): Two integers, specifying maximum shift in the two FOV dimensions (height, width)
            frames_per_split (int): Integer larger than 1. Number of frames we use to generate each local template.
            num_splits_to_process_rig (int): Number of splits we process per iteration of rigid motion correction
            niter_rig (int): Number of iterations of rigid motion correction
            pw_rigid (bool): Whether we additionally run piecewise rigid registration
            strides (Tuple): Two integers, used to specify patch dimensions for pwrigid registration
            overlaps (Tuple): Overlap b/w patches. strides[i] + overlaps[i] are the patch size dimensions.
            max_deviation_rigid (int): Specifies max number of pixels a patch can deviate from the rigid shifts.
            num_splits_to_process_els (int): Number of splits we process per iteration of pwrigid motion correction
            niter_els: Number of iterations of piecewise rigid registration
            min_mov (float). The minimum value of the movie, if known
            gSig_filt (list): List with 1 positive integer describing a Gaussian standard deviation. We use this to construct a kernel to
                high-pass filter data which has large background contamination.
            bigtiff (bool): Indicates whether or not movie is saved as a bigtiff or regular tiff
        """
        if not isinstance(niter_els, int) or niter_els < 1:
            raise ValueError(f"please provide niter_els as an int of 1 or higher.")

        if not isinstance(niter_rig, int) or niter_rig < 1:
            raise ValueError(f"please provide niter_rig as an int of 1 or higher.")

        self.lazy_dataset = lazy_dataset
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.niter_els = niter_els
        self.frames_per_split = frames_per_split
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.min_mov = min_mov
        self.pw_rigid = bool(pw_rigid)
        self.bigtiff = bigtiff
        self.file_FOV_dims = self.lazy_dataset.shape[1], self.lazy_dataset.shape[2]
        self.file_num_frames = self.lazy_dataset.shape[0]

        # In case gSig_filt is not None, we define a kernel which we use for 1p processing:
        if gSig_filt is not None:
            self.filter_kernel = get_kernel(gSig_filt)
        else:
            self.filter_kernel = None

    def motion_correct(self, template: Optional[np.ndarray] = None,
                       save_movie: Optional[bool] = False) -> tuple[frame_corrector, str]:
        """General driver function which performs motion correction

        Args:
            template (ndarray): Template provided by user for motion correction default
            save_movie (bool): Flag for saving motion corrected file(s) as memory mapped file(s)

        Returns:
            frame_corrector_obj (jnormcorre.motion_correction.frame_corrector): Object for applying frame correction
                with final inferred template
            target_file (str): path to saved file
        """
        frame_constant = 400
        if self.min_mov is None:
            if self.filter_kernel is None:
                mi = np.inf
                for j in range(min(self.lazy_dataset.shape[0], frame_constant)):
                    try:
                        mi = min(mi, np.min(self.lazy_dataset[j, :, :]))
                    except StopIteration:
                        break
                self.min_mov = mi
            else:
                self.min_mov = np.array([high_pass_filter_cv(m_, self.filter_kernel)
                                         for m_ in self.lazy_dataset[:frame_constant, :, :]]).min()

        if self.pw_rigid:
            # Verify that the strides and overlaps are meaningfully defined
            verify_strides_and_overlaps(self.file_FOV_dims[0], self.strides[0], self.overlaps[0])
            verify_strides_and_overlaps(self.file_FOV_dims[1], self.strides[1], self.overlaps[1])
            self._motion_correct_pwrigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                    np.max(np.abs(self.y_shifts_els))))
        else:
            self._motion_correct_rigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))
        self.border_to_0 = b0.astype(int)
        self.target_file = self.fname_tot_els if self.pw_rigid else self.fname_tot_rig

        if self.pw_rigid:
            template = self.total_template_els
        else:
            template = self.total_template_rig
        frame_correction_obj = frame_corrector(template, self.max_shifts,
                                               self.strides, self.overlaps,
                                               self.max_deviation_rigid, min_mov=self.min_mov)
        return frame_correction_obj, self.target_file

    def _motion_correct_rigid(self, template: Optional[np.ndarray] = None,
                              save_movie: Optional[bool] = False) -> None:
        """
        Perform rigid motion correction

        Args:
            template (np.ndarray) Optional template (if known) for performing registration.
            save_movie (bool): flag to save final motion corrected movie
        """
        self.total_template_rig = template
        self.templates_rig: List = []
        self.fname_tot_rig: List = []
        self.shifts_rig: List = []

        _fname_tot_rig, _total_template_rig, _templates_rig, _shifts_rig = _motion_correct_batch_rigid(
            self.lazy_dataset,
            self.max_shifts,
            frames_per_split=self.frames_per_split,
            num_splits_to_process=self.num_splits_to_process_rig,
            num_iter=self.niter_rig,
            template=self.total_template_rig,
            save_movie_rigid=save_movie,
            add_to_movie=-self.min_mov,
            filter_kernel=self.filter_kernel,
            bigtiff=self.bigtiff)
        if template is None:
            self.total_template_rig = _total_template_rig

        self.templates_rig += _templates_rig
        self.fname_tot_rig += [_fname_tot_rig]
        self.shifts_rig += _shifts_rig

    def _motion_correct_pwrigid(self, template: Optional[np.ndarray] = None,
                                save_movie: Optional[bool] = False) -> None:
        """
        Perform pw-rigid motion correction

        Args:
            template (np.ndarray) Optional template (if known) for performing registration.
            save_movie (bool): flag to save final motion corrected movie
        """

        num_iter = self.niter_els
        if template is None:
            self._motion_correct_rigid(save_movie=False)
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.fname_tot_els: List = []
        self.templates_els: List = []
        self.x_shifts_els: List = []
        self.y_shifts_els: List = []

        self.coord_shifts_els: List = []

        (_fname_tot_els, new_template_els, _templates_els, _x_shifts_els, _y_shifts_els,
         _z_shifts_els, _coord_shifts_els) = _motion_correct_batch_pwrigid(
            self.lazy_dataset, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
            upsample_factor_grid=self.upsample_factor_grid, max_deviation_rigid=self.max_deviation_rigid,
            num_splits_to_process=self.num_splits_to_process_els, num_iter=num_iter, template=self.total_template_els,
            save_movie=save_movie, filter_kernel=self.filter_kernel, bigtiff=self.bigtiff)

        if np.isnan(np.sum(new_template_els)):
            raise Exception(
                'Template contains NaNs, something went wrong. Reconsider the parameters')

        if template is None:
            self.total_template_els = new_template_els

        self.fname_tot_els += [_fname_tot_els]
        self.templates_els += _templates_els
        self.x_shifts_els += _x_shifts_els
        self.y_shifts_els += _y_shifts_els
        self.coord_shifts_els += _coord_shifts_els


def _motion_correct_batch_rigid(lazy_dataset: lazy_data_loader, max_shifts: tuple[int, int],
                                frames_per_split: int = 1000, num_splits_to_process: int = None, num_iter: int = 1,
                                template: np.ndarray = None, save_movie_rigid: bool = False, add_to_movie: float = None,
                                filter_kernel: np.ndarray = None,
                                bigtiff: bool = False) -> tuple[str, np.ndarray, list, list]:
    """
    Performs 1 pass of rigid motion correction; see the following functions for parameter details:
        (1) MotionCorrection object constructor
        (2) MotionCorrection.motion_correct
        (3) MotionCorrection._motion_correct_rigid

    Returns:
        fname_tot_rig (str): Filename of saved movie (None if no movie is saved at this point)
        total_template (np.ndarray): 2D estimated template
        templates: (list): List of 2D local templates identified in this pass
        shifts: (list). List of length (T) where T is the number of frames registered.
            Each element is a np.ndarray describing the applied shifts in both FOV dimensions.
    """

    T = lazy_dataset.shape[0]
    Ts = np.arange(T).shape[0]
    step = Ts // 50
    corrected_slicer = slice(None, min(T - 1, 4000), step + 1)  # Don't need too many frames to init the template
    m = lazy_dataset[corrected_slicer, :, :]

    # Initialize template by sampling frames uniformly throughout the movie and taking the median
    if template is None:
        if filter_kernel is not None:
            m = np.array([high_pass_filter_cv(filter_kernel, m_) for m_ in m])

        template = bin_median(m)

    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        logging.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    fname_tot_rig = None
    res_rig: List = []
    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1 and save_movie_rigid:
            save_flag = True
        else:
            save_flag = False

        fname_tot_rig, res_rig = _execute_motion_correction_iteration(lazy_dataset, frames_per_split, strides=None,
                                                                      overlaps=None,
                                                                      add_to_movie=add_to_movie, template=old_templ,
                                                                      max_shifts=max_shifts, max_deviation_rigid=0,
                                                                      save_movie=save_flag,
                                                                      num_splits=num_splits_to_process,
                                                                      filter_kernel=filter_kernel,
                                                                      bigtiff=bigtiff)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_rig]), -1)
        if filter_kernel is not None:
            new_templ = high_pass_filter_cv(filter_kernel, new_templ)

    total_template = new_templ
    templates = []
    shifts: List = []
    for rr in res_rig:
        shift_info, idxs, tmpl = rr
        templates.append(tmpl)
        num_idxs = len(list(range(idxs.start, idxs.stop, 1 if idxs.step is None else idxs.step)))
        shifts += [sh[0] for sh in shift_info[:num_idxs]]

    return fname_tot_rig, total_template, templates, shifts


def _motion_correct_batch_pwrigid(lazy_dataset: lazy_data_loader, max_shifts: tuple[int, int], strides: tuple[int, int],
                                  overlaps: tuple[int, int], add_to_movie: float,
                                  upsample_factor_grid: int = 4, max_deviation_rigid: int = 3,
                                  frames_per_split: int = 1000, num_splits_to_process: Optional[int] = None,
                                  num_iter: int = 1,
                                  template: Optional[np.ndarray] = None, save_movie: bool = False,
                                  filter_kernel: Optional[np.ndarray] = None,
                                  bigtiff=False) -> tuple[str, np.ndarray, list, list, list, list, list]:
    """
    Performs 1 pass of piecewise rigid motion correction; see the following functions for parameter details:
        (1) MotionCorrection object constructor
        (2) MotionCorrection.motion_correct
        (3) MotionCorrection._motion_correct_pwrigid

    Returns:
        fname_tot_els (str). String describing the filename saved out (None if nothing is saved)
        total_template (np.ndarray). Estimated global template from this step
        templates (list): list of local templates from this pass of motion correction
        x_shifts (list). List of x-dimension shifts across patches and frames
        y_shifts (list). List of y-dimension shifts across patches and frames
        z_shifts (list). List of z-dimension shifts across patches and frames
        coord_shifts: list
    """
    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()

        if iter_ == num_iter - 1 and save_movie:
            save_flag = True
        else:
            save_flag = False

        if iter_ == num_iter - 1 and save_flag:
            num_splits_to_process = None
        fname_tot_els, res_el = _execute_motion_correction_iteration(lazy_dataset, frames_per_split, strides, overlaps,
                                                                     add_to_movie=add_to_movie, template=old_templ,
                                                                     max_shifts=max_shifts,
                                                                     max_deviation_rigid=max_deviation_rigid,
                                                                     upsample_factor_grid=upsample_factor_grid,
                                                                     save_movie=save_flag,
                                                                     num_splits=num_splits_to_process,
                                                                     filter_kernel=filter_kernel,
                                                                     bigtiff=bigtiff)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_el]), -1)
        if filter_kernel is not None:
            new_templ = high_pass_filter_cv(filter_kernel, new_templ)

    total_template = new_templ
    templates = []
    x_shifts = []
    y_shifts = []
    z_shifts = []
    coord_shifts = []
    for rr in res_el:
        shift_info_chunk, idxs_chunk, tmpl_chunk = rr
        templates.append(tmpl_chunk)

        for shift_info in shift_info_chunk:
            total_shift = shift_info
            x_shifts.append(np.array([sh[0] for sh in total_shift]))
            y_shifts.append(np.array([sh[1] for sh in total_shift]))
            coord_shifts.append(None)
    return fname_tot_els, total_template, templates, x_shifts, y_shifts, z_shifts, coord_shifts


def _execute_motion_correction_iteration(lazy_dataset: lazy_data_loader, frames_per_split: int,
                                         strides: Optional[tuple[int, int]], overlaps: Optional[tuple[int, int]],
                                         add_to_movie: float = 0.0,
                                         template: Optional[np.ndarray] = None,
                                         max_shifts: tuple[int, int] = (12, 12), max_deviation_rigid: int = 3,
                                         upsample_factor_grid: int = 4,
                                         save_movie: bool = True, num_splits: Optional[int] = None,
                                         filter_kernel: np.ndarray = None,
                                         bigtiff: bool = False) -> tuple[str, list[tuple]]:
    """
    Executes a single iteration of motion correction. See the following functions for details:
    (1) MotionCorrection constructor
    (2) MotionCorrection.motion_correct

    Returns:
        fname_tot (str): Filename of the saved data (if it exists)
        res (list of tuples): For every split (chunk of data) we generate 1 tuple containing
            (1) list of shifts for each frame (2) array of frame indices which were registered (3) the local template.
             res holds all of these individual tuples.
    """
    if template is None:
        raise Exception('Template must be well-defined for the registration step')

    dims = lazy_dataset.shape[1], lazy_dataset.shape[2]
    T = lazy_dataset.shape[0]

    idxs = calculate_splits(T, frames_per_split)

    if num_splits is not None and not save_movie:
        num_splits = min(num_splits, len(idxs))
        idxs = random.sample(idxs, num_splits)

    if save_movie:
        current_datetime = datetime.datetime.now()
        timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        fname_tot = f"data_{timestamp_str}.tiff"
    else:
        fname_tot = None

    pars = []
    for idx in idxs:
        logging.debug('Processing: frames: {}'.format(idx))
        pars.append(
            [lazy_dataset, fname_tot, idx, template, strides, overlaps, max_shifts, np.array(
                add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
             filter_kernel])

    split_constant = load_split_heuristic(dims[0], dims[1], T)
    res = _tile_and_correct_dataloader(pars, lazy_dataset, split_constant=split_constant, bigtiff=bigtiff)
    return fname_tot, res


def _tile_and_correct_dataloader(param_list, lazy_dataset, split_constant=200, bigtiff=False) -> list[tuple]:
    """
    See _execute_motion_correction_iteration for details on what parameters this function uses to perform registration.
    If specified, writes corrected frames to a tiff memmap file (name given by out_fname)
    """
    num_workers = 0
    movie_shape = lazy_dataset.shape
    tile_and_correct_dataobj = tile_and_correct_dataset(param_list)
    loader_obj = torch.utils.data.DataLoader(tile_and_correct_dataobj, batch_size=1,
                                             shuffle=False, num_workers=num_workers, collate_fn=regular_collate,
                                             timeout=0)

    results_list = []
    start_pt_save = 0
    memmap_placeholder = None
    for dataloader_index, data in enumerate(tqdm(loader_obj), 0):
        num_iters = math.ceil(data[0].shape[0] / split_constant)
        imgs_net, mc, out_fname, idxs, template, strides, overlaps, max_shifts, \
            add_to_movie, max_deviation_rigid, upsample_factor_grid, \
            filter_kernel = data
        if out_fname is not None:
            if memmap_placeholder is None:
                memmap_placeholder = tifffile.memmap(out_fname, shape=movie_shape, dtype=mc.dtype,
                                                     bigtiff=bigtiff)
        for j in range(num_iters):

            start_pt = split_constant * j
            end_pt = min(data[0].shape[0], start_pt + split_constant)
            imgs = imgs_net[start_pt:end_pt, :, :]
            shift_info = []
            upsample_factor_fft = 10  # Hardcoded from original method

            if max_deviation_rigid == 0:
                if filter_kernel is None:
                    outs = register_frames_to_template_rigid(imgs, template, max_shifts, add_to_movie)
                else:
                    imgs_filtered = high_pass_batch(filter_kernel, imgs)
                    outs = register_frames_to_template_1p_rigid(imgs, imgs_filtered, template, max_shifts, add_to_movie)
                mc[start_pt:end_pt, :, :] = outs[0]
                shift_info.extend([[k] for k in np.array(outs[1])])
            else:
                if filter_kernel is None:
                    outs = register_frames_to_template_pwrigid(imgs, template, strides[0], strides[1], overlaps[0],
                                                               overlaps[1], \
                                                               max_shifts, upsample_factor_fft, max_deviation_rigid,
                                                               add_to_movie)
                else:
                    imgs_filtered = high_pass_batch(filter_kernel, imgs)
                    outs = register_frames_to_template_1p_pwrigid(imgs, imgs_filtered, template, strides[0], strides[1],
                                                                  overlaps[0], overlaps[1], \
                                                                  max_shifts, upsample_factor_fft, max_deviation_rigid,
                                                                  add_to_movie)

                mc[start_pt:end_pt, :, :] = outs[0]
                shift_info.extend([[k] for k in np.array(outs[1])])

        if out_fname is not None:
            memmap_placeholder[idxs, :, :] = mc
            start_pt_save += mc.shape[0]
            memmap_placeholder.flush()
        new_temp = generate_template_chunk(mc)

        results_list.append((shift_info, idxs, new_temp))

    return results_list


class tile_and_correct_dataset():
    """
    Basic dataloading class for loading chunks of data. Written like this so that code can support prefetching from disk
    """

    def __init__(self, param_list):
        self.param_list = param_list

    def __len__(self):
        return len(self.param_list)

    def __getitem__(self, index):
        lazy_dataset, out_fname, idxs, template, strides, overlaps, max_shifts, \
            add_to_movie, max_deviation_rigid, upsample_factor_grid, \
            filter_kernel = self.param_list[index]

        imgs = lazy_dataset[idxs, :, :]
        mc = np.zeros(imgs.shape, dtype=np.float32)

        return imgs, mc, out_fname, idxs, template, strides, overlaps, max_shifts, \
            add_to_movie, max_deviation_rigid, upsample_factor_grid, \
            filter_kernel


def generate_template_chunk(arr: np.ndarray, batch_size: int = 250000) -> np.ndarray:
    dim_1_step = int(math.sqrt(batch_size))
    dim_2_step = int(math.sqrt(batch_size))

    dim1_net_iters = math.ceil(arr.shape[1] / dim_1_step)
    dim2_net_iters = math.ceil(arr.shape[2] / dim_2_step)

    total_output = np.zeros((arr.shape[1], arr.shape[2]))
    for k in range(dim1_net_iters):
        for j in range(dim2_net_iters):
            start_dim1 = k * dim_1_step
            end_dim1 = min(start_dim1 + dim_1_step, arr.shape[1])
            start_dim2 = k * dim_2_step
            end_dim2 = min(start_dim2 + dim_2_step, arr.shape[2])
            total_output[start_dim1:end_dim1, start_dim2:end_dim2] = nan_processing(
                arr[:, start_dim1:end_dim1, start_dim2:end_dim2])

    return total_output


@partial(jit)
def nan_processing(arr: ArrayLike) -> ArrayLike:
    p = jnp.nanmean(arr, 0)
    q = jnp.nanmin(p)
    r = jnp.nan_to_num(p, q)
    return r


def regular_collate(batch):
    return batch[0]


def calculate_splits(T: int, frames_per_split: int) -> list:
    """
    Function used to build a computation work plan for motion correction (decide which frames to run per split, etc.)
    """
    if frames_per_split <= 1:
        raise ValueError("frames_per_split must be an integer greater than 1")

    start_point = list(range(0, T, frames_per_split))
    if T - frames_per_split < start_point[-1] and len(start_point) > 1:
        start_point[-1] = T - frames_per_split

    slice_list = []
    for k in range(len(start_point)):
        start = start_point[k]
        end = min(T, start + frames_per_split)
        slice_list.append(slice(start, end, 1))

    return slice_list


def load_split_heuristic(d1, d2, T):
    '''
    Heuristic for determining how many frames to register at a time (to avoid GPU OOM)
    '''

    if d1 * d2 > 512 * 512:
        new_T = 20
    elif d1 * d2 > 100000:
        new_T = 100
    else:
        new_T = 2000

    return min(T, new_T)


def bin_median(mat: np.ndarray, window: int = 10, exclude_nans: bool = True):
    """
    Compute median of 3D array in along axis 0 by binning values

    Args:
        mat (np.ndarray). Input 3D matrix, time along first dimension
        window (int). Number of frames in a bin

    Returns:
        img (np.ndarray). Median image
    """

    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img


def _upsampled_dft_full(data, upsampled_region_size, upsample_factor, axis_offsets):
    return np.array(_upsampled_dft_jax(data, upsampled_region_size,
                                       upsample_factor, axis_offsets))


# @partial(jit, static_argnums=(1,))
def _upsampled_dft_jax(data: ArrayLike, upsampled_region_size: int,
                       upsample_factor: int, axis_offsets: ArrayLike) -> ArrayLike:
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data (jnp.array). The input data array (DFT of original data) to upsample.
        upsampled_region_size (int). The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.
        upsample_factor (int). The upsampling factor for the DFT.
        axis_offsets (jnp.array).
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)
    Returns:
        output (jnp.array)
            The upsampled DFT of the specified region.
    """

    # Calculate col_kernel
    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)

    term_A = shifted - jnp.floor(data.shape[1] / 2)

    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis=0) - axis_offsets[1]

    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )

    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))

    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - axis_offsets[0]
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0] / 2)

    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )

    output = jnp.tensordot(row_kernel, data, axes=[1, 0])
    output = jnp.tensordot(output, col_kernel, axes=[1, 0])

    return output


@partial(jit)
def _upsampled_dft_jax_no_size(data: ArrayLike, upsample_factor: int) -> ArrayLike:
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data (np.ndarray). The input data array (DFT of original data) to upsample.
        upsample_factor (int). Upsampling factor

    Returns:
        output (ArrayLike)
    """

    upsampled_region_size = 1

    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)

    term_A = shifted - jnp.floor(data.shape[1] / 2)

    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis=0) - 0

    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )

    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))

    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - 0
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0] / 2)

    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )

    output = jnp.tensordot(row_kernel, data, axes=[1, 0])
    output = jnp.tensordot(output, col_kernel, axes=[1, 0])

    return output


# @partial(jit)
def _compute_phasediff(cross_correlation_max: ArrayLike) -> ArrayLike:
    '''
    Compute global phase difference between the two images (should be zero if images are non-negative).
    Args:
        cross_correlation_max (complex)
    Returns:
        The complex value of the cross correlation at its maximum point.
    
    '''
    return jnp.angle(cross_correlation_max)


# @partial(jit)
def get_freq_comps_jax(src_image: ArrayLike, target_image: ArrayLike) -> tuple[ArrayLike]:
    """
    Routine to compute frequency components of two images
    """
    src_image_cpx = jnp.complex64(src_image)
    target_image_cpx = jnp.complex64(target_image)
    src_freq = jnp.fft.fftn(src_image_cpx)
    src_freq = jnp.divide(src_freq, jnp.size(src_freq))
    target_freq = jnp.fft.fftn(target_image_cpx)
    target_freq = jnp.divide(target_freq, jnp.size(target_freq))
    return src_freq, target_freq


# @partial(jit)
def threshold_dim1(img: ArrayLike, ind: int) -> ArrayLike:
    a = img.shape[0]

    row_ind_first = jnp.arange(a) < ind
    row_ind_second = jnp.arange(a) > a - ind - 1

    prod = row_ind_first + row_ind_second

    broadcasted = jnp.broadcast_to(jnp.expand_dims(prod, axis=1), img.shape)
    return broadcasted * img


# @partial(jit)
def threshold_dim2(img: ArrayLike, ind: int) ->ArrayLike:
    b = img.shape[1]

    col_ind_first = jnp.arange(b) < ind
    col_ind_second = jnp.arange(b) > b - ind - 1

    prod = col_ind_first + col_ind_second

    broadcasted = jnp.broadcast_to(jnp.expand_dims(prod, axis=0), img.shape)
    return img * broadcasted


# @partial(jit)
def subtract_values(a, b):
    return a - b


# @partial(jit)
def return_identity(a, b):
    return a


# @partial(jit, static_argnums=(2,))
def register_translation_jax_simple(src_image: ArrayLike, target_image: ArrayLike,
                                    upsample_factor: int,
                                    max_shifts: tuple[int, int] = (10, 10)) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Finds optimal rigid shifts to register target_image (template) with src_image (input image). Negate
        these shifts to get the optimal rigid transformation from src_image to template.

    Args:
        src_image (np.ndarray). Input image
        target_image (np.ndarray). Template. Must be same dimensionality as src_image
        upsample_factor (int). Images will be registered to within 1 / upsample_factor of a pixel.
        max_shifts (tuple). Tuple of two integers describing maximum rigid shift in each dimension

    Returns:
        shifts (ndarray). Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
        sfr_freq (jnp.array). Frequency domain representation of src_image.
        phasediff (jnp.array). Global phase difference between the two images (should be
            zero if images are non-negative).
    """

    ##Now, must FFT the data:
    src_freq, target_freq = get_freq_comps_jax(src_image, target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = jnp.multiply(src_freq, jnp.conj(target_freq))

    cross_correlation = jnp.fft.ifftn(image_product)

    # Locate maximum
    new_cross_corr = jnp.abs(cross_correlation)
    new_cross_corr = threshold_dim1(new_cross_corr, max_shifts[0])
    new_cross_corr = threshold_dim2(new_cross_corr, max_shifts[1])

    maxima = jnp.unravel_index(jnp.argmax(new_cross_corr),
                               cross_correlation.shape)

    midpoints = jnp.array([jnp.fix(shape[0] / 2), jnp.fix(shape[1] / 2)])

    shifts = jnp.array(maxima, dtype=jnp.float32)

    first_shift = jax.lax.cond(shifts[0] > midpoints[0], subtract_values, return_identity, *(shifts[0], shape[0]))
    second_shift = jax.lax.cond(shifts[1] > midpoints[1], subtract_values, return_identity, *(shifts[1], shape[1]))
    shifts = jnp.array([first_shift, second_shift])

    shifts = jnp.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(upsample_factor * 1.5 + 0.5)
    # Center of output array at dftshift + 1
    dftshift = jnp.fix(upsampled_region_size / 2.0)
    upsample_factor = jnp.array(upsample_factor, dtype=jnp.float32)
    normalization = (src_freq.size * upsample_factor ** 2)
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shifts * upsample_factor

    cross_correlation = _upsampled_dft_jax(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
    cross_correlation /= normalization
    # Locate maximum and map back to original pixel grid
    maxima = jnp.array(jnp.unravel_index(
        jnp.argmax(jnp.abs(cross_correlation)),
        cross_correlation.shape),
        dtype=jnp.float32)
    maxima -= dftshift
    shifts = shifts + maxima / upsample_factor
    CCmax = cross_correlation.max()

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    shape_new = jax.nn.relu(jnp.array(shape) - 1) > 0
    shifts = shifts * shape_new
    return shifts, src_freq, _compute_phasediff(CCmax)

# @partial(jit, static_argnums=(1,))
def _upsampled_dft_jax_full(data: ArrayLike, upsampled_region_size: int,
                            upsample_factor: int, axis_offsets: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data (np.ndarray):
            The input data array (DFT of original data) to upsample.
        upsampled_region_size (integer). The size of the region to be sampled
        upsample_factor (int). The upsampling factor for registration.
        axis_offsets (tuple). Offsets from the image to be sampled.
    Returns:
        output (jnp.ndarray). The upsampled DFT of the specified region.
    """

    # Calculate col_kernel
    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)

    term_A = shifted - jnp.floor(data.shape[1] / 2)

    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis=0) - axis_offsets[1]

    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )

    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))

    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - axis_offsets[0]
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0] / 2)

    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )

    output = jnp.tensordot(row_kernel, data, axes=[1, 0])
    output = jnp.tensordot(output, col_kernel, axes=[1, 0])

    return output


# @partial(jit)
def threshold_shifts_0_if(new_cross_corr, shift_ub, shift_lb):
    ## In this case, shift_lb is negative and shift_ub is nonnegative
    a, b = new_cross_corr.shape
    first_thres = np.arange(a) < shift_ub
    second_thres = np.arange(a) >= a + shift_lb
    prod = first_thres + second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis=1), \
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit)
def threshold_shifts_0_else(new_cross_corr, shift_ub, shift_lb):
    # In this case shift_lb is nonnegative OR shift_ub is negative, we can go case by case
    a, b = new_cross_corr.shape
    lb_threshold = jax.lax.cond(shift_lb >= 0, lambda p, q: q, \
                                lambda p, q: p + q, *(a, shift_lb))
    first_thres = np.arange(a) >= lb_threshold
    ub_threshold = jax.lax.cond(shift_ub >= 0, lambda p, q: q, \
                                lambda p, q: p + q, *(a, shift_ub))

    second_thres = np.arange(a) < ub_threshold
    prod = first_thres * second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis=1), \
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit)
def threshold_shifts_1_if(new_cross_corr, shift_ub, shift_lb):
    ## In this case, shift_lb is negative and shift_ub is nonnegative
    a, b = new_cross_corr.shape
    first_thres = np.arange(b) < shift_ub
    second_thres = np.arange(b) >= b + shift_lb
    prod = first_thres + second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis=0), \
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit)
def threshold_shifts_1_else(new_cross_corr, shift_ub, shift_lb):
    # In this case shift_lb is nonnegative OR shift_ub is negative, we can go case by case
    a, b = new_cross_corr.shape
    lb_threshold = jax.lax.cond(shift_lb >= 0, lambda p, q: q, \
                                lambda p, q: p + q, *(b, shift_lb))
    first_thres = np.arange(b) >= lb_threshold
    ub_threshold = jax.lax.cond(shift_ub >= 0, lambda p, q: q, \
                                lambda p, q: p + q, *(a, shift_ub))

    second_thres = np.arange(b) < ub_threshold
    prod = first_thres * second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis=0), \
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit, static_argnums=(2,))
def register_translation_jax_full(src_image: ArrayLike, target_image: ArrayLike, upsample_factor: int,
                                  shifts_lb: ArrayLike,
                                  shifts_ub: ArrayLike, max_shifts=(10, 10)) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Estimates piecewise rigid shifts which would align target_image TO the src_image. Negate these to get shifts going
    from src_image to target.
    Args:
        src_image (np.ndarray). Input data/images.
        target_image (np.ndarray). Template. Must have same shape as src_image.
        upsample_factor (int). Upsampling which occurs to estimate the shifts
        shifts_lb (ArrayLike). Lower bound on the shifts which can be applied at each subpatch.
        shifts_ub (ArrayLike). Upper bound on the shifts which can be applied at each subpatch.

    Returns:
        shifts (np.ndarray). Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.
        src_freq (jnp.array). Frequency domain representation of input image data.
        phasediff (jnp.array). Float value, global phase difference between the two images (should be
            zero if images are non-negative).
    """

    ##Now, must FFT the data:
    src_freq, target_freq = get_freq_comps_jax(src_image, target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = jnp.multiply(src_freq, jnp.conj(target_freq))

    cross_correlation = jnp.fft.ifftn(image_product)

    # Locate maximum
    new_cross_corr = jnp.abs(cross_correlation)

    first_truth_value = (shifts_lb[0] < 0) * (shifts_ub[0] >= 0)
    new_cross_corr = jax.lax.cond(first_truth_value, \
                                  threshold_shifts_0_if, threshold_shifts_0_else, \
                                  *(new_cross_corr, shifts_ub[0], shifts_lb[0]))

    second_truth_value = (shifts_lb[1]) < 0 * (shifts_ub[1] >= 0)
    new_cross_corr = jax.lax.cond(second_truth_value, \
                                  threshold_shifts_1_if, threshold_shifts_1_else, \
                                  *(new_cross_corr, shifts_ub[1], shifts_lb[1]))

    maxima = jnp.unravel_index(jnp.argmax(new_cross_corr),
                               cross_correlation.shape)

    midpoints = jnp.array([jnp.fix(shape[0] / 2), jnp.fix(shape[1] / 2)])

    shifts = jnp.array(maxima, dtype=jnp.float32)

    first_shift = jax.lax.cond(shifts[0] > midpoints[0], subtract_values, return_identity, *(shifts[0], shape[0]))
    second_shift = jax.lax.cond(shifts[1] > midpoints[1], subtract_values, return_identity, *(shifts[1], shape[1]))
    shifts = jnp.array([first_shift, second_shift])

    shifts = jnp.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(upsample_factor * 1.5 + 0.5)
    # Center of output array at dftshift + 1
    dftshift = jnp.fix(upsampled_region_size / 2.0)
    upsample_factor = jnp.array(upsample_factor, dtype=jnp.float32)
    normalization = (src_freq.size * upsample_factor ** 2)
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shifts * upsample_factor

    cross_correlation = _upsampled_dft_jax_full(image_product.conj(),
                                                upsampled_region_size,
                                                upsample_factor,
                                                sample_region_offset).conj()
    cross_correlation /= normalization
    # Locate maximum and map back to original pixel grid
    maxima = jnp.array(jnp.unravel_index(
        jnp.argmax(jnp.abs(cross_correlation)),
        cross_correlation.shape),
        dtype=jnp.float32)
    maxima -= dftshift
    shifts = shifts + maxima / upsample_factor
    CCmax = cross_correlation.max()

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    shape_new = jax.nn.relu(jnp.array(shape) - 1) > 0
    shifts = shifts * shape_new

    return shifts, src_freq, _compute_phasediff(CCmax)


vmap_register_translation = vmap(register_translation_jax_full, in_axes=(0, 0, None, None, None, None))

@partial(jit)
def update_src_freq_jax(src_freq):
    out = jnp.fft.fftn(src_freq)
    out_norm = jnp.divide(out, jnp.size(out))
    return jnp.complex128(out_norm)

@partial(jit)
def update_src_freq_identity(src_freq):
    return jnp.complex128(src_freq)


def update_src_freq_flag(src_freq, flag):
    output = jnp.complex128(jax.lax.cond(~flag, update_src_freq_jax, update_src_freq_identity, src_freq))
    return output


def first_value(a, b):
    return a


def second_value(a, b):
    return b


# @partial(jit)
def ceil_max(a, b):
    interm = jax.lax.cond(a < b, second_value, first_value, a, b)
    return jnp.ceil(interm)


# @partial(jit)
def floor_min(a, b):
    interm = jax.lax.cond(a > b, second_value, first_value, a, b)
    return jnp.fix(interm)


# @partial(jit)
def apply_shifts_dft_fast_1(src_freq_in: ArrayLike, shift_a: ArrayLike,
                            shift_b: ArrayLike, diffphase: ArrayLike) -> ArrayLike:
    """
    use the inverse dft to apply shifts
    Args:
        src_freq_in (jnp.array). Frequency domain representatio of an image
        shift_a (jnp.array). One element, describing shift in dimension 1
        shift_b (jnp.array). One element, describing shift in dimension 2
        diffphase (jnp.array). Global phase difference; see register translation functions

    Returns:
        Shifted image
    """

    src_freq = jnp.complex64(src_freq_in)

    nc, nr = src_freq.shape
    val_1 = -int(nr / 2)
    val_2 = int(nr / 2. + 0.5)
    val_3 = -int(nc / 2)
    val_4 = int(nc / 2. + 0.5)
    Nr = jnp.fft.ifftshift(jnp.arange(val_1, val_2))
    Nc = jnp.fft.ifftshift(jnp.arange(val_3, val_4))
    Nr, Nc = jnp.meshgrid(Nr, Nc)
    Greg = jnp.multiply(src_freq, jnp.exp(1j * 2 * jnp.pi *
                                          (-shift_b * 1. * Nr / nr - shift_a * 1. * Nc / nc)))

    Greg = jnp.dot(Greg, jnp.exp(jnp.multiply(1j, diffphase)))
    new_img = jnp.real(jnp.fft.ifftn(Greg, norm="forward"))

    max_h = ceil_max(shift_a, 0.).astype(jnp.int32)
    max_w = ceil_max(shift_b, 0.).astype(jnp.int32)
    min_h = floor_min(shift_a, 0.).astype(jnp.int32)
    min_w = floor_min(shift_b, 0.).astype(jnp.int32)

    new_img_1 = fill_maxh(new_img, max_h)
    new_img_2 = jax.lax.cond(min_h < 0, fill_minh, return_identity_mins, *(new_img_1, min_h))
    new_img_3 = jax.lax.cond(max_w > 0, fill_maxw, return_identity_mins, *(new_img_2, max_w))
    new_img_4 = jax.lax.cond(min_w < 0, fill_minw, return_identity_mins, *(new_img_3, min_w))

    return new_img_4


def fill_minw(img, k):
    x, y = img.shape
    key = y + k

    filter_mat = (jnp.arange(y) < key).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x, y))

    img_filter = filter_mat * img

    addend = (jnp.arange(y) >= key).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x, y))
    addend = addend * img[:, k - 1, None]

    return img_filter + addend


def fill_maxw(img, k):
    x, y = img.shape
    filter_mat = (jnp.arange(y) >= k).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x, y))

    img_filtered = filter_mat * img

    addend = (jnp.arange(y) < k).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x, y))
    addend = addend * img[:, k, None]

    return img_filtered + addend


def fill_maxh(img, k):
    x, y = img.shape
    filter_mat = jnp.reshape((jnp.arange(x) >= k), (-1, 1)).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x, y))
    img_filtered = img * filter_mat

    addend_binary = jnp.reshape((jnp.arange(x) < k), (-1, 1))
    addend_binary = jnp.broadcast_to(addend_binary, (x, y))
    addend_binary = addend_binary * img[k]
    return addend_binary + img_filtered


def fill_minh(img, k):
    x, y = img.shape
    key = x + k
    filtered_mat = jnp.reshape((jnp.arange(x) < key), (-1, 1)).astype(jnp.int32)
    filtered_mat = jnp.broadcast_to(filtered_mat, (x, y))

    filtered_img = img * filtered_mat

    addend = jnp.reshape((jnp.arange(x) >= key), (-1, 1)).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x, y))
    addend_final = addend * img[key - 1]

    return filtered_img + addend_final


def return_identity_mins(in_var, k):
    return in_var


# @partial(jit, static_argnums=(4,))
def _register_to_template_1p_rigid(img: ArrayLike, img_filtered: ArrayLike, template: ArrayLike,
                                   max_shifts: tuple[int, int], add_to_movie: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Same as _register_to_template_rigid; only difference is that we align img_filtered (the
    high-pass thresholded movie) to template, but we apply the compute shifts and apply those shifts to img.
    """
    upsample_factor_fft = 10
    img = jnp.add(img, add_to_movie).astype(jnp.float32)
    template = jnp.add(template, add_to_movie).astype(jnp.float32)

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img_filtered, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    # Second input doesn't matter here
    sfr_freq, _ = get_freq_comps_jax(img, img)

    new_img = apply_shifts_dft_fast_1(sfr_freq, -rigid_shts[0], -rigid_shts[1], diffphase)

    return new_img - add_to_movie, jnp.array([-rigid_shts[0], -rigid_shts[1]])


register_frames_to_template_1p_rigid = jit(
    vmap(_register_to_template_1p_rigid, in_axes=(0, 0, None, (None, None), None)))

register_frames_to_template_1p_rigid_docs = \
    """
    Performs rigid registration of a series of frames to a single template for 1p data. 
    
    Args:
        img (np.array): Shape (T, x, y), frames we want to register.  T is number of frames, x and y are spatial dims
        img_filtered (np.array). Shape (T, x, y), a high-pass filtered version of imgs. This is used to compute shifts relative to the template.
        template (np.array): Shape (x, y). Template image
        max_shifts (np.array): Has 2 integers specifying max shift in both FOV dimensions
        add_to_movie (np.array): Scalar value in jnp.array for adding to each frame.

    Returns:
        aligned (jnp.array): Shape (T, x, y). Aligned version of "img" to template.
        shifts (jnp.array): Shifts which were applied to img.
    """
register_frames_to_template_1p_rigid.__doc__ = register_frames_to_template_1p_rigid_docs


# @partial(jit, static_argnums=(3,))
def _register_to_template_rigid(img: ArrayLike, template: ArrayLike,
                                max_shifts: ArrayLike, add_to_movie: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Registers img to template, subject to constraint that max shift in either FOV dimension is bounded by values in
    max_shifts.

    Args:
        img (jnp.array): Input image of interest.
        template (jnp.array): Template image
        max_shifts (jnp.array): Has 2 integers specifying max shift in both FOV dimensions
        add_to_movie (jnp.array): Scalar value in jnp.array for adding to each frame.
    Returns:
        aligned (jnp.array): Aligned version of "img" to template.
        shifts (jnp.array): Shifts which were applied to img.
    """
    upsample_factor_fft = 10

    img = jnp.add(img, add_to_movie).astype(jnp.float32)
    template = jnp.add(template, add_to_movie).astype(jnp.float32)

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    new_img = apply_shifts_dft_fast_1(sfr_freq, -rigid_shts[0], -rigid_shts[1], diffphase)

    return new_img - add_to_movie, jnp.array([-rigid_shts[0], -rigid_shts[1]])


register_frames_to_template_rigid = jit(vmap(_register_to_template_rigid, in_axes=(0, None, None, None)))

register_frames_to_template_rigid_docs = \
    """
    Performs rigid registration of a series of frames to a single template. 
    
    Args:
        img (jnp.array): Shape (T, x, y), frames we want to register.  T is number of frames, x and y are spatial dims
        template (jnp.array): Shape (x, y). Template image
        max_shifts (jnp.array): Has 2 integers specifying max shift in both FOV dimensions
        add_to_movie (jnp.array): Scalar value in jnp.array for adding to each frame.
        
    Returns:
        aligned (jnp.array): Aligned version of "img" to template.
        shifts (jnp.array): Shifts which were applied to img.
    """
register_frames_to_template_rigid.__doc__ = register_frames_to_template_rigid_docs

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim = jnp.arange(0, img.shape[0] - overlaps_0 - strides_0, strides_0)
    first_dim = jnp.append(first_dim, img.shape[0] - overlaps_0 - strides_0)

    second_dim = jnp.arange(0, img.shape[1] - overlaps_1 - strides_1, strides_1)
    second_dim = jnp.append(second_dim, img.shape[1] - overlaps_1 - strides_1)
    return first_dim, second_dim


@partial(jit, static_argnums=(3, 4))
def crop_image(img, x, y, length_1, length_2):
    out = jax.lax.dynamic_slice(img, (x, y), (length_1, length_2))
    return out


crop_image_vmap = vmap(crop_image, in_axes=(None, 0, 0, None, None))


# @partial(jit, static_argnums=(1,2,3,4))
def get_patches_jax(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim, second_dim = get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1)
    product = jnp.array(jnp.meshgrid(first_dim, second_dim)).T.reshape((-1, 2))
    first_dim_new = product[:, 0]
    second_dim_new = product[:, 1]
    return crop_image_vmap(img, first_dim_new, second_dim_new, overlaps_0 + strides_0, overlaps_1 + strides_1)


# @partial(jit, static_argnums=(1,2,3,4))
def get_xy_grid(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim, second_dim = get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1)
    first_dim_updated = np.arange(jnp.size(first_dim))
    second_dim_updated = np.arange(jnp.size(second_dim))
    product = jnp.array(jnp.meshgrid(first_dim_updated, second_dim_updated)).T.reshape((-1, 2))
    return product


# @partial(jit, static_argnums=(3,4,5,6,8))
def _register_to_template_1p_pwrigid(img: ArrayLike, img_filtered: ArrayLike,
                                     template: ArrayLike, strides_0: int, strides_1: int, overlaps_0: int,
                                     overlaps_1: int,
                                     max_shifts: ArrayLike,
                                     upsample_factor_fft: int,
                                     max_deviation_rigid: int, add_to_movie: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    This is the same as _register_to_template_pwrigid; the only difference is that there is an extra
    parameter, img_filtered, which is a high-pass thresholded version of img. We align that to template, and
    apply the shifts to image to do the alignment. See _register_to_template_pwrigid for parameter info.
    """
    strides = [strides_0, strides_1]
    overlaps = [overlaps_0, overlaps_1]

    img = jnp.array(img).astype(jnp.float32)
    template = jnp.array(template).astype(jnp.float32)

    img_filtered = img_filtered + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img_filtered, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    # extract patches

    templates = get_patches_jax(template, overlaps[0], overlaps[1], strides[0], strides[1])
    xy_grid = get_xy_grid(template, overlaps[0], overlaps[1], strides[0], strides[1])
    imgs = get_patches_jax(img_filtered, overlaps[0], overlaps[1], strides[0], strides[1])
    sum_0 = img_filtered.shape[0] - strides_0 - overlaps_0
    sum_1 = img_filtered.shape[1] - strides_1 - overlaps_1
    comp_a = sum_0 // strides_0 + 1 + (sum_0 % strides_0 > 0)
    comp_b = sum_1 // strides_1 + 1 + (sum_1 % strides_1 > 0)
    dim_grid = [comp_a, comp_b]
    num_tiles = comp_a * comp_b

    lb_shifts = jnp.ceil(jnp.subtract(
        rigid_shts, max_deviation_rigid)).astype(jnp.int16)
    ub_shifts = jnp.floor(
        jnp.add(rigid_shts, max_deviation_rigid)).astype(jnp.int16)

    # extract shifts for each patch
    src_image_inputs = jnp.array(imgs)
    target_image_inputs = jnp.array(templates)
    shfts_et_all = vmap_register_translation(src_image_inputs, target_image_inputs, upsample_factor_fft, lb_shifts,
                                             ub_shifts, max_shifts)

    shift_img_y = jnp.reshape(jnp.array(shfts_et_all[0])[:, 1], dim_grid)
    shift_img_x = jnp.reshape(jnp.array(shfts_et_all[0])[:, 0], dim_grid)
    diffs_phase_grid = jnp.reshape(jnp.array(shfts_et_all[2]), dim_grid)

    dims = img.shape

    x_grid, y_grid = jnp.meshgrid(jnp.arange(0., img.shape[1]).astype(
        jnp.float32), jnp.arange(0., img.shape[0]).astype(jnp.float32))

    remap_input_2 = jax.image.resize(shift_img_y.astype(jnp.float32), dims, method="cubic") + x_grid
    remap_input_1 = jax.image.resize(shift_img_x.astype(jnp.float32), dims, method="cubic") + y_grid
    m_reg = jax.scipy.ndimage.map_coordinates(img, [remap_input_1, remap_input_2], order=1, mode='nearest')

    shift_img_x_r = shift_img_x.reshape(num_tiles)
    shift_img_x_y = shift_img_y.reshape(num_tiles)
    total_shifts = jnp.stack([shift_img_x_r, shift_img_x_y], axis=1) * -1
    return m_reg - add_to_movie, total_shifts


register_frames_to_template_1p_pwrigid = jit(
    vmap(_register_to_template_1p_pwrigid, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)), \
    static_argnums=(3, 4, 5, 6, 8))

register_frames_to_template_1p_pwrigid_docs = \
    """
    Perform piecewise rigid motion correction on 1p data by
    (1) dividing the FOV in patches
    (2) motion correcting each patch separately
    (3) upsampling the motion correction vector field
    (4) stiching back together the corrected subpatches 
    
    Args:
        img (np.ndarray): Shape (T, x, y) Frames to register to template. T is number of frames, x and y spatial dims
        imgs_filtered (np.ndarray). Shape (T, x, y). Spatially high-pass filtered version of img
        template (np.ndarray): Shape (x, y). The reference image
        strides_0 (int): The strides of the patches in which the FOV is subdivided along dimension 0.
        strides_1 (int): The strides of the patches in which the FOV is subdivided along dimension 1.
        overlaps_0 (int): Amount of pixel overlap between patches along dimension 0
        overlaps_1 (int): Amount of pixel overlap between patches along dimension 1
        max_shifts (tuple): Max shifts in x and y
        upsample_factor_fft (int): The resolution of fractional shifts
        max_deviation_rigid (int): Maximum deviation in shifts of each patch from the rigid shift (should not be large)
        add_to_movie (np.array): Constant offset to add to movie before registration to avoid negative values.
    
    Returns:
        new_img (jnp.array): Shape (T, x, y), motion corrected version of img. 
        total_shifts (jnp.array): Shifts applied to each patch.
    
    """

register_frames_to_template_1p_pwrigid.__doc__ = register_frames_to_template_1p_pwrigid_docs




# @partial(jit, static_argnums=(2,3,4,5,7))
def _register_to_template_pwrigid(img: ArrayLike, template: ArrayLike, strides_0: int, strides_1: int,
                                  overlaps_0: int, overlaps_1: int, max_shifts: ArrayLike,
                                  upsample_factor_fft: int,
                                  max_deviation_rigid: int, add_to_movie: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Perform piecewise rigid motion correction iteration by
    (1) dividing the FOV in patches
    (2) motion correcting each patch separately
    (3) upsampling the motion correction vector field
    (4) stiching back together the corrected subpatches

    Args:
        img (np.ndarray): image to correct
        template (np.ndarray): The reference image
        strides_0 (int): The strides of the patches in which the FOV is subdivided along dimension 0.
        strides_1 (int): The strides of the patches in which the FOV is subdivided along dimension 1.
        overlaps_0 (int): Amount of pixel overlap between patches along dimension 0
        overlaps_1 (int): Amount of pixel overlap between patches along dimension 1
        max_shifts (tuple): Max shifts in x and y
        upsample_factor_fft (int): The resolution of fractional shifts
        max_deviation_rigid (int): Maximum deviation in shifts of each patch from the rigid shift (should not be large)
        add_to_movie (jnp.array): Constant offset to add to movie before registration to avoid negative values.

    Returns:
        new_img (jnp,array): Registered movie
        total_shifts (jnp.array): Shifts applied to each patch.

    """
    strides = [strides_0, strides_1]
    overlaps = [overlaps_0, overlaps_1]

    img = jnp.array(img).astype(jnp.float32)
    template = jnp.array(template).astype(jnp.float32)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    # extract patches

    templates = get_patches_jax(template, overlaps[0], overlaps[1], strides[0], strides[1])
    xy_grid = get_xy_grid(template, overlaps[0], overlaps[1], strides[0], strides[1])
    imgs = get_patches_jax(img, overlaps[0], overlaps[1], strides[0], strides[1])
    sum_0 = img.shape[0] - strides_0 - overlaps_0
    sum_1 = img.shape[1] - strides_1 - overlaps_1
    comp_a = sum_0 // strides_0 + 1 + (sum_0 % strides_0 > 0)
    comp_b = sum_1 // strides_1 + 1 + (sum_1 % strides_1 > 0)
    dim_grid = [comp_a, comp_b]
    num_tiles = comp_a * comp_b

    lb_shifts = jnp.ceil(jnp.subtract(
        rigid_shts, max_deviation_rigid)).astype(jnp.int16)
    ub_shifts = jnp.floor(
        jnp.add(rigid_shts, max_deviation_rigid)).astype(jnp.int16)

    # extract shifts for each patch
    src_image_inputs = jnp.array(imgs)
    target_image_inputs = jnp.array(templates)
    shfts_et_all = vmap_register_translation(src_image_inputs, target_image_inputs, upsample_factor_fft, lb_shifts,
                                             ub_shifts, max_shifts)

    shift_img_x = jnp.reshape(jnp.array(shfts_et_all[0])[:, 0], dim_grid)
    shift_img_y = jnp.reshape(jnp.array(shfts_et_all[0])[:, 1], dim_grid)
    diffs_phase_grid = jnp.reshape(jnp.array(shfts_et_all[2]), dim_grid)

    dims = img.shape

    x_grid, y_grid = jnp.meshgrid(jnp.arange(0., img.shape[1]).astype(
        jnp.float32), jnp.arange(0., img.shape[0]).astype(jnp.float32))

    remap_input_2 = jax.image.resize(shift_img_y.astype(jnp.float32), dims, method="cubic") + x_grid
    remap_input_1 = jax.image.resize(shift_img_x.astype(jnp.float32), dims, method="cubic") + y_grid
    m_reg = jax.scipy.ndimage.map_coordinates(img, [remap_input_1, remap_input_2], order=1, mode='nearest')

    shift_img_x_r = shift_img_x.reshape(num_tiles)
    shift_img_x_y = shift_img_y.reshape(num_tiles)
    total_shifts = jnp.stack([shift_img_x_r, shift_img_x_y], axis=1) * -1
    return m_reg - add_to_movie, total_shifts


register_frames_to_template_pwrigid = jit(
    vmap(_register_to_template_pwrigid, in_axes=(0, None, None, None, None, None, None, None, None, None)), \
    static_argnums=(2, 3, 4, 5, 7))

register_frames_to_template_pwrigid_docs = \
    """
    Perform piecewise rigid motion correction iteration by
    (1) dividing the FOV in patches
    (2) motion correcting each patch separately
    (3) upsampling the motion correction vector field
    (4) stiching back together the corrected subpatches
    
    Args:
        img (np.ndarray): Shape (T, x, y) Frames to register to template. T is number of frames, x and y spatial dims
        template (np.ndarray): Shape (x, y). The reference image
        strides_0 (int): The strides of the patches in which the FOV is subdivided along dimension 0.
        strides_1 (int): The strides of the patches in which the FOV is subdivided along dimension 1.
        overlaps_0 (int): Amount of pixel overlap between patches along dimension 0
        overlaps_1 (int): Amount of pixel overlap between patches along dimension 1
        max_shifts (tuple): Max shifts in x and y
        upsample_factor_fft (int): The resolution of fractional shifts
        max_deviation_rigid (int): Maximum deviation in shifts of each patch from the rigid shift (should not be large)
        add_to_movie (np.array): Constant offset to add to movie before registration to avoid negative values.
    
    Returns:
        new_img (jnp.array): Shape (T, x, y), motion corrected version of img. 
        total_shifts (jnp.array): Shifts applied to each patch.
    
    """

register_frames_to_template_pwrigid.__doc__ = register_frames_to_template_pwrigid_docs