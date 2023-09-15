#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The functions apply_shifts_dft, register_translation, _compute_phasediff, and _upsampled_dft are from
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991.
These same functions were adapted from sckikit-image, licensed as follows:

Copyright (C) 2011, the scikit-image team
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. Neither the name of skimage nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
"""
import os


import jax
import torch
from past.builtins import basestring
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
import collections
import cv2
import gc
import h5py
import itertools
import logging
logging.basicConfig(level = logging.ERROR)
import numpy as np
from numpy.fft import ifftshift
import os
import sys
import pylab as pl
import tifffile
from typing import List, Optional, Tuple
from skimage.transform import resize as resize_sk
from skimage.transform import warp as warp_sk

from jnormcorre.onephotonmethods import get_kernel, high_pass_filter_cv, high_pass_batch
import jnormcorre.utils.movies as movies
from jnormcorre.utils.movies import load, load_iter

import pathlib ##MOVE THIS ONE...
from tqdm import tqdm 
from cv2 import dft as fftn
from cv2 import idft as ifftn

import math

## TODO: Check whether enable x64 is worth it
# config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial
import time

import random
import multiprocessing



### General object for motion correction
## Use case: you have a good template (maybe you run registration on 10K frames, or you used a clever method to find a great template from data. Now you want to register lots of frames to this template. This object is a transparent way to use that functionality from this repo
class frame_corrector():
    def __init__(self, corrector, batching=100):
        '''
        Inputs: 
            corrector: jnormcorre motion correction object (jnormcorre.motion_correction). This object stores the necessary info to do motion correction (templates, shifts, overlaps, etc.)
            batching: number of frames to register at a time. This helps exploit the JIT acceleration of motion correction (via caching) and but more importantly prevents the GPU from being overloaded with frames.
            register_flag: If this is false, this object does not register data
        '''
        
        if corrector.pw_rigid:
            self.corr_method = "pwrigid"
        else:
            self.corr_method = "rigid"
        
        self.max_shifts = corrector.max_shifts
        
        self.upsample_factor_fft = 10 ##NOTE: This is a hardcoded value in the original normcorre method, eventually actually treat it like a constant
        self.add_to_movie = -corrector.min_mov
        self.strides = corrector.strides
        self.overlaps = corrector.overlaps
        self.max_deviation_rigid = corrector.max_deviation_rigid
        self.batching = batching
        
        if self.corr_method == "pwrigid":
            self.template = corrector.total_template_els
            self.registration_method = jit(vmap(tile_and_correct_ideal, in_axes = (0, None, None, None, None, None, None, None, None, None)), static_argnums=(2,3,4,5,7))
                           
            def simplified_registration_func(frames):
                return self.registration_method(frames, self.template, self.strides[0], self.strides[1], \
                                                                     self.overlaps[0], self.overlaps[1], self.max_shifts,self.upsample_factor_fft, \
                                                                     self.max_deviation_rigid, self.add_to_movie)[0]
            self.jitted_method = simplified_registration_func
            
        elif self.corr_method == "rigid":
            self.template = corrector.total_template_rig
            
            self.registration_method = vmap(tile_and_correct_rigid, in_axes=(0, None, None, None))
            
            def simplified_registration_func(frames):
                return self.registration_method(frames, self.template, self.max_shifts, self.add_to_movie)[0]
            self.jitted_method = jit(simplified_registration_func)
        else:
            raise ValueError("Invalid method provided. Must either be pwrigid or rigid")
    
    def register_frames(self, frames):
        '''
        Inputs: 
            frames: np.ndarray, dimensions (T, d1, d2), where T is the number of frames and d1, d2 are the FOV dimensions
        Outputs: 
            corrected_frames: jnp.array. Dimensions (T, d1, d2). The registered output from the input (frames)
        
        TODO: Add logic to get 
        '''
        return self.jitted_method(frames)     



####PLACE IN FUNCTIONS######


def get_file_size(file_name, var_name_hdf5='mov'):
    """ Computes the dimensions of a file or a list of files without loading
    it/them in memory. An exception is thrown if the files have FOVs with
    different sizes
        Args:
            file_name: str/filePath or various list types
                locations of file(s)

            var_name_hdf5: 'str'
                if loading from hdf5 name of the dataset to load

        Returns:
            dims: tuple
                dimensions of FOV

            T: int or tuple of int
                number of timesteps in each file
    """
    if isinstance(file_name, pathlib.Path):
        # We want to support these as input, but str has a broader set of operations that we'd like to use, so let's just convert.
	# (specifically, filePath types don't support subscripting)
        file_name = str(file_name)
    if isinstance(file_name, str):
        if os.path.exists(file_name):
            _, extension = os.path.splitext(file_name)[:2]
            extension = extension.lower()
            if extension == '.mat':
                byte_stream, file_opened = scipy.io.matlab.mio._open_file(file_name, appendmat=False)
                mjv, mnv = scipy.io.matlab.mio.get_matfile_version(byte_stream)
                if mjv == 2:
                    extension = '.h5'
            if extension in ['.tif', '.tiff', '.btf']:
                tffl = tifffile.TiffFile(file_name)
                siz = tffl.series[0].shape
                T, dims = siz[0], siz[1:]
            elif extension in ('.avi', '.mkv'):
                cap = cv2.VideoCapture(file_name)
                dims = [0, 0]
                try:
                    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dims[1] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    dims[0] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except():
                    print('Roll back to opencv 2')
                    T = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                    dims[1] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    dims[0] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            elif extension == '.mmap':
                filename = os.path.split(file_name)[-1]
                Yr, dims, T = load_memmap(os.path.join(
                        os.path.split(file_name)[0], filename))
            elif extension in ('.h5', '.hdf5', '.nwb'):
                # FIXME this doesn't match the logic in movies.py:load()
                # Consider pulling a lot of the "data source" code out into one place
                with h5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    if len(kk) == 1 and var_name_hdf5 not in f:
                        siz = f[kk[0]].shape
                    elif var_name_hdf5 in f:
                        if extension == '.nwb':
                            siz = f[var_name_hdf5]['data'].shape
                        else:
                            siz = f[var_name_hdf5].shape
                    elif var_name_hdf5 in f['acquisition']:
                        siz = f['acquisition'][var_name_hdf5]['data'].shape
                    else:
                        logging.error('The file does not contain a variable' +
                                      'named {0}'.format(var_name_hdf5))
                        raise Exception('Variable not found. Use one of the above')
                T, dims = siz[0], siz[1:]
            elif extension in ('.n5', '.zarr'):
                try:
                    import z5py
                except:
                    raise Exception("z5py not available; if you need this use the conda-based setup")

                with z5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    if len(kk) == 1:
                        siz = f[kk[0]].shape
                    elif var_name_hdf5 in f:
                        if extension == '.nwb':
                            siz = f[var_name_hdf5]['data'].shape
                        else:
                            siz = f[var_name_hdf5].shape
                    elif var_name_hdf5 in f['acquisition']:
                        siz = f['acquisition'][var_name_hdf5]['data'].shape
                    else:
                        logging.error('The file does not contain a variable' +
                                      'named {0}'.format(var_name_hdf5))
                        raise Exception('Variable not found. Use one of the above')
                T, dims = siz[0], siz[1:]
            elif extension in ('.sbx'):
                raise ValueError("sbx file type no longer supported")
            else:
                raise Exception('Unknown file type')
            dims = tuple(dims)
        else:
            raise Exception('File not found!')
    elif isinstance(file_name, tuple):
        dims = load(file_name[0], var_name_hdf5=var_name_hdf5).shape
        T = len(file_name)

    elif isinstance(file_name, list):
        if len(file_name) == 1:
            dims, T = get_file_size(file_name[0], var_name_hdf5=var_name_hdf5)
        else:
            dims, T = zip(*[get_file_size(fn, var_name_hdf5=var_name_hdf5)
                for fn in file_name])
            if len(set(dims)) > 1:
                raise Exception('Files have FOVs with different sizes')
            else:
                dims = dims[0]
    else:
        raise Exception('Unknown input type')
    return dims, T


def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    # Some functions calling this have the first part of *their* dims Tuple be the number of frames.
    # They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}_.mmap"

def tiff_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    # Some functions calling this have the first part of *their* dims Tuple be the number of frames.
    # They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}_.tiff"


def prepare_shape(mytuple: Tuple) -> Tuple:
    """ This promotes the elements inside a shape into np.uint64. It is intended to prevent overflows
        with some numpy operations that are sensitive to it, e.g. np.memmap """
    if not isinstance(mytuple, tuple):
        raise Exception("Internal error: prepare_shape() passed a non-tuple")
    return tuple(map(lambda x: np.uint64(x), mytuple))



####### END OF PLACE IN FUNCTIONS #######

class MotionCorrect(object):
    """
        class implementing motion correction operations
       """

    def __init__(self, fname, min_mov=None, max_shifts=(6, 6), niter_rig=1, niter_els=1, splits_rig=14,
                 num_splits_to_process_rig=None,num_splits_to_process_els=None, strides=(96, 96), overlaps=(32, 32),
                 splits_els=14, upsample_factor_grid=4, max_deviation_rigid=3, nonneg_movie=True, pw_rigid=False,
                 var_name_hdf5='mov', indices=(slice(None), slice(None)), gSig_filt=None, bigtiff=False):
        """
        Constructor class for motion correction operations

        Args:
           fname: str
               path to file to motion correct

           min_mov: int16 or float32
               estimated minimum value of the movie to produce an output that is positive

           max_shifts: tuple
               maximum allow rigid shift

           niter_rig':int
               maximum number of iterations rigid motion correction, in general is 1. 0
               will quickly initialize a template with the first frames
               
           niter_els:int
                maximum number of iterations of piecewise rigid motion correction. Default value of 1

           splits_rig': int
            for parallelization split the movies in num_splits chuncks across time

           num_splits_to_process_rig: list,
               For rigid and piecewise rigid motion correction, the template is often update over many iterations. If there are "n" iterations, then num_spits_to_process_rig tells us how many splits (chunks of data) look at per iteration.  
               num_splits_to_process_rig are considered
               
            num_splits_to_process_rig: list,
               if none all the splits are processed and the movie is saved, otherwise at each iteration
               num_splits_to_process_rig are considered

           strides: tuple
               intervals at which patches are laid out for motion correction

           overlaps: tuple
               overlap between pathes (size of patch strides+overlaps)

           pw_rigid: bool, default: False
               flag for performing motion correction when calling motion_correct

           splits_els':list
               for parallelization split the movies in num_splits chuncks across time

           num_splits_to_process_els: list,
               if none all the splits are processed and the movie is saved  otherwise at each iteration
                num_splits_to_process_els are considered

           upsample_factor_grid:int,
               upsample factor of shifts per patches to avoid smearing when merging patches

           max_deviation_rigid:int
               maximum deviation allowed for patch with respect to rigid shift

           nonneg_movie: boolean
               make the SAVED movie and template mostly nonnegative by removing min_mov from movie

           var_name_hdf5: str, default: 'mov'
               If loading from hdf5, name of the variable to load

           indices: tuple(slice), default: (slice(None), slice(None))
               Use that to apply motion correction only on a part of the FOV
               
           gSig_filt: tuple. Default None. 
                Contains 2 components describing the dimensions of a kernel. We use the kernel to high-pass filter data which has large background contamination.

       Returns:
           self

        """
        if 'ndarray' in str(type(fname)):
            logging.info('Creating file for motion correction "tmp_mov_mot_corr.hdf5"')
            movies.movie(fname).save('tmp_mov_mot_corr.hdf5', var_name_hdf5=var_name_hdf5)
            fname = ['tmp_mov_mot_corr.hdf5']

        if not isinstance(fname, list):
            fname = [fname]

        if not isinstance(niter_els, int) or niter_els < 1:
            raise ValueError(f"please provide n_iter as an int of 1 or higher.")

        if not isinstance(var_name_hdf5, str):
            raise ValueError(f"pleaes provide 'var_name_hdf5' as string")

        # if max_shifts

        self.fname = fname
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.niter_els = niter_els
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.pw_rigid = bool(pw_rigid)
        self.var_name_hdf5 = var_name_hdf5
        self.indices = indices
        self.bigtiff = bigtiff
        
        #In case gSig_filt is not None, we define a kernel which we use for 1p processing:
        if gSig_filt is not None: 
            self.filter_kernel = get_kernel(gSig_filt)
        else:
            self.filter_kernel = None

    def motion_correct(self, template=None, save_movie=False):
        """general function for performing all types of motion correction. The
        function will perform either rigid or piecewise rigid motion correction
        depending on the attribute self.pw_rigid. A template can be passed, and the
        output can be saved as a memory mapped file.

        Args:
            template: ndarray, default: None
                template provided by user for motion correction

            save_movie: bool, default: False
                flag for saving motion corrected file(s) as memory mapped file(s)

        Returns:
            self
        """
        # TODO: Review the docs here, and also why we would ever return self
        #       from a method that is not a constructor
        if self.min_mov is None:
            if self.filter_kernel is None:
                iterator = load_iter(self.fname[0], var_name_hdf5=self.var_name_hdf5)
                mi = np.inf
                for _ in range(400):
                    try:
                        mi = min(mi, next(iterator).min()[()])
                    except StopIteration:
                        break
                self.min_mov = mi
            else: 
                self.min_mov = np.array([high_pass_filter_cv(m_, self.filter_kernel)
                    for m_ in movies.load(self.fname[0], var_name_hdf5=self.var_name_hdf5,
                                      subindices=slice(400))]).min()


        if self.pw_rigid:
            self.motion_correct_pwrigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                    np.max(np.abs(self.y_shifts_els))))
        else:
            self.motion_correct_rigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))
        self.border_to_0 = b0.astype(int)
        self.target_file = self.fname_tot_els if self.pw_rigid else self.fname_tot_rig
        
        frame_correction_obj = frame_corrector(self)
        return frame_correction_obj, self.target_file

    def motion_correct_rigid(self, template=None, save_movie=False) -> None:
        """
        Perform rigid motion correction

        Args:
            template: ndarray 2D 
                if known, one can pass a template to register the frames to

            save_movie_rigid:Bool
                save the movies vs just get the template

        Important Fields:
            self.fname_tot_rig: name of the mmap file saved

            self.total_template_rig: template updated by iterating  over the chunks

            self.templates_rig: list of templates. one for each chunk

            self.shifts_rig: shifts in x and y per frame
        """
        logging.debug('Entering Rigid Motion Correction')
        logging.debug(-self.min_mov)  # XXX why the minus?
        self.total_template_rig = template
        self.templates_rig:List = []
        self.fname_tot_rig:List = []
        self.shifts_rig:List = []

        for fname_cur in self.fname:
            _fname_tot_rig, _total_template_rig, _templates_rig, _shifts_rig = motion_correct_batch_rigid(
                fname_cur,
                self.max_shifts,
                splits=self.splits_rig,
                num_splits_to_process=self.num_splits_to_process_rig,
                num_iter=self.niter_rig,
                template=self.total_template_rig,
                save_movie_rigid=save_movie,
                add_to_movie=-self.min_mov,
                nonneg_movie=self.nonneg_movie,
                var_name_hdf5=self.var_name_hdf5,
                indices=self.indices,
                filter_kernel=self.filter_kernel,
                bigtiff=self.bigtiff)
            if template is None:
                self.total_template_rig = _total_template_rig

            self.templates_rig += _templates_rig
            self.fname_tot_rig += [_fname_tot_rig]
            self.shifts_rig += _shifts_rig

    def motion_correct_pwrigid(self, save_movie:bool=True, template:np.ndarray=None, show_template:bool=False) -> None:
        """Perform pw-rigid motion correction

        Args:
            save_movie:Bool
                save the movies vs just get the template

            template: ndarray 2D 
                if known, one can pass a template to register the frames to

            show_template: boolean
                whether to show the updated template at each iteration

        Important Fields:
            self.fname_tot_els: name of the mmap file saved
            self.templates_els: template updated by iterating  over the chunks
            self.x_shifts_els: shifts in x per frame per patch
            self.y_shifts_els: shifts in y per frame per patch
            self.z_shifts_els: shifts in z per frame per patch 
            self.coord_shifts_els: coordinates associated to the patch for
            values in x_shifts_els and y_shifts_els
            self.total_template_els: list of templates. one for each chunk

        Raises:
            Exception: 'Error: Template contains NaNs, Please review the parameters'
        """

        num_iter = self.niter_els
        if template is None:
            logging.info('Generating template by rigid motion correction')
            self.motion_correct_rigid(save_movie=False)
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.fname_tot_els:List = []
        self.templates_els:List = []
        self.x_shifts_els:List = []
        self.y_shifts_els:List = []

        self.coord_shifts_els:List = []
        for name_cur in self.fname:
            _fname_tot_els, new_template_els, _templates_els,\
                _x_shifts_els, _y_shifts_els, _z_shifts_els, _coord_shifts_els = motion_correct_batch_pwrigid(
                    name_cur, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
                    upsample_factor_grid=self.upsample_factor_grid,
                    max_deviation_rigid=self.max_deviation_rigid, splits=self.splits_els,
                    num_splits_to_process=self.num_splits_to_process_els, num_iter=num_iter, template=self.total_template_els,
                    save_movie=save_movie, nonneg_movie=self.nonneg_movie, var_name_hdf5=self.var_name_hdf5,
                    indices=self.indices, filter_kernel=self.filter_kernel, bigtiff=self.bigtiff)

            if show_template:
                pl.imshow(new_template_els)
                pl.pause(.5)
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




def bin_median(mat, window=10, exclude_nans=True):
    """ compute median of 3D array in along axis o by binning values

    Args:
        mat: ndarray
            input 3D matrix, time along first dimension

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    Raises:
        Exception 'Path to template does not exist:'+template
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

def _upsampled_dft_no_size(data, upsample_factor):
    return np.array(_upsampled_dft_jax_no_size(data, upsample_factor))

# @partial(jit, static_argnums=(1,))
def _upsampled_dft_jax(data, upsampled_region_size,
                   upsample_factor, axis_offsets):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

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
        data : 2D array
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    

    
    #Calculate col_kernel
    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)
    
    #ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(old_div(data.shape[1], 2))
    term_A = shifted - jnp.floor(data.shape[1]/2)
    
    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis = 0) - axis_offsets[1]
    
    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )
    
    
    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))
    
    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - axis_offsets[0]
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0]/2)
    
    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )
    
    output = jnp.tensordot(row_kernel, data, axes = [1,0])
    output = jnp.tensordot(output, col_kernel, axes = [1,0])

    return output


@partial(jit)
def _upsampled_dft_jax_no_size(data, upsample_factor):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

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
        data : 2D array
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    

    upsampled_region_size = 1
    
    
    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)
    
    term_A = shifted - jnp.floor(data.shape[1]/2)
    
    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis = 0) - 0
    
    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )
    
    
    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))
    
    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - 0
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0]/2)
    
    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )
    
    output = jnp.tensordot(row_kernel, data, axes = [1,0])
    output = jnp.tensordot(output, col_kernel, axes = [1,0])

    return output



### CODE FOR REGISTER TRANSLATION FIRST CALL

# @partial(jit)
def _compute_phasediff(cross_correlation_max):
    '''
    Compute global phase difference between the two images (should be zero if images are non-negative).
    Args:
        cross_correlation_max : complex
    The complex value of the cross correlation at its maximum point.
    
    '''
    return jnp.angle(cross_correlation_max)


# #Eventually get rid of this wrapper..
def get_freq_comps(src_image, target_image):
    a, b = get_freq_comps_jax(src_image, target_image)
    return np.array(a, dtype=np.complex128), np.array(b, dtype=np.complex128)

# def format_array(x):
#     return np.array([np.real(x), np.imag(x)])

# @partial(jit)
def get_freq_comps_jax(src_image, target_image):
    src_image_cpx = jnp.complex64(src_image)
    target_image_cpx = jnp.complex64(target_image)
    src_freq = jnp.fft.fftn(src_image_cpx)
    src_freq = jnp.divide(src_freq, jnp.size(src_freq))
    target_freq = jnp.fft.fftn(target_image_cpx)
    target_freq = jnp.divide(target_freq, jnp.size(target_freq))
    return src_freq, target_freq


# @partial(jit)
def threshold_dim1(img, ind):
    a = img.shape[0]
    
    row_ind_first = jnp.arange(a) < ind 
    row_ind_second = jnp.arange(a) > a - ind - 1
    
    prod = row_ind_first + row_ind_second
    
    broadcasted = jnp.broadcast_to(jnp.expand_dims(prod, axis = 1), img.shape)
    return broadcasted * img

# @partial(jit)
def threshold_dim2(img, ind):
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
def register_translation_jax_simple(src_image, target_image, upsample_factor, max_shifts=(10, 10)):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
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
    
    midpoints = jnp.array([jnp.fix(shape[0]/2), jnp.fix(shape[1]/2)])

    
    shifts = jnp.array(maxima, dtype=jnp.float32)

    first_shift = jax.lax.cond(shifts[0] > midpoints[0], subtract_values, return_identity, *(shifts[0], shape[0]))
    second_shift = jax.lax.cond(shifts[1] > midpoints[1], subtract_values, return_identity, *(shifts[1], shape[1]))
    shifts = jnp.array([first_shift, second_shift])


    shifts = jnp.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(upsample_factor*1.5 + 0.5)
    # Center of output array at dftshift + 1
    dftshift = jnp.fix(upsampled_region_size/ 2.0)
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

### END OF CODE FOR REGISTER TRANSLATION FIRST CALL



### START OF CODE FOR REGISTER TRANSLATION SECOND CALL

# @partial(jit, static_argnums=(1,))
def _upsampled_dft_jax_full(data, upsampled_region_size,
                   upsample_factor, axis_offsets):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

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
        data : 2D array
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    

    
    #Calculate col_kernel
    multiplier = (-1j * 2 * jnp.pi / (data.shape[1] * upsample_factor))
    shifted = jnp.fft.ifftshift(jnp.arange(data.shape[1]))
    shifted = jnp.expand_dims(shifted, axis=1)
    
    #ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(old_div(data.shape[1], 2))
    term_A = shifted - jnp.floor(data.shape[1]/2)
    
    term_B = jnp.expand_dims(jnp.arange(upsampled_region_size), axis = 0) - axis_offsets[1]
    
    col_kernel = jnp.exp(
        multiplier * jnp.dot(term_A, term_B)
    )
    
    
    multiplier = (-1j * 2 * jnp.pi / (data.shape[0] * upsample_factor))
    
    term_A = jnp.expand_dims(jnp.arange(upsampled_region_size), 1) - axis_offsets[0]
    term_B = jnp.expand_dims(jnp.fft.ifftshift(jnp.arange(data.shape[0])), axis=0) - jnp.floor(data.shape[0]/2)
    
    row_kernel = jnp.exp(
        (multiplier) * jnp.dot(term_A, term_B)
    )
    
    output = jnp.tensordot(row_kernel, data, axes = [1,0])
    output = jnp.tensordot(output, col_kernel, axes = [1,0])

    return output


# @partial(jit)
def threshold_shifts_0_if(new_cross_corr, shift_ub, shift_lb):
    ## In this case, shift_lb is negative and shift_ub is nonnegative
    a, b = new_cross_corr.shape
    first_thres = np.arange(a) < shift_ub
    second_thres = np.arange(a) >= a + shift_lb
    prod = first_thres + second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis = 1),\
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod

# @partial(jit)
def threshold_shifts_0_else(new_cross_corr, shift_ub, shift_lb):
    #In this case shift_lb is nonnegative OR shift_ub is negative, we can go case by case
    a, b = new_cross_corr.shape
    lb_threshold = jax.lax.cond(shift_lb >= 0, lambda p, q : q,\
                                lambda p, q : p + q, *(a, shift_lb))
    first_thres = np.arange(a) >= lb_threshold
    ub_threshold = jax.lax.cond(shift_ub >= 0, lambda p, q: q,\
                                lambda p, q: p + q, *(a, shift_ub)) 
    
    second_thres = np.arange(a) < ub_threshold
    prod = first_thres * second_thres 
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis = 1),\
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit)
def threshold_shifts_1_if(new_cross_corr, shift_ub, shift_lb):
    ## In this case, shift_lb is negative and shift_ub is nonnegative
    a, b = new_cross_corr.shape
    first_thres = np.arange(b) < shift_ub
    second_thres = np.arange(b) >= b + shift_lb
    prod = first_thres + second_thres
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis = 0),\
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod

# @partial(jit)
def threshold_shifts_1_else(new_cross_corr, shift_ub, shift_lb):
    #In this case shift_lb is nonnegative OR shift_ub is negative, we can go case by case
    a, b = new_cross_corr.shape
    lb_threshold = jax.lax.cond(shift_lb >= 0, lambda p, q : q,\
                                lambda p, q : p + q, *(b, shift_lb))
    first_thres = np.arange(b) >= lb_threshold
    ub_threshold = jax.lax.cond(shift_ub >= 0, lambda p, q: q,\
                                lambda p, q: p + q, *(a, shift_ub)) 
    
    second_thres = np.arange(b) < ub_threshold
    prod = first_thres * second_thres 
    expanded_prod = jnp.broadcast_to(jnp.expand_dims(prod, axis = 0),\
                                     new_cross_corr.shape)
    return new_cross_corr * expanded_prod


# @partial(jit, static_argnums=(2,))
def register_translation_jax_full(src_image, target_image, upsample_factor,\
                                  shifts_lb,shifts_ub,max_shifts=(10, 10)):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
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
    
    midpoints = jnp.array([jnp.fix(shape[0]/2), jnp.fix(shape[1]/2)])
    
    shifts = jnp.array(maxima, dtype=jnp.float32)

    first_shift = jax.lax.cond(shifts[0] > midpoints[0], subtract_values, return_identity, *(shifts[0], shape[0]))
    second_shift = jax.lax.cond(shifts[1] > midpoints[1], subtract_values, return_identity, *(shifts[1], shape[1]))
    shifts = jnp.array([first_shift, second_shift])


    shifts = jnp.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(upsample_factor*1.5 + 0.5)
    # Center of output array at dftshift + 1
    dftshift = jnp.fix(upsampled_region_size/ 2.0)
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

# vmap_register_translation = jit(vmap(register_translation_jax_full,\
#                                      in_axes=(0,0, None, None, None, None)), static_argnums=(2,))

vmap_register_translation = vmap(register_translation_jax_full, in_axes=(0,0, None, None, None, None))


### END OF CODE FOR REGSISTER TRANSLATION SECOND CALL

def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10)):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    '''
    src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10)
    '''
    ##Now, must FFT the data:
    src_freq, target_freq = get_freq_comps(src_image, target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = jnp.multiply(src_freq, jnp.conj(target_freq))

    
    cross_correlation = jnp.fft.ifftn(image_product)

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2))
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(
            np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float32)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft_full(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float32)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft_no_size(src_freq * src_freq.conj(), upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft_no_size(target_freq * target_freq.conj(), upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

#########
### apply_shifts function + helper code
########

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
    interm = jax.lax.cond(a<b, second_value, first_value, a,b)
    return jnp.ceil(interm)

# @partial(jit)
def floor_min(a, b):
    interm = jax.lax.cond(a>b, second_value, first_value, a,b)
    return jnp.fix(interm)

def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True):
    return np.array(apply_shifts_dft_fast_1(src_freq, shifts[0], shifts[1], diffphase, is_freq))

# @partial(jit)
def apply_shifts_dft_fast_1(src_freq_in, shift_a, shift_b, diffphase):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """


    src_freq = jnp.complex64(src_freq_in)
    



    nc, nr = src_freq.shape
    val_1 = -int(nr/2)
    val_2 = int(nr/2. + 0.5)
    val_3 = -int(nc/2)
    val_4 = int(nc/2. + 0.5)
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



# @partial(jit)
def fill_minw(img, k):
    x, y = img.shape
    key = y + k
    
    filter_mat = (jnp.arange(y) < key).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x,y))
    
    img_filter = filter_mat * img
    
    addend = (jnp.arange(y) >= key).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x,y))
    addend = addend * img[:, k-1, None]
    
    return img_filter + addend


# @partial(jit)
def fill_maxw(img, k):
    x,y = img.shape
    filter_mat = (jnp.arange(y) >= k).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x,y))
    
    img_filtered = filter_mat * img
    
    addend = (jnp.arange(y) < k).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x,y))
    addend = addend * img[:, k, None]
    
    return img_filtered + addend


# @partial(jit)
def fill_maxh(img, k):
    x, y = img.shape
    filter_mat = jnp.reshape((jnp.arange(x) >= k), (-1, 1)).astype(jnp.int32)
    filter_mat = jnp.broadcast_to(filter_mat, (x, y))
    img_filtered = img * filter_mat
    
    addend_binary = jnp.reshape((jnp.arange(x) < k), (-1, 1))
    addend_binary = jnp.broadcast_to(addend_binary, (x, y))
    addend_binary = addend_binary * img[k]
    return addend_binary + img_filtered


# @partial(jit)
def fill_minh(img, k):
    x, y = img.shape
    key = x + k
    filtered_mat = jnp.reshape((jnp.arange(x) < key), (-1, 1)).astype(jnp.int32)
    filtered_mat = jnp.broadcast_to(filtered_mat, (x, y))
    
    filtered_img = img * filtered_mat
    
    addend = jnp.reshape((jnp.arange(x) >= key), (-1, 1)).astype(jnp.int32)
    addend = jnp.broadcast_to(addend, (x,y))
    addend_final = addend * img[key-1]
    
    return filtered_img + addend_final


# @partial(jit)
def return_identity_mins(in_var, k):
    return in_var


#%%
def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args: 
        img:ndarray 2D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])


def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Args:
        img: original image, ndarray

        shapes, overlaps, strides:  tuples
            shapes, overlaps and strides of the patches

    Returns:
        weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)

    max_grid_1, max_grid_2 = np.max(
        np.array([it[:2] for it in sliding_window(img, overlaps, strides)]), 0)

    for grid_1, grid_2, _, _, _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 > 0:
            weight_mat[:overlaps[0], :] = np.linspace(
                0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0]:,
                       :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, :overlaps[1]] = weight_mat[:, :overlaps[1]
                                                     ] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1]:] = weight_mat[:, -
                                                      overlaps[1]:] * np.linspace(1, 0, overlaps[1])[None, :]

        yield weight_mat

# @partial(jit, static_argnums=(4,))
def tile_and_correct_rigid_1p(img, img_filtered, template, max_shifts, add_to_movie):
    
    upsample_factor_fft = 10
    img = jnp.add(img, add_to_movie).astype(jnp.float32)
    template = jnp.add(template, add_to_movie).astype(jnp.float32)


    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img_filtered, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)
    
    #Second input doesn't matter here
    sfr_freq, _ = get_freq_comps_jax(img, img) 

    new_img = apply_shifts_dft_fast_1(sfr_freq, -rigid_shts[0], -rigid_shts[1], diffphase)

    return new_img - add_to_movie, jnp.array([-rigid_shts[0], -rigid_shts[1]])

tile_and_correct_rigid_1p_vmap = jit(vmap(tile_and_correct_rigid_1p, in_axes=(0, 0, None, (None, None), None)))
        
        
# @partial(jit, static_argnums=(3,))
def tile_and_correct_rigid(img, template, max_shifts, add_to_movie):
    
    upsample_factor_fft = 10
    
    img = jnp.add(img, add_to_movie).astype(jnp.float32)
    template = jnp.add(template, add_to_movie).astype(jnp.float32)

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_jax_simple(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    new_img = apply_shifts_dft_fast_1(sfr_freq, -rigid_shts[0], -rigid_shts[1], diffphase)

    return new_img - add_to_movie, jnp.array([-rigid_shts[0], -rigid_shts[1]])

tile_and_correct_rigid_vmap = jit(vmap(tile_and_correct_rigid, in_axes=(0, None, None, None)))
 

@partial(jit, static_argnums=(1,2,3,4))
def get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim = jnp.arange(0, img.shape[0] - overlaps_0 - strides_0, strides_0)
    first_dim = jnp.append(first_dim,img.shape[0] - overlaps_0 - strides_0 )
    
    second_dim = jnp.arange(0, img.shape[1] - overlaps_1 - strides_1, strides_1)
    second_dim = jnp.append(second_dim, img.shape[1] - overlaps_1 - strides_1)
    return first_dim, second_dim

@partial(jit, static_argnums=(3,4))
def crop_image(img, x, y, length_1, length_2):
    out = jax.lax.dynamic_slice(img, (x,y), (length_1, length_2))
    return out

# crop_image_vmap = jit(vmap(crop_image, in_axes=(None, 0, 0, None, None)), static_argnums=(3,4))
crop_image_vmap = vmap(crop_image, in_axes=(None, 0, 0, None, None))

# @partial(jit, static_argnums=(1,2,3,4))
def get_patches_jax(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim, second_dim = get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1)
    product = jnp.array(jnp.meshgrid(first_dim, second_dim)).T.reshape((-1, 2))
    first_dim_new = product[:, 0]
    second_dim_new = product[:, 1]
    return crop_image_vmap(img, first_dim_new, second_dim_new, overlaps_0+strides_0, overlaps_1+strides_1)

# @partial(jit, static_argnums=(1,2,3,4))
def get_xy_grid(img, overlaps_0, overlaps_1, strides_0, strides_1):
    first_dim, second_dim = get_indices(img, overlaps_0, overlaps_1, strides_0, strides_1)
    first_dim_updated = np.arange(jnp.size(first_dim))
    second_dim_updated = np.arange(jnp.size(second_dim))
    product = jnp.array(jnp.meshgrid(first_dim_updated, second_dim_updated)).T.reshape((-1, 2))
    return product  
 
# @partial(jit, static_argnums=(3,4,5,6,8))
def tile_and_correct_pwrigid_1p(img, img_filtered, template, strides_0, strides_1, overlaps_0, overlaps_1, max_shifts, upsample_factor_fft,\
                     max_deviation_rigid, add_to_movie):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


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
    shfts_et_all = vmap_register_translation(src_image_inputs, target_image_inputs, upsample_factor_fft, lb_shifts, ub_shifts, max_shifts)

    


    shift_img_y = jnp.reshape(jnp.array(shfts_et_all[0])[:, 1], dim_grid)
    shift_img_x = jnp.reshape(jnp.array(shfts_et_all[0])[:, 0], dim_grid)
    diffs_phase_grid = jnp.reshape(jnp.array(shfts_et_all[2]), dim_grid)

    

    dims = img.shape
    
    

    x_grid, y_grid = jnp.meshgrid(jnp.arange(0., img.shape[1]).astype(
        jnp.float32), jnp.arange(0., img.shape[0]).astype(jnp.float32))
    
    
    
    remap_input_2 = jax.image.resize(shift_img_y.astype(jnp.float32), dims, method="cubic") + x_grid
    remap_input_1 = jax.image.resize(shift_img_x.astype(jnp.float32), dims, method="cubic") + y_grid
    m_reg = jax.scipy.ndimage.map_coordinates(img, [remap_input_1, remap_input_2],order=1, mode='nearest')
    

    
    shift_img_x_r = shift_img_x.reshape(num_tiles)
    shift_img_x_y = shift_img_y.reshape(num_tiles)
    total_shifts = jnp.stack([shift_img_x_r, shift_img_x_y], axis=1) * -1
    return m_reg - add_to_movie, total_shifts   

   
tile_and_correct_pwrigid_1p_vmap = jit(vmap(tile_and_correct_pwrigid_1p, in_axes = (0, 0, None, None, None, None, None, None, None, None, None)), \
                           static_argnums=(3,4,5,6,8))    

   
    
@partial(jit, static_argnums=(2,3,4,5,7))
def tile_and_correct(img, template, strides_0, strides_1, overlaps_0, overlaps_1, max_shifts, upsample_factor_fft,\
                     max_deviation_rigid, add_to_movie):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


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
    shfts_et_all = vmap_register_translation(src_image_inputs, target_image_inputs, upsample_factor_fft, lb_shifts, ub_shifts, max_shifts)

    


    shift_img_y = jnp.reshape(jnp.array(shfts_et_all[0])[:, 1], dim_grid)
    shift_img_x = jnp.reshape(jnp.array(shfts_et_all[0])[:, 0], dim_grid)
    diffs_phase_grid = jnp.reshape(jnp.array(shfts_et_all[2]), dim_grid)

    

    dims = img.shape
    
    

    x_grid, y_grid = jnp.meshgrid(jnp.arange(0., img.shape[1]).astype(
        jnp.float32), jnp.arange(0., img.shape[0]).astype(jnp.float32))
    
    shift_img_x_r = shift_img_x.reshape(num_tiles)
    shift_img_x_y = shift_img_y.reshape(num_tiles)
    total_shifts = jnp.stack([shift_img_x_r, shift_img_x_y], axis=1) * -1
    return img, shift_img_x, shift_img_y, x_grid, y_grid, total_shifts   
    
    
def opencv_interpolation(img, dims, shift_img_x, shift_img_y, x_grid, y_grid, add_value):
    img = np.array(img)
    shift_img_x = np.array(shift_img_x)
    shift_img_y = np.array(shift_img_y)
    x_grid = np.array(x_grid)
    y_grid = np.array(y_grid)
    m_reg = cv2.remap(img, cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
                      cv2.resize(shift_img_x.astype(np.float32), dims[::-1]) + y_grid,
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return m_reg - add_value

    
# Note that jax does not have higher-order spline implemented, eventually switch to that
# @partial(jit, static_argnums=(2,3,4,5,7))
def tile_and_correct_ideal(img, template, strides_0, strides_1, overlaps_0, overlaps_1, max_shifts, upsample_factor_fft,\
                     max_deviation_rigid, add_to_movie):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


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
    shfts_et_all = vmap_register_translation(src_image_inputs, target_image_inputs, upsample_factor_fft, lb_shifts, ub_shifts, max_shifts)

    

    shift_img_x = jnp.reshape(jnp.array(shfts_et_all[0])[:, 0], dim_grid)
    shift_img_y = jnp.reshape(jnp.array(shfts_et_all[0])[:, 1], dim_grid)
    diffs_phase_grid = jnp.reshape(jnp.array(shfts_et_all[2]), dim_grid)

    

    dims = img.shape
    
    

    x_grid, y_grid = jnp.meshgrid(jnp.arange(0., img.shape[1]).astype(
        jnp.float32), jnp.arange(0., img.shape[0]).astype(jnp.float32))
    
    
    
    remap_input_2 = jax.image.resize(shift_img_y.astype(jnp.float32), dims, method="cubic") + x_grid
    remap_input_1 = jax.image.resize(shift_img_x.astype(jnp.float32), dims, method="cubic") + y_grid
    m_reg = jax.scipy.ndimage.map_coordinates(img, [remap_input_1, remap_input_2],order=1, mode='nearest')
    

    
    shift_img_x_r = shift_img_x.reshape(num_tiles)
    shift_img_x_y = shift_img_y.reshape(num_tiles)
    total_shifts = jnp.stack([shift_img_x_r, shift_img_x_y], axis=1) * -1
    return m_reg - add_to_movie, total_shifts   


tile_and_correct_pwrigid_vmap = jit(vmap(tile_and_correct_ideal, in_axes = (0, None, None, None, None, None, None, None, None, None)), \
                           static_argnums=(2,3,4,5,7))    


def motion_correct_batch_rigid(fname, max_shifts, splits=56, num_splits_to_process=None, num_iter=1,
                               template=None, save_movie_rigid=False, add_to_movie=None,
                               nonneg_movie=False, subidx=slice(None, None, 1), var_name_hdf5='mov',
                               indices=(slice(None), slice(None)), filter_kernel = None, bigtiff=False):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        max_shifts: tuple
            x and y (and z if 3D) maximum allowed shifts

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        save_movie_rigid: boolean
             toggle save movie

        subidx: slice
            Indices to slice

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV
           
        filter_kernel: ndarray. Default: None. 
            Used to high-pass filter 1p data for template estimation/registration

    Returns:
         fname_tot_rig: str

         total_template:ndarray

         templates:list
              list of produced templates, one per batch

         shifts: list
              inferred rigid shifts to correct the movie

    Raises:
        Exception 'The movie contains nans. Nans are not allowed!'

    """

    dims, T = get_file_size(fname, var_name_hdf5=var_name_hdf5)
    Ts = np.arange(T)[subidx].shape[0]
    step = Ts // 50
    corrected_slicer = slice(subidx.start, subidx.stop, step + 1)
    m = load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)

    if len(m.shape) < 3:
        m = load(fname, var_name_hdf5=var_name_hdf5)
        m = m[corrected_slicer]
        logging.warning("Your original file was saved as a single page " +
                        "file. Consider saving it in multiple smaller files" +
                        "with size smaller than 4GB (if it is a .tif file)")

    m = m[:, indices[0], indices[1]]

    if template is None:
        if filter_kernel is not None:
            m = movies.movie(
                np.array([high_pass_filter_cv(filter_kernel, m_) for m_ in m]))
            
        if not m.flags['WRITEABLE']:
            m = m.copy()
        template = bin_median(
                m.motion_correct(max_shifts[1], max_shifts[0], template=None)[0])

    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        logging.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    save_movie = save_movie_rigid
    fname_tot_rig = None
    res_rig:List = []
    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1:
            save_movie = save_movie_rigid
            logging.debug('saving!')

        if isinstance(fname, tuple):
            base_name=os.path.split(fname[0])[-1][:-4] + '_rig_'
        else:
            base_name=os.path.split(fname)[-1][:-4] + '_rig_'

        if iter_ == num_iter - 1 and save_movie_rigid:
            save_flag = True
        else:
            save_flag = False
            
        if iter_ == num_iter - 1 and save_flag: #Idea: If we are saving out the full movie, sampling to just get the template is insufficient
            num_splits_to_process = None
        logging.info("We are about to enter motion correction piecewise")
        fname_tot_rig, res_rig = motion_correction_piecewise(fname, splits, strides=None, overlaps=None,
                                                             add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                                             save_movie=save_flag, base_name=base_name, subidx = subidx,
                                                             num_splits=num_splits_to_process, nonneg_movie=nonneg_movie,
                                                             var_name_hdf5=var_name_hdf5, indices=indices,
                                                             filter_kernel=filter_kernel, bigtiff=bigtiff)
        
        if filter_kernel is not None:
            new_templ = high_pass_filter_cv(filter_kernel, new_templ)
        else:
            new_templ = np.nanmedian(np.dstack([r[-1] for r in res_rig]), -1)

    total_template = new_templ
    templates = []
    shifts:List = []
    for rr in res_rig:
        shift_info, idxs, tmpl = rr
        templates.append(tmpl)
        shifts += [sh[0] for sh in shift_info[:len(idxs)]]

    return fname_tot_rig, total_template, templates, shifts

def motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None, newstrides=None,
                                 upsample_factor_grid=4, max_deviation_rigid=3,
                                 splits=56, num_splits_to_process=None, num_iter=1,
                                 template=None, save_movie=False, nonneg_movie=False,
                                 var_name_hdf5='mov', indices=(slice(None), slice(None)),
                                 filter_kernel=None, bigtiff=False):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        strides: tuple
            strides of patches along x and y 

        overlaps:
            overlaps of patches along x and y. example: If strides = (64,64) and overlaps (32,32) patches will be (96,96)

        newstrides: tuple
            overlaps after upsampling

        newoverlaps: tuple
            strides after upsampling

        max_shifts: tuple
            x and y maximum allowed shifts 

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        save_movie_rigid: boolean
             toggle save movie

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
        fname_tot_rig: str

        total_template:ndarray

        templates:list
            list of produced templates, one per batch

        shifts: list
            inferred rigid shifts to corrrect the movie

    Raises:
        Exception 'You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function'
    """
    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        logging.error('The template contains NaNs. NaNs are not allowed!')
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()

        if iter_ == num_iter - 1:
            save_movie = save_movie
            if save_movie:

                if isinstance(fname, tuple):
                    logging.debug(f'saving mmap of {fname[0]} to {fname[-1]}')
                else:
                    logging.debug(f'saving mmap of {fname}')

        if isinstance(fname, tuple):
            base_name=os.path.split(fname[0])[-1][:-4] + '_els_'
        else:
            base_name=os.path.split(fname)[-1][:-4] + '_els_'

        if iter_ == num_iter - 1 and save_movie:
            save_flag = True
        else:
            save_flag = False
            
        if iter_ == num_iter - 1 and save_flag:
            num_splits_to_process = None
        fname_tot_els, res_el = motion_correction_piecewise(fname, splits, strides, overlaps,
                                                            add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts,
                                                            max_deviation_rigid=max_deviation_rigid,
                                                            newoverlaps=newoverlaps, newstrides=newstrides,
                                                            upsample_factor_grid=upsample_factor_grid, order='F', save_movie=save_flag,
                                                            base_name=base_name, num_splits=num_splits_to_process,
                                                            nonneg_movie=nonneg_movie, var_name_hdf5=var_name_hdf5,
                                                            indices=indices, filter_kernel=filter_kernel, bigtiff=bigtiff)

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
        for shift_info, _ in zip(shift_info_chunk, idxs_chunk):
            total_shift, _, xy_grid = shift_info
            x_shifts.append(np.array([sh[0] for sh in total_shift]))
            y_shifts.append(np.array([sh[1] for sh in total_shift]))
            coord_shifts.append(xy_grid)

    return fname_tot_els, total_template, templates, x_shifts, y_shifts, z_shifts, coord_shifts


def generate_template_chunk(arr, batch_size = 250000):
    dim_1_step = int(math.sqrt(batch_size))
    dim_2_step = int(math.sqrt(batch_size))
    
    dim1_net_iters = math.ceil(arr.shape[1]/dim_1_step)
    dim2_net_iters = math.ceil(arr.shape[2]/dim_2_step)
    
    total_output = np.zeros((arr.shape[1], arr.shape[2]))
    for k in range(dim1_net_iters):
        for j in range(dim2_net_iters):
            start_dim1 = k*dim_1_step
            end_dim1 = min(start_dim1 +dim_1_step, arr.shape[1])
            start_dim2 =k*dim_2_step
            end_dim2 = min(start_dim2 + dim_2_step, arr.shape[2])
            total_output[start_dim1:end_dim1, start_dim2:end_dim2] = nan_processing(arr[:, start_dim1:end_dim1, start_dim2:end_dim2])
            
    return total_output
            


@partial(jit)
def nan_processing(arr):
    p = jnp.nanmean(arr, 0)
    q = jnp.nanmin(p)
    r = jnp.nan_to_num(p, q)
    return r

class tile_and_correct_dataset():
    def __init__(self, param_list):
        self.param_list = param_list
    
    def __len__(self):
        return len(self.param_list)
    
    def __getitem__(self, index):
        img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        nonneg_movie, is_fiji, var_name_hdf5, indices, filter_kernel= self.param_list[index]
        
        imgs = load(img_name, subindices=idxs, var_name_hdf5=var_name_hdf5)
        imgs = np.array(imgs[(slice(None),) + indices])
        mc = np.zeros(imgs.shape, dtype=np.float32)
        
        return imgs, mc,img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        nonneg_movie, is_fiji, var_name_hdf5, indices, filter_kernel



def regular_collate(batch):
    # print("the type of batch is {}".format(type(batch)))
    # print("the length of this batch is {}".format(len(batch)))
    return batch[0]

def tile_and_correct_dataloader(param_list, split_constant=200, bigtiff=False):
    """Does motion correction on specified image frames

    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image (to get individ frames, use out_fname to write them to disk)

    Notes:
    Also writes corrected frames to the mmap file specified by out_fname (if not None)

    """
    # todo todocument


    try:
        cv2.setNumThreads(0)
    except:
        pass  # 'Open CV is naturally single threaded'

    num_workers = 0
    prefetch_factor = 0
    tile_and_correct_dataobj = tile_and_correct_dataset(param_list)
    loader_obj= torch.utils.data.DataLoader(tile_and_correct_dataobj, batch_size=1,
                                             shuffle=False, num_workers=num_workers, collate_fn=regular_collate, timeout=0)
    
    results_list = []
    for dataloader_index, data in enumerate(tqdm(loader_obj), 0):
        num_iters = math.ceil(data[0].shape[0]/split_constant)
        imgs_net, mc,img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        nonneg_movie, is_fiji, var_name_hdf5, indices, filter_kernel = data
        for j in range(num_iters):
            
            start_pt = split_constant * j
            end_pt = min(data[0].shape[0], start_pt + split_constant)
            imgs = imgs_net[start_pt:end_pt, :, :]
            if isinstance(img_name, tuple):
                name, extension = os.path.splitext(img_name[0])[:2]
            else:
                name, extension = os.path.splitext(img_name)[:2]
            extension = extension.lower()
            shift_info = []

            load_time = time.time()
            upsample_factor_fft = 10 #Was originally hardcoded...

            if not imgs[0].shape == template.shape:
                template = template[indices]
            if max_deviation_rigid == 0:
                if filter_kernel is None: 
                    outs= tile_and_correct_rigid_vmap(imgs, template, max_shifts, add_to_movie)
                else:
                    imgs_filtered = high_pass_batch(filter_kernel, imgs)
                    outs = tile_and_correct_rigid_1p_vmap(imgs, imgs_filtered, template, max_shifts, add_to_movie)
                mc[start_pt:end_pt, :, :] = outs[0]
                temp_Nones_1 = [None for temp_i in range(outs[1].shape[0])]
                shift_info.extend(list(zip(np.array(outs[1]), temp_Nones_1, temp_Nones_1)))
            else:
                if filter_kernel is None:
                    outs = tile_and_correct_pwrigid_vmap(imgs, template, strides[0], strides[1], overlaps[0], overlaps[1], \
                                                                                       max_shifts,upsample_factor_fft, max_deviation_rigid, add_to_movie)
                else:
                    imgs_filtered = high_pass_batch(filter_kernel, imgs)
                    outs = tile_and_correct_pwrigid_1p_vmap(imgs, imgs_filtered, template, strides[0], strides[1], overlaps[0], overlaps[1], \
                                                                                       max_shifts,upsample_factor_fft, max_deviation_rigid, add_to_movie) 


                mc[start_pt:end_pt, :, :] = outs[0]
                temp_Nones_1 = [None for temp_i in range(outs[1].shape[0])]
                shift_info.extend(list(zip(np.array(outs[1]), temp_Nones_1, temp_Nones_1)))

                
        if out_fname is not None:
             tifffile.imwrite(out_fname, mc, append=True, metadata=None, bigtiff=bigtiff)

        new_temp = generate_template_chunk(mc)
        
        results_list.append((shift_info, idxs, new_temp))
        
    return results_list
  
def calculate_splits(T, splits):
    '''
    Heuristic for calculating splits
    '''
    out = np.array_split(list(range(T)), splits)
    return out

def load_split_heuristic(d1, d2, T):
    '''
    Heuristic for determining how many frames to register at a time (to avoid GPU OOM)
    '''
    
    if d1*d2 > 512*512:
        new_T = 20
    elif d1*d2 > 100000:
        new_T = 100
    else:
        new_T = 2000
    
    return min(T, new_T)

def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template=None,
                                max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', save_movie=True,
                                base_name=None, subidx = None, num_splits=None, nonneg_movie=False, var_name_hdf5='mov',
                                indices=(slice(None), slice(None)), filter_kernel=None, bigtiff=False):
    """
    TODO DOCUMENT
    """
    
    if isinstance(fname, tuple):
        name, extension = os.path.splitext(fname[0])[:2]
    else:
        name, extension = os.path.splitext(fname)[:2]
    extension = extension.lower()
    is_fiji = False

    dims, T = get_file_size(fname, var_name_hdf5=var_name_hdf5)
    z = np.zeros(dims)
    dims = z[indices].shape
    logging.debug('Number of Splits: {}'.format(splits))
    
    if isinstance(splits, int):
        if subidx is None:
            rng = range(T)
        else:
            rng = range(T)[subidx]

        idxs = calculate_splits(T, splits)
        
    else:
        idxs = splits
        save_movie = False
    if template is None:
        raise Exception('Not implemented')
    
    shape_mov = (np.prod(dims), T)
    if num_splits is not None:
        num_splits = min(num_splits, len(idxs))
        idxs = random.sample(idxs, num_splits)
        save_movie = False

    if save_movie:
        if base_name is None:
            base_name = os.path.split(fname)[1][:-4]
        fname_tot:Optional[str] = tiff_frames_filename(base_name, dims, T, order)
        if isinstance(fname, tuple):
            fname_tot = os.path.join(os.path.split(fname[0])[0], fname_tot)
        else:
            fname_tot = os.path.join(os.path.split(fname)[0], fname_tot)
            
            
        if os.path.exists(fname_tot):
            os.remove(fname_tot)
            print(f"File '{fname_tot}' already exists, likely from a different run of motion correction. It will be overwritten.")


        logging.info('Saving file as {}'.format(fname_tot))
    else:
        fname_tot = None

    pars = []
    for idx in idxs:
        logging.debug('Processing: frames: {}'.format(idx))
        pars.append([fname, fname_tot, np.array(idx), shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
            newoverlaps, newstrides, nonneg_movie, is_fiji, var_name_hdf5, indices, filter_kernel])

    import time
    start_time = time.time()
    split_constant = load_split_heuristic(dims[0], dims[1], T)
    res = tile_and_correct_dataloader(pars, split_constant=split_constant, bigtiff=bigtiff)
    print("this motion correction step took {}".format(time.time() - start_time))
    return fname_tot, res
