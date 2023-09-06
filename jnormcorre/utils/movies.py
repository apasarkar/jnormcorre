
""" 
This file is a modified version of the movies.py file from the CaImAn repository: 
https://github.com/flatironinstitute/CaImAn/blob/master/caiman/base/movies.py

See README for complete attribution.
"""


from builtins import str
from builtins import range
from past.utils import old_div

import cv2
from functools import partial
import h5py
import logging
from matplotlib import animation
import numpy as np
import os
from PIL import Image  
import pylab as pl
import scipy.ndimage
import scipy
from scipy.io import loadmat
from skimage.transform import warp, AffineTransform
from skimage.feature import match_template
import sys
import tifffile
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
import warnings
from zipfile import ZipFile

from . import timeseries

try:
    cv2.setNumThreads(0)
except:
    pass

from . import timeseries as ts
from .traces import trace




##THIS IS A UTILITY FUNCTION
from typing import Tuple
def fn_relocated(fn:str) -> str:
    """ If the provided filename does not contain any path elements, this returns what would be its absolute pathname
        as located in get_tempdir(). Otherwise it just returns what it is passed.

        The intent behind this is to ease having functions that explicitly mention pathnames have them go where they want,
        but if all they think about is filenames, they go under CaImAn's notion of its temporary dir. This is under the
        principle of "sensible defaults, but users can override them".
    """
    if not 'CAIMAN_NEW_TEMPFILE' in os.environ: # XXX We will ungate this in a future version of caiman
        return fn
    if str(os.path.basename(fn)) == str(fn): # No path stuff
        return os.path.join(get_tempdir(), fn)
    else:
        return fn


import pathlib
from typing import Tuple
def prepare_shape(mytuple: Tuple) -> Tuple:
    """ This promotes the elements inside a shape into np.uint64. It is intended to prevent overflows
        with some numpy operations that are sensitive to it, e.g. np.memmap """
    if not isinstance(mytuple, tuple):
        raise Exception("Internal error: prepare_shape() passed a non-tuple")
    return tuple(map(lambda x: np.uint64(x), mytuple))

def load_memmap(filename: str, mode: str = 'r') -> Tuple[Any, Tuple, int]:
    """ Load a memory mapped file created by the function save_memmap

    Args:
        filename: str
            path of the file to be loaded
        mode: str
            One of 'r', 'r+', 'w+'. How to interact with files

    Returns:
        Yr:
            memory mapped variable

        dims: tuple
            frame dimensions

        T: int
            number of frames


    Raises:
        ValueError "Unknown file extension"

    """
    if pathlib.Path(filename).suffix != '.mmap':
        logging.error(f"Unknown extension for file {filename}")
        raise ValueError(f'Unknown file extension for file {filename} (should be .mmap)')
    # Strip path components and use CAIMAN_DATA/example_movies
    # TODO: Eventually get the code to save these in a different dir
    fn_without_path = os.path.split(filename)[-1]
    fpart = fn_without_path.split('_')[1:-1]  # The filename encodes the structure of the map
    d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]

    filename = fn_relocated(filename)
    Yr = np.memmap(filename, mode=mode, shape=prepare_shape((d1 * d2 * d3, T)), dtype=np.float32, order=order)
    if d3 == 1:
        return (Yr, (d1, d2), T)
    else:
        return (Yr, (d1, d2, d3), T)


#### UTILITY FUNCTION    
    
class movie(ts.timeseries):
    """
    Class representing a movie. This class subclasses timeseries,
    that in turn subclasses ndarray

    movie(input_arr, fr=None,start_time=0,file_name=None, meta_data=None)

    Example of usage:
        input_arr = 3d ndarray
        fr=33; # 33 Hz
        start_time=0
        m=movie(input_arr, start_time=0,fr=33);

    See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
    notes on objects that are descended from ndarray
    """

    def __new__(cls, input_arr, **kwargs):
        """
        Args:
            input_arr:  np.ndarray, 3D, (time,height,width)

            fr: frame rate

            start_time: time beginning movie, if None it is assumed 0

            meta_data: dictionary including any custom meta data

            file_name: name associated with the file (e.g. path to the original file)
        """

        if isinstance(input_arr, movie):
            return input_arr

        if (isinstance(input_arr, np.ndarray)) or \
           (isinstance(input_arr, h5py._hl.dataset.Dataset)) or \
           ('mmap' in str(type(input_arr))) or \
           ('tifffile' in str(type(input_arr))):
            return super().__new__(cls, input_arr, **kwargs)  
        else:
            raise Exception('Input must be an ndarray, use load instead!')

    def calc_min(self) -> 'movie':
        # todo: todocument

        tmp = []
        bins = np.linspace(0, self.shape[0], 10).round(0)
        for i in range(9):
            tmp.append(np.nanmin(self[int(bins[i]):int(bins[i + 1]), :, :]).tolist() + 1)
        minval = np.ndarray(1)
        minval[0] = np.nanmin(tmp)
        return movie(input_arr=minval)

    def motion_correct(self,
                       max_shift_w=5,
                       max_shift_h=5,
                       num_frames_template=None,
                       template=None,
                       method: str = 'opencv',
                       remove_blanks: bool = False,
                       interpolation: str = 'cubic') -> Tuple[Any, Tuple, Any, Any]:
        """
        Extract shifts and motion corrected movie automatically,

        for more control consider the functions extract_shifts and apply_shifts
        Disclaimer, it might change the object itself.

        Args:
            max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting
                                     in the width and height direction

            template: if a good template for frame by frame correlation exists
                      it can be passed. If None it is automatically computed

            method: depends on what is installed 'opencv' or 'skimage'. 'skimage'
                    is an order of magnitude slower

            num_frames_template: if only a subset of the movies needs to be loaded
                                 for efficiency/speed reasons


        Returns:
            self: motion corected movie, it might change the object itself

            shifts : tuple, contains x & y shifts and correlation with template

            xcorrs: cross correlation of the movies with the template

            template: the computed template
        """

        if template is None:   # if template is not provided it is created
            if num_frames_template is None:
                num_frames_template = old_div(10e7, (self.shape[1] * self.shape[2]))

            frames_to_skip = int(np.maximum(1, old_div(self.shape[0], num_frames_template)))

            # sometimes it is convenient to only consider a subset of the
            # movie when computing the median
            submov = self[::frames_to_skip, :].copy()
            templ = submov.bin_median()                                        # create template with portion of movie
            shifts, xcorrs = submov.extract_shifts(max_shift_w=max_shift_w,
                                                   max_shift_h=max_shift_h,
                                                   template=templ,
                                                   method=method)
            submov.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = submov.bin_median()
            del submov
            m = self.copy()
            shifts, xcorrs = m.extract_shifts(max_shift_w=max_shift_w,
                                              max_shift_h=max_shift_h,
                                              template=template,
                                              method=method)
            m = m.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = (m.bin_median())
            del m
        else:
            template = template - np.percentile(template, 8)

        # now use the good template to correct
        shifts, xcorrs = self.extract_shifts(max_shift_w=max_shift_w,
                                             max_shift_h=max_shift_h,
                                             template=template,
                                             method=method)
        self = self.apply_shifts(shifts, interpolation=interpolation, method=method)

        if remove_blanks:
            max_h, max_w = np.max(shifts, axis=0)
            min_h, min_w = np.min(shifts, axis=0)
            self.crop(crop_top=max_h,
                      crop_bottom=-min_h + 1,
                      crop_left=max_w,
                      crop_right=-min_w,
                      crop_begin=0,
                      crop_end=0)

        return self, shifts, xcorrs, template

    def bin_median(self, window: int = 10) -> np.ndarray:
        """ compute median of 3D array in along axis o by binning values

        Args:
            mat: ndarray
                input 3D matrix, time along first dimension

            window: int
                number of frames in a bin

        Returns:
            img:
                median image

        """
        T, d1, d2 = np.shape(self)
        num_windows = int(old_div(T, window))
        num_frames = num_windows * window
        return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    def extract_shifts(self, max_shift_w: int = 5, max_shift_h: int = 5, template=None,
                       method: str = 'opencv') -> Tuple[List, List]:
        """
        Performs motion correction using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.

        Args:
            max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting in the width and height direction

            template: if a good template for frame by frame correlation is available it can be passed. If None it is automatically computed

            method: depends on what is installed 'opencv' or 'skimage'. 'skimage' is an order of magnitude slower

        Returns:
            shifts : tuple, contains shifts in x and y and correlation with template

            xcorrs: cross correlation of the movies with the template

        Raises:
            Exception 'Unknown motion correction method!'

        """
        min_val = np.percentile(self, 1)
        if min_val < -0.1:
            logging.debug("min_val in extract_shifts: " + str(min_val))
            logging.warning('Movie average is negative. Removing 1st percentile.')
            self = self - min_val
        else:
            min_val = 0

        if not isinstance(self[0, 0, 0], np.float32):
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        _, h_i, w_i = self.shape

        ms_w = max_shift_w
        ms_h = max_shift_h

        if ms_w >= w_i / 2:
            raise ValueError(f"max_shift[0] must be smaller than half of image width")

        if ms_h >= h_i / 2:
            raise ValueError(f"max_shift[1] must be smaller than half of image height")

        if template is None:
            template = np.median(self, axis=0)
        else:
            if np.percentile(template, 8) < -0.1:
                logging.warning('Movie average is negative. Removing 1st percentile.')
                template = template - np.percentile(template, 1)

        template = template[ms_h:h_i - ms_h, ms_w:w_i - ms_w].astype(np.float32)

        #% run algorithm, press q to stop it
        shifts = []    # store the amount of shift in each frame
        xcorrs = []

        for i, frame in enumerate(self):
            if i % 100 == 99:
                logging.debug(f"Frame {i + 1}")
            if method == 'opencv':
                res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
                top_left = cv2.minMaxLoc(res)[3]
            elif method == 'skimage':
                res = match_template(frame, template)
                top_left = np.unravel_index(np.argmax(res), res.shape)
                top_left = top_left[::-1]
            else:
                raise Exception('Unknown motion correction method!')
            avg_corr = np.mean(res)
            sh_y, sh_x = top_left

            if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
                # if max is internal, check for subpixel shift using gaussian
                # peak registration
                log_xm1_y = np.log(res[sh_x - 1, sh_y])
                log_xp1_y = np.log(res[sh_x + 1, sh_y])
                log_x_ym1 = np.log(res[sh_x, sh_y - 1])
                log_x_yp1 = np.log(res[sh_x, sh_y + 1])
                four_log_xy = 4 * np.log(res[sh_x, sh_y])

                sh_x_n = -(sh_x - ms_h + old_div((log_xm1_y - log_xp1_y),
                                                 (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
                sh_y_n = -(sh_y - ms_w + old_div((log_x_ym1 - log_x_yp1),
                                                 (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
            else:
                sh_x_n = -(sh_x - ms_h)
                sh_y_n = -(sh_y - ms_w)

            shifts.append([sh_x_n, sh_y_n])
            xcorrs.append([avg_corr])

        self = self + min_val

        return (shifts, xcorrs)

    def apply_shifts(self, shifts, interpolation: str = 'linear', method: str = 'opencv', remove_blanks: bool = False):
        """
        Apply precomputed shifts to a movie, using subpixels adjustment (cv2.INTER_CUBIC function)

        Args:
            shifts: array of tuples representing x and y shifts for each frame

            interpolation: 'linear', 'cubic', 'nearest' or cvs.INTER_XXX

            method: (undocumented)

            remove_blanks: (undocumented)

        Returns:
            self

        Raise:
            Exception 'Interpolation method not available'

            Exception 'Method not defined'
        """
        if not isinstance(self[0, 0, 0], np.float32):
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        if interpolation == 'cubic':
            if method == 'opencv':
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = 3
            logging.debug('cubic interpolation')

        elif interpolation == 'nearest':
            if method == 'opencv':
                interpolation = cv2.INTER_NEAREST
            else:
                interpolation = 0
            logging.debug('nearest interpolation')

        elif interpolation == 'linear':
            if method == 'opencv':
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = 1
            logging.debug('linear interpolation')
        elif interpolation == 'area':
            if method == 'opencv':
                interpolation = cv2.INTER_AREA
            else:
                raise Exception('Method not defined')
            logging.debug('area interpolation')
        elif interpolation == 'lanczos4':
            if method == 'opencv':
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = 4
            logging.debug('lanczos/biquartic interpolation')
        else:
            raise Exception('Interpolation method not available')

        _, h, w = self.shape
        for i, frame in enumerate(self):
            if i % 100 == 99:
                logging.debug(f"Frame {i + 1}")

            sh_x_n, sh_y_n = shifts[i]

            if method == 'opencv':
                M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
                min_, max_ = np.min(frame), np.max(frame)
                self[i] = np.clip(cv2.warpAffine(frame, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT),
                                  min_, max_)

            elif method == 'skimage':

                tform = AffineTransform(translation=(-sh_y_n, -sh_x_n))
                self[i] = warp(frame, tform, preserve_range=True, order=interpolation)

            else:
                raise Exception('Unknown shift application method')

        if remove_blanks:
            max_h, max_w = np.max(shifts, axis=0)
            min_h, min_w = np.min(shifts, axis=0)
            self.crop(crop_top=max_h,
                      crop_bottom=-min_h + 1,
                      crop_left=max_w,
                      crop_right=-min_w,
                      crop_begin=0,
                      crop_end=0)

        return self

    def crop(self, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, crop_begin=0, crop_end=0) -> None:
        """
        Crop movie (inline)

        Args:
            crop_top/crop_bottom/crop_left,crop_right: (undocumented)

            crop_begin/crop_end: (undocumented)
        """
        t, h, w = self.shape
        self[:, :, :] = self[crop_begin:t - crop_end, crop_top:h - crop_bottom, crop_left:w - crop_right]

    def resize(self, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        """
        Resizing caiman movie into a new one. Note that the temporal
        dimension is controlled by fz and fx, fy, fz correspond to
        magnification factors. For example to downsample in time by
        a factor of 2, you need to set fz = 0.5.

        Args:
            fx (float):
                Magnification factor along x-dimension

            fy (float):
                Magnification factor along y-dimension

            fz (float):
                Magnification factor along temporal dimension

        Returns:
            self (caiman movie)
        """
        T, d1, d2 = self.shape
        d = d1 * d2
        elm = d * T
        max_els = 2**61 - 1    # the bug for sizes >= 2**31 is appears to be fixed now
        if elm > max_els:
            chunk_size = old_div((max_els), d)
            new_m: List = []
            logging.debug('Resizing in chunks because of opencv bug')
            for chunk in range(0, T, chunk_size):
                logging.debug([chunk, np.minimum(chunk + chunk_size, T)])
                m_tmp = self[chunk:np.minimum(chunk + chunk_size, T)].copy()
                m_tmp = m_tmp.resize(fx=fx, fy=fy, fz=fz, interpolation=interpolation)
                if len(new_m) == 0:
                    new_m = m_tmp
                else:
                    new_m = timeseries.concatenate([new_m, m_tmp], axis=0)

            return new_m
        else:
            if fx != 1 or fy != 1:
                logging.debug("reshaping along x and y")
                t, h, w = self.shape
                newshape = (int(w * fy), int(h * fx))
                mov = []
                logging.debug("New shape is " + str(newshape))
                for frame in self:
                    mov.append(cv2.resize(frame, newshape, fx=fx, fy=fy, interpolation=interpolation))
                self = movie(np.asarray(mov), **self.__dict__)
            if fz != 1:
                logging.debug("reshaping along z")
                t, h, w = self.shape
                self = np.reshape(self, (t, h * w))
                mov = cv2.resize(self, (h * w, int(fz * t)), fx=1, fy=fz, interpolation=interpolation)
                mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
                self = movie(mov, **self.__dict__)
                self.fr = self.fr * fz

        return self

    def guided_filter_blur_2D(self, guide_filter, radius: int = 5, eps=0):
        """
        performs guided filtering on each frame. See opencv documentation of cv2.ximgproc.guidedFilter
        """
        for idx, fr in enumerate(self):
            if idx % 1000 == 0:
                logging.debug("At index: " + str(idx))
            self[idx] = cv2.ximgproc.guidedFilter(guide_filter, fr, radius=radius, eps=eps)

        return self

    def bilateral_blur_2D(self, diameter: int = 5, sigmaColor: int = 10000, sigmaSpace=0):
        """
        performs bilateral filtering on each frame. See opencv documentation of cv2.bilateralFilter
        """
        if not isinstance(self[0, 0, 0], np.float32):
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        for idx, fr in enumerate(self):
            if idx % 1000 == 0:
                logging.debug("At index: " + str(idx))
            self[idx] = cv2.bilateralFilter(fr, diameter, sigmaColor, sigmaSpace)

        return self

    def gaussian_blur_2D(self,
                         kernel_size_x=5,
                         kernel_size_y=5,
                         kernel_std_x=1,
                         kernel_std_y=1,
                         borderType=cv2.BORDER_REPLICATE):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Args:
            kernel_size: double
                see opencv documentation of GaussianBlur
            kernel_std_: double
                see opencv documentation of GaussianBlur
            borderType: int
                see opencv documentation of GaussianBlur

        Returns:
            self: ndarray
                blurred movie
        """

        for idx, fr in enumerate(self):
            logging.debug(idx)
            self[idx] = cv2.GaussianBlur(fr,
                                         ksize=(kernel_size_x, kernel_size_y),
                                         sigmaX=kernel_std_x,
                                         sigmaY=kernel_std_y,
                                         borderType=borderType)

        return self

    def median_blur_2D(self, kernel_size: float = 3.0):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Args:
            kernel_size: double
                see opencv documentation of GaussianBlur

            kernel_std_: double
                see opencv documentation of GaussianBlur

            borderType: int
                see opencv documentation of GaussianBlur

        Returns:
            self: ndarray
                blurred movie
        """

        for idx, fr in enumerate(self):
            logging.debug(idx)
            self[idx] = cv2.medianBlur(fr, ksize=kernel_size)

        return self

    def to_2D(self, order='F') -> np.ndarray:
        [T, d1, d2] = self.shape
        d = d1 * d2
        return np.reshape(self, (T, d), order=order)

    def zproject(self, method: str = 'mean', cmap=pl.cm.gray, aspect='auto', **kwargs) -> np.ndarray:
        """
        Compute and plot projection across time:

        Args:
            method: String
                'mean','median','std'

            **kwargs: dict
                arguments to imagesc

        Raises:
             Exception 'Method not implemented'
        """
        # TODO: make the imshow optional
        # TODO: todocument
        if method == 'mean':
            zp = np.mean(self, axis=0)
        elif method == 'median':
            zp = np.median(self, axis=0)
        elif method == 'std':
            zp = np.std(self, axis=0)
        else:
            raise Exception('Method not implemented')
        pl.imshow(zp, cmap=cmap, aspect=aspect, **kwargs)
        return zp


def load(file_name: Union[str, List[str]],
         fr: float = 30,
         start_time: float = 0,
         meta_data: Dict = None,
         subindices=None,
         shape: Tuple[int, int] = None,
         var_name_hdf5: str = 'mov',
         in_memory: bool = False,
         is_behavior: bool = False,
         bottom=0,
         top=0,
         left=0,
         right=0,
         channel=None,
         outtype=np.float32,
         is3D: bool = False) -> Any:
    """
    load movie from file. Supports a variety of formats. tif, hdf5, npy and memory mapped. Matlab is experimental.

    Args:
        file_name: string or List[str]
            name of file. Possible extensions are tif, avi, npy, h5, n5, zarr (npz and hdf5 are usable only if saved by calblitz)

        fr: float
            frame rate

        start_time: float
            initial time for frame 1

        meta_data: dict
            dictionary containing meta information about the movie

        subindices: iterable indexes
            for loading only portion of the movie

        shape: tuple of two values
            dimension of the movie along x and y if loading from a two dimensional numpy array

        var_name_hdf5: str
            if loading from hdf5/n5 name of the dataset inside the file to load (ignored if the file only has one dataset)

        in_memory: bool=False
            This changes the behaviour of the function for npy files to be a readwrite rather than readonly memmap,
            And it adds a type conversion for .mmap files.
            Use of this flag is discouraged (and it may be removed in the future)

        bottom,top,left,right: (undocumented)

        channel: (undocumented)

        outtype: The data type for the movie

    Returns:
        mov: caiman.movie

    Raises:
        Exception 'Subindices not implemented'
    
        Exception 'Subindices not implemented'
    
        Exception 'Unknown file type'
    
        Exception 'File not found!'
    """
    # case we load movie from file
    if max(top, bottom, left, right) > 0 and isinstance(file_name, str):
        file_name = [file_name]        # type: ignore # mypy doesn't like that this changes type

    if isinstance(file_name, list):
        if shape is not None:
            logging.error('shape parameter not supported for multiple movie input')

        return load_movie_chain(file_name,
                                fr=fr,
                                start_time=start_time,
                                meta_data=meta_data,
                                subindices=subindices,
                                bottom=bottom,
                                top=top,
                                left=left,
                                right=right,
                                channel=channel,
                                outtype=outtype,
                                var_name_hdf5=var_name_hdf5,
                                is3D=is3D)

    elif isinstance(file_name, tuple):
        print(f'**** Processing input file {file_name} as individualframes *****')
        if shape is not None:
            # XXX Should this be an Exception?
            logging.error('movies.py:load(): A shape parameter is not supported for multiple movie input')
        else:
            return load_movie_chain(tuple([iidd for iidd in np.array(file_name)[subindices]]),
                     fr=fr, start_time=start_time,
                     meta_data=meta_data, subindices=None,
                     bottom=bottom, top=top, left=left, right=right,
                     channel = channel, outtype=outtype)

    # If we got here we're parsing a single movie file
    if max(top, bottom, left, right) > 0:
        logging.error('movies.py:load(): Parameters top,bottom,left,right are not supported for single movie input')

    if channel is not None:
        logging.error('movies.py:load(): channel parameter is not supported for single movie input')

    if os.path.exists(file_name):
        _, extension = os.path.splitext(file_name)[:2]

        extension = extension.lower()
        if extension == '.mat':
            logging.warning('Loading a *.mat file. x- and y- dimensions ' +
                            'might have been swapped.')
            byte_stream, file_opened = scipy.io.matlab.mio._open_file(file_name, appendmat=False)
            mjv, mnv = scipy.io.matlab.mio.get_matfile_version(byte_stream)
            if mjv == 2:
                extension = '.h5'

        if extension in ['.tif', '.tiff', '.btf']:  # load tif file
            with tifffile.TiffFile(file_name) as tffl:
                multi_page = True if tffl.series[0].shape[0] > 1 else False
                if len(tffl.pages) == 1:
                    logging.warning('Your tif file is saved a single page' +
                                    'file. Performance will be affected')
                    multi_page = False
                if subindices is not None:
                    # if isinstance(subindices, (list, tuple)): # is list or tuple:
                    if isinstance(subindices, list):  # is list or tuple:
                        if multi_page:
                            if len(tffl.series[0].shape) < 4:
                                input_arr = tffl.asarray(key=subindices[0])[:, subindices[1], subindices[2]]
                            else:  # 3D
                                shape = tffl.series[0].shape
                                ts = np.arange(shape[0])[subindices[0]]
                                input_arr = tffl.asarray(key=np.ravel(ts[:, None] * shape[1] +
                                                                      np.arange(shape[1]))
                                                         ).reshape((len(ts),) + shape[1:])[
                                    :, subindices[1], subindices[2], subindices[3]]
                        else:
                            input_arr = tffl.asarray()[tuple(subindices)]

                    else:
                        if multi_page:
                            if len(tffl.series[0].shape) < 4:
                                input_arr = tffl.asarray(key=subindices)
                            else:  # 3D
                                shape = tffl.series[0].shape
                                ts = np.arange(shape[0])[subindices]
                                input_arr = tffl.asarray(key=np.ravel(ts[:, None] * shape[1] +
                                                                      np.arange(shape[1]))
                                                         ).reshape((len(ts),) + shape[1:])
                        else:
                            input_arr = tffl.asarray(out='memmap')
                            input_arr = input_arr[subindices]

                else:
                    input_arr = tffl.asarray()

                input_arr = np.squeeze(input_arr)

        elif extension in ('.avi', '.mkv'):      # load video file
            cap = cv2.VideoCapture(file_name)

            try:
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except:
                logging.info('Roll back to opencv 2')
                length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

            cv_failed = False
            dims = [length, height, width]                     # type: ignore # a list in one block and a tuple in another
            if length == 0 or width == 0 or height == 0:       #CV failed to load
                cv_failed = True
            if subindices is not None:
                if not isinstance(subindices, list):
                    subindices = [subindices]
                for ind, sb in enumerate(subindices):
                    if isinstance(sb, range):
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif isinstance(sb, slice):
                        if sb.start is None:
                            sb = slice(0, sb.stop, sb.step)
                        if sb.stop is None:
                            sb = slice(sb.start, dims[ind], sb.step)
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif isinstance(sb, np.ndarray):
                        dims[ind] = sb.shape[0]

                start_frame = subindices[0][0]
            else:
                subindices = [np.r_[range(dims[0])]]
                start_frame = 0
            if not cv_failed:
                input_arr = np.zeros((dims[0], height, width), dtype=np.uint8)
                counter = 0
                cap.set(1, start_frame)
                current_frame = start_frame
                while True and counter < dims[0]:
                    # Capture frame-by-frame
                    if current_frame != subindices[0][counter]:
                        current_frame = subindices[0][counter]
                        cap.set(1, current_frame)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    input_arr[counter] = frame[:, :, 0]
                    counter += 1
                    current_frame += 1

                if len(subindices) > 1:
                    input_arr = input_arr[:, subindices[1]]
                if len(subindices) > 2:
                    input_arr = input_arr[:, :, subindices[2]]
            else:      #use pims to load movie
                import pims

                def rgb2gray(rgb):
                    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

                pims_movie = pims.Video(file_name)
                length = len(pims_movie)
                height, width = pims_movie.frame_shape[0:2]    #shape is (h,w,channels)
                input_arr = np.zeros((length, height, width), dtype=np.uint8)
                for i in range(len(pims_movie)):               #iterate over frames
                    input_arr[i] = rgb2gray(pims_movie[i])

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        elif extension == '.npy':      # load npy file
            if fr is None:
                fr = 30
            if in_memory:
                input_arr = np.load(file_name)
            else:
                input_arr = np.load(file_name, mmap_mode='r')

            if subindices is not None:
                input_arr = input_arr[subindices]

            if input_arr.ndim == 2:
                if shape is not None:
                    _, T = np.shape(input_arr)
                    d1, d2 = shape
                    input_arr = np.transpose(np.reshape(input_arr, (d1, d2, T), order='F'), (2, 0, 1))
                else:
                    input_arr = input_arr[np.newaxis, :, :]

        elif extension == '.mat':      # load npy file
            input_arr = loadmat(file_name)['data']
            input_arr = np.rollaxis(input_arr, 2, -3)
            if subindices is not None:
                input_arr = input_arr[subindices]

        elif extension == '.npz':      # load movie from saved file
            if subindices is not None:
                raise Exception('Subindices not implemented')
            with np.load(file_name) as f:
                return movie(**f).astype(outtype)

        elif extension in ('.hdf5', '.h5', '.nwb'):
           with h5py.File(file_name, "r") as f:

                if extension == '.nwb': # Apparently nwb files are specially-formatted hdf5 files
                    try:
                        fgroup = f[var_name_hdf5]['data']
                    except:
                        fgroup = f['acquisition'][var_name_hdf5]['data']
                else:
                    fgroup = f[var_name_hdf5]

                if var_name_hdf5 in f or var_name_hdf5 in f['acquisition']:
                    if subindices is None:
                        images = np.array(fgroup).squeeze()
                    else:
                        if type(subindices).__module__ == 'numpy':
                            subindices = subindices.tolist()
                        if len(fgroup.shape) > 3:
                            logging.warning(f'fgroup.shape has dimensionality greater than 3 {fgroup.shape} in load')
                        images = np.array(fgroup[subindices]).squeeze()

                    return movie(images.astype(outtype))
                else:
                    logging.debug('KEYS:' + str(f.keys()))
                    raise Exception('Key not found in hdf5 file')

        elif extension in ('.n5', '.zarr'):
           try:
               import z5py
           except ImportError:
               raise Exception("z5py library not available; if you need this functionality use the conda package")
           with z5py.File(file_name, "r") as f:
                fkeys = list(f.keys())
                if len(fkeys) == 1: # If the n5/zarr file we're parsing has only one dataset inside it, ignore the arg and pick that dataset
                    var_name_hdf5 = fkeys[0]

                fgroup = f[var_name_hdf5]

                if var_name_hdf5 in f or var_name_hdf5 in f['acquisition']:
                    if subindices is None:
                        images = np.array(fgroup).squeeze()
                    else:
                        if type(subindices).__module__ == 'numpy':
                            subindices = subindices.tolist()
                        if len(fgroup.shape) > 3:
                            logging.warning(f'fgroup.shape has dimensionality greater than 3 {fgroup.shape} in load')
                        images = np.array(fgroup[subindices]).squeeze()

                    return movie(images.astype(outtype))
                else:
                    logging.debug('KEYS:' + str(f.keys()))
                    raise Exception('Key not found in n5 or zarr file')

        elif extension == '.mmap':
            filename = os.path.split(file_name)[-1]
            Yr, dims, T = load_memmap(
                os.path.join(                  # type: ignore # same dims typing issue as above
                    os.path.split(file_name)[0], filename))
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            if subindices is not None:
                images = images[subindices]

            if in_memory:
                logging.debug('loading mmap file in memory')
                images = np.array(images).astype(outtype)

            logging.debug('mmap')
            return movie(images, fr=fr)

        elif extension == '.sbx':
            logging.debug('sbx')
            if subindices is not None:
                return movie(sbxreadskip(file_name[:-4], subindices), fr=fr).astype(outtype)
            else:
                return movie(sbxread(file_name[:-4], k=0, n_frames=np.inf), fr=fr).astype(outtype)

        elif extension == '.sima':
            raise Exception("movies.py:load(): FATAL: sima support was removed in 1.9.8")

        else:
            raise Exception('Unknown file type')
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception(f'File {file_name} not found!')

    return movie(input_arr.astype(outtype),
                 fr=fr,
                 start_time=start_time,
                 file_name=os.path.split(file_name)[-1],
                 meta_data=meta_data)


def load_movie_chain(file_list: List[str],
                     fr: float = 30,
                     start_time=0,
                     meta_data=None,
                     subindices=None,
                     var_name_hdf5: str = 'mov',
                     bottom=0,
                     top=0,
                     left=0,
                     right=0,
                     z_top=0,
                     z_bottom=0,
                     is3D: bool = False,
                     channel=None,
                     outtype=np.float32) -> Any:
    """ load movies from list of file names

    Args:
        file_list: list
           file names in string format
    
        the other parameters as in load_movie except
    
        bottom, top, left, right, z_top, z_bottom : int
            to load only portion of the field of view
    
        is3D : bool
            flag for 3d data (adds a fourth dimension)

    Returns:
        movie: movie
            movie corresponding to the concatenation og the input files

    """
    mov = []
    for f in tqdm(file_list):
        m = load(f,
                 fr=fr,
                 start_time=start_time,
                 meta_data=meta_data,
                 subindices=subindices,
                 in_memory=True,
                 outtype=outtype,
                 var_name_hdf5=var_name_hdf5)
        if channel is not None:
            logging.debug(m.shape)
            m = m[channel].squeeze()
            logging.debug(f"Movie shape: {m.shape}")

        if not is3D:
            if m.ndim == 2:
                m = m[np.newaxis, :, :]

            _, h, w = np.shape(m)
            m = m[:, top:h - bottom, left:w - right]
        else:
            if m.ndim == 3:
                m = m[np.newaxis, :, :, :]

            _, h, w, d = np.shape(m)
            m = m[:, top:h - bottom, left:w - right, z_top:d - z_bottom]

        mov.append(m)
    return ts.concatenate(mov, axis=0)


def loadmat_sbx(filename: str):
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_


def _check_keys(checkdict: Dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mio5_params.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> Dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, k: int = 0, n_frames=np.inf) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1

    # Paramters
    N = max_idx + 1    # Last frame
    N = np.minimum(N, n_frames)

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
    fo.seek(k * nSamples, 0)
    ii16 = np.iinfo(np.uint16)
    x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N)), order='F')

    x = x[0, :, :, :]

    fo.close()

    return x.transpose([2, 1, 0])


def sbxreadskip(filename: str, subindices: slice) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        slice: pass a slice to slice along the last dimension
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = int(os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1)

    # Paramters
    if isinstance(subindices, slice):
        if subindices.start is None:
            start = 0
        else:
            start = subindices.start

        if subindices.stop is None:
            N = max_idx + 1    # Last frame
        else:
            N = np.minimum(subindices.stop, max_idx + 1).astype(int)

        if subindices.step is None:
            skip = 1
        else:
            skip = subindices.step

        iterable_elements = range(start, N, skip)

    else:

        N = len(subindices)
        iterable_elements = subindices
        skip = 0

    N_time = len(list(iterable_elements))

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']
    assert nSamples >= 0

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones

    counter = 0

    if skip == 1:
        # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
        assert start * nSamples > 0
        fo.seek(start * nSamples, 0)
        ii16 = np.iinfo(np.uint16)
        x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * (N - start)))
        x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N - start)),
                      order='F')

        x = x[0, :, :, :]

    else:
        for k in iterable_elements:
            assert k >= 0
            if counter % 100 == 0:
                print(f'Reading Iteration: {k}')
            fo.seek(k * nSamples, 0)
            ii16 = np.iinfo(np.uint16)
            tmp = ii16.max - \
                np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * 1))

            tmp = tmp.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer'])), order='F')
            if counter == 0:
                x = np.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], N_time))

            x[:, :, :, counter] = tmp
            counter += 1

        x = x[0, :, :, :]
    fo.close()

    return x.transpose([2, 1, 0])


def sbxshape(filename: str) -> Tuple[int, int, int]:
    """
    Args:
        filename should be full path excluding .sbx
    """
    # TODO: Document meaning of return values

    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1
    N = max_idx + 1    # Last frame
    x = (int(info['sz'][1]), int(info['recordsPerBuffer']), int(N))
    return x


def to_3D(mov2D: np.ndarray, shape: Tuple, order='F') -> np.ndarray:
    """
    transform a vectorized movie into a 3D shape
    """
    return np.reshape(mov2D, shape, order=order)




def rolling_window(ndarr, window_size, stride):   
        """
        generates efficient rolling window for running statistics
        Args:
            ndarr: ndarray
                input pixels in format pixels x time
            window_size: int
                size of the sliding window
            stride: int
                stride of the sliding window
        Returns:
                iterator with views of the input array
                
        """
        for i in range(0,ndarr.shape[-1]-window_size-stride+1,stride): 
            yield ndarr[:,i:np.minimum(i+window_size, ndarr.shape[-1])]
            
        if i+stride != ndarr.shape[-1]:
           yield ndarr[:,i+stride:]


def load_iter(file_name, subindices=None, var_name_hdf5: str = 'mov', outtype=np.float32):
    """
    load iterator over movie from file. Supports a variety of formats. tif, hdf5, avi.

    Args:
        file_name: string
            name of file. Possible extensions are tif, avi and hdf5

        subindices: iterable indexes
            for loading only a portion of the movie

        var_name_hdf5: str
            if loading from hdf5 name of the variable to load

        outtype: The data type for the movie

    Returns:
        iter: iterator over movie

    Raises:
        Exception 'Subindices not implemented'

        Exception 'Unknown file type'

        Exception 'File not found!'
    """
    if os.path.exists(file_name):
        extension = os.path.splitext(file_name)[1].lower()
        if extension in ('.tif', '.tiff', '.btf'):
            Y = tifffile.TiffFile(file_name).pages
            if subindices is not None:
                if isinstance(subindices, range):
                    subindices = slice(subindices.start, subindices.stop, subindices.step)
                Y = Y[subindices]
            for y in Y:
                yield y.asarray().astype(outtype)
        elif extension in ('.avi', '.mkv'):
            cap = cv2.VideoCapture(file_name)
            if subindices is None:
                while True:
                    ret, frame = cap.read()
                    if ret:
                        yield frame[..., 0].astype(outtype)
                    else:
                        cap.release()
                        return
                        #raise StopIteration
            else:
                if isinstance(subindices, slice):
                    subindices = range(
                        subindices.start,
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if subindices.stop is None else subindices.stop,
                        1 if subindices.step is None else subindices.step)
                t = 0
                for ind in subindices:
#                    cap.set(1, ind)
#                    ret, frame = cap.read()
#                    if ret:
#                        yield frame[..., 0]
#                    else:
#                        raise StopIteration
                    while t <= ind:
                        ret, frame = cap.read()
                        t += 1
                    if ret:
                        yield frame[..., 0].astype(outtype)
                    else:
                        return
                        #raise StopIteration
                cap.release()

                return
                #raise StopIteration
        elif extension in ('.hdf5', '.h5', '.nwb', '.mat'):
            with h5py.File(file_name, "r") as f:
                Y = f.get('acquisition/' + var_name_hdf5 + '/data'
                           if extension == '.nwb' else var_name_hdf5)
                if subindices is None:
                    for y in Y:
                        yield y.astype(outtype)
                else:
                    if isinstance(subindices, slice):
                        subindices = range(subindices.start,
                                           len(Y) if subindices.stop is None else subindices.stop,
                                           1 if subindices.step is None else subindices.step)
                    for ind in subindices:
                        yield Y[ind].astype(outtype)
        else:  # fall back to memory inefficient version
            for y in load(file_name, var_name_hdf5=var_name_hdf5,
                          subindices=subindices, outtype=outtype):
                yield y
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception('File not found!')
