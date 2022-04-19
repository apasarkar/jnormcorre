import multiprocessing
import os
import shutil
import pathlib
import sys
import math
import glob

import numpy as np
import tifffile

from jnormcorre.utils.movies import load
from jnormcorre import motion_correction


from tqdm import tqdm

import time



def get_caiman_memmap_shape(filename):
    fn_without_path = os.path.split(filename)[-1]
    fpart = fn_without_path.split('_')[1:-1]  # The filename encodes the structure of the map
    d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]
    return (d1, d2, T)


def write_output_simple(targets, out_file, batch_size = 1000, dtype = np.float64):   
    with tifffile.TiffWriter(out_file, bigtiff=True) as tffw:
        for index in tqdm(range(len(targets))):
            file = targets[index]
            file_split = file.rsplit(".", maxsplit=1)
            shape = get_caiman_memmap_shape(file)
            num_iters = math.ceil(shape[2] / batch_size)
            for k in range(num_iters):
                start = k*batch_size
                end = min((k+1)*batch_size, shape[2])
                data = load(file, subindices=range(start, end)).astype(dtype)
                for j in range(min(end - start, batch_size)):
                    tffw.write(data[j, :, :], contiguous=True)           

 
  
    
def parinit():
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))

                    


def motion_correct():
    """
    Runs motion correction from caiman on the input dataset with the
    option to process the same dataset in multiple passes.
    Parameters
    ----------
    filename : string
        Full path + name for destination of output config file.
    outdir : string
        Full path to location where outputs should be written.
    fr: int
        Imaging rate in frames per second
    decay_time: float
        Length of a typical transient in seconds
    dxy: tuple (2 elements)
        Spatial resolution in x and y in (um per pixel)
    max_shift_um: tuple (2 elements)
        Maximum shift in um
    max_deviation_rigid: int
        Maximum deviation allowed for patch with respect to rigid shifts
    patch_motion_um: 
        Patch size for non rigid correction in um
    overlaps:
        Overlap between patches
    border_nan: 
        See linked caiman docs for details
    niter_rig: int
        Number of passes of rigid motion correction (used to estimate template)
    pw_rigid: boolean 
        Indicates whether or not to run piecewise rigid motion correction
    Returns
    -------
    None :
    """

    

    # Iteratively Run MC On Input File
    print("Running motion correction...")
    
    # Create multiprocessing pool to parallelize across batches of frames
    os.system('taskset -cp 0-%d %s' % (multiprocessing.cpu_count(), os.getpid()))
    dview = multiprocessing.Pool(initializer=parinit, processes=multiprocessing.cpu_count())

    
    max_shift_um = (12., 12.)
    max_deviation_rigid = 3
    patch_motion_um = (100., 100)
    dxy = (2., 2.)


    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    overlaps = (24,24)

    mc_dict = {
        'border_nan': 'copy',               # flag for allowing NaN in the boundaries
        'max_deviation_rigid': 3,           # maximum deviation between rigid and non-rigid
        'max_shifts': (6, 6),               # maximum shifts per dimension (in pixels)
        'min_mov': None,                    # minimum value of movie
        'niter_rig': 1,                     # number of iterations rigid motion correction
        'nonneg_movie': True,               # flag for producing a non-negative movie
        'num_frames_split': 80,             # split across time every x frames
        'num_splits_to_process_els': None,  # DO NOT MODIFY
        'num_splits_to_process_rig': None,  # DO NOT MODIFY
        'overlaps': (32, 32),               # overlap between patches in pw-rigid motion correction
        'pw_rigid': False,                  # flag for performing pw-rigid motion correction
        'shifts_opencv': True,              # flag for applying shifts using cubic interpolation (otherwise FFT)
        'splits_els': 14,                   # number of splits across time for pw-rigid registration
        'splits_rig': 14,                   # number of splits across time for rigid registration
        'strides': (96, 96),                # how often to start a new patch in pw-rigid registration
        'upsample_factor_grid': 4,          # motion field upsampling factor during FFT shifts
        'indices': (slice(None), slice(None)),  # part of FOV to be corrected
        'gSig_filt': None
    }


    ##UPDATE ABOVE DEFAULT PARAMETERS: 
    mc_dict['pw_rigid']=True
    mc_dict['strides'] = strides
    mc_dict['overlaps'] = overlaps
    mc_dict['max_deviation_rigid'] = 15
    mc_dict['border_nan'] = 'copy'
    mc_dict['niter_rig'] = 4
    mc_dict['gSig_filt'] = (10, 10)



#     target = ["test_data_amol/S9VisualCombine.tiff"]
    target = ["../test_data/FlowerBesselDataCombineCrop.tiff"]
#     target = ["test_data_amol/training2_moco2_{}.tif".format(i) for i in range(6)]

    corrector = motion_correction.MotionCorrect(target, dview=dview, **mc_dict)

    start_time = time.time()
    # Run MC, Always Saving Non-Final Outputs For Use In Next Iteration
    corrector.motion_correct(
        save_movie=True, 
    )
    
    
    print("that took {}".format(time.time() - start_time))
    
    target = corrector.mmap_file
    output_path = os.path.join("..", "test_data", "corrected_movie.tif")
    write_output_simple(target, output_path, dtype=np.float32)


    print("Motion correction completed.")
if __name__ == "__main__":

    
    # Run Single Pass Motion Correction
    motion_correct()