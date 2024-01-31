import datetime
import math
import os
import sys

import numpy as np
import tifffile

from jnormcorre import motion_correction
from jnormcorre.utils import registrationarrays

def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()


'''
dxy = (2., 2.),
max_shift_um = (12., 12.),
patch_motion_um = (100., 100.),
'''


def motion_correct(lazy_dataset,
                   outdir,
                   max_shifts,
                   max_deviation_rigid=3,
                   frames_per_split=1000,
                   pw_rigid=True,
                   strides=(30, 30),
                   overlaps=(10, 10),
                   niter_rig = 4,
                   gSig_filt=None,
                   save_movie=True,
                   sketch_template=False,
                   num_splits_to_process_els=None,
                   num_splits_to_process_rig=None):
    """
    Runs the full motion correction pipeline (with the option to do rigid and piecewise rigid registration after, if desired)
    Parameters
    ----------
    filename : string
        Full path + name for destination of output config file.
    outdir : string
        Full path to location where outputs should be written.
    max_shifts: tuple (2 elements)
        Max allowed rigid shifts when performing rigid registration
    frames_per_split: integer
        When we perform the motion correction, we register the data in chunks. This parameter dictates how many frames are in each chunk
    pw_rigid: Boolean
        Indicates whether after rigid registration, we perform piecewise-rigid registration

    ----
    The params here are for setting the size of the local patch when we perform piecewise-rigid registration on the data (so these are only relevant if pw_rigid
    is True

    (1) strides: tuple of two positive integers
    (2) overlaps: tuple of two positive integers

    Conceptually, we partition the field of view into overlapping rectangular patches of size (strides[0] + overlaps[0], strides[1] + overlaps[1]).
    The degree of overlap between patches is given by (overlaps[0], overlaps[1]).
    Critical Points:
        (A) strides[0] + overlaps[0] must be less than the first dimension of the field of view (first dimension in python indexing of course)
        (B) strides[1] + overlaps[1] must be less than the first dimension of the field of view (second dimension in python indexing of numpy.ndarray)
    If these conditions are not met, the algorithm will throw an error.
    ------

    niter_rig: int greater than or equal to 1
        This is the number of times we update the template when performing rigid registration
    gSig_filt: Boolean
        Indicates whether we apply a high-pass filter to the data before estimating what shifts to apply (and the template). Useful for data with high background contamination.
    save_movie: Boolean
        Indicates whether we save the fully registered movie.

    ------
    The below parameters control the template estimation step

    sketch_template: Boolean
        Motion correction relies on the ability to iteratively estimate a good template of the data (and register each frame to this template).
        If sketch_template is true, we will sample different temporal chunks of data at each iteration, registering them, updating the template, and repeating this process.
        This allows us to perform more template updates in aggregate.
        If save_movie is true, then at the last iteration (the last rigid iteration if we pw_rigid is False or just the pw_rigid registration step), we don't use this
        sketching strategy and instead take a pass through the whole data.
    num_splits_to_process_rig:
        If sketch_template is true, this parameter tells us how many "temporal chunks" of the data to look at in each template update step.
    num_splits_to_process_els:
        If sketch_template is True (and save_movie is False), this parameter tells us how many "temporal chunks" of the data to look at during piecewise rigid registration.
    -----

    min_mov: Float
        Indicates a known minimum value of the data, which is subtracted from frames before doing registration.
    indices: default (slice(None), slice(None))
        If one wants to register a spatial subset of the data, these indices "slice" the numpy ndarray data in the spatial dimensions accordingly.

    Returns
    -------
    frame_corrector_obj: this is an object which stores a template and some metadata about the registration procedure.
        The motivation for returning this is that it is a lightweight object for registering new data to a previously estimated template (applying both rigid and
        nonrigid registration, etc.)
        Furthermore, it contains the "just-in-time compiled" (jitted) jax function to do the registration. If you want to build your own data processing pipelines,
        this gives you a modular way to "compose" the registration function with the rest of your pipeline for fast end-to-end GPU processing.

    target_file: string
        This is path of the final filename. If save_movie was true, target_file will point to a new filename. If not, it won't.
    """



    mc_dict = {}
    mc_dict['upsample_factor_grid'] = 4  # This was reliably set to 4 in original method

    # Iteratively Run MC On Input File
    display("Running motion correction...")

    total_frames = lazy_dataset.n_frames
    splits = math.ceil(total_frames/frames_per_split)
    display("Number of chunks is {}".format(splits))


    mc_dict['strides'] = strides
    mc_dict['overlaps'] = overlaps
    mc_dict['max_shifts'] = max_shifts
    mc_dict['max_deviation_rigid'] = max_deviation_rigid
    if pw_rigid:
        mc_dict['pw_rigid'] = True
        mc_dict['niter_els'] = 1
        mc_dict['strides'] = strides
        mc_dict['overlaps'] = overlaps
    else:
        mc_dict['pw_rigid'] = False
    mc_dict['niter_rig'] = niter_rig

    if sketch_template:
        mc_dict['num_splits_to_process_els'] = min(5,
                                                   splits) if num_splits_to_process_els is None else num_splits_to_process_els
        mc_dict['num_splits_to_process_rig'] = min(5,
                                                   splits) if num_splits_to_process_rig is None else num_splits_to_process_rig
    mc_dict['gSig_filt'] = gSig_filt
    mc_dict['splits_els'] = splits
    mc_dict['splits_rig'] = splits

    corrector = motion_correction.MotionCorrect(lazy_dataset, **mc_dict)

    # Run MC, Always Saving Non-Final Outputs For Use In Next Iteration
    corrector_obj, target_file = corrector.motion_correct(
        save_movie=save_movie
    )

    display("Motion correction completed.")

    # Save Frame-wise Shifts
    display(f"Saving computed shifts to ({outdir})...")
    np.savez(os.path.join(outdir, "shifts.npz"),
             shifts_rig=corrector.shifts_rig,
             x_shifts_els=corrector.x_shifts_els if pw_rigid else None,
             y_shifts_els=corrector.y_shifts_els if pw_rigid else None)
    display('Shifts saved as "shifts.npz".')

    return corrector_obj, target_file

def main():
    print("IN PYCHARM")
    filename = "../datasets/demoMovie.tif"
    lazy_dataset = registrationarrays.TiffArray(filename)

    physical_params = True  # Turn this on or off based on how you want to set parameters and reason about your dataset

    if physical_params:  # Set the params by reasoning in terms of physical space (um)

        dxy = (2., 2.)  # This is the resolution of your imaging data (um per pixel)
        patch_motion_um = (50.,
                           50.)  # If you do piecewise rigid registration, this shows how the "tiles" are spaced out on the FOV (in um)
        strides = tuple([int(a / b) for a, b in zip(patch_motion_um,
                                                    dxy)])  # From the bio parameters, we can infer the pixel spacing between the tiles in X and Y dimensions here

        max_shift_um = (
            12.,
            12.)  # This is the maximum rigid shift of the data in um (so this is in physical space, not pixel space)
        max_shifts = [int(a / b) for a, b in zip(max_shift_um,
                                                 dxy)]  # Based on the above physical parameters, we can define the max shifts for rigid registration and the strides

        pw_rigid = True  # You can turn this off to disable piecewise rigid registration

        # Modify this to dictate how much these local patches (defined by "strides") overlap when doing piecewise rigid registration
        # As mentioned above in this notebook, overlaps[i] + strides[i] must be smaller than dataset.shape[i]
        overlaps = (round(strides[0] / 4), round(strides[1] / 4))

    else:  # Use this if you want to think in terms of pixels

        max_shifts = (6, 6)  # Max allowed shift in pixels for rigid registration
        pw_rigid = True

        # Read the docstring in the motion correction function above for how to set these params for your dataset
        strides = (30, 30)
        overlaps = (round(strides[0] / 4), round(strides[1] / 4))

    registration_obj, registered_filename = motion_correct(lazy_dataset, ".", max_shifts,
                                                           max_deviation_rigid=3,
                                                           frames_per_split=1000,
                                                           pw_rigid=pw_rigid,
                                                           strides=strides,
                                                           overlaps=overlaps,
                                                           niter_rig=4,
                                                           gSig_filt=None,
                                                           save_movie=True,
                                                           sketch_template=False,
                                                           num_splits_to_process_els=None,
                                                           num_splits_to_process_rig=None)


if __name__ == "__main__":
    main()