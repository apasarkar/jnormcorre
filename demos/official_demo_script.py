import datetime
import math
import os
import sys

import numpy as np
import tifffile

from jnormcorre import motion_correction
from jnormcorre.utils import registrationarrays
from typing import *

from jnormcorre.utils.lazy_array import lazy_data_loader


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()


def motion_correct_pipeline(lazy_dataset: lazy_data_loader,
                            outdir: str,
                            max_shifts: Tuple[int, int],
                            frames_per_split: int = 1000,
                            num_splits_to_process_rig: int = 5,
                            niter_rig: int = 4,
                            pw_rigid: bool = False,
                            strides: Tuple[int, int] = (30, 30),
                            overlaps: Tuple[int, int] = (10, 10),
                            max_deviation_rigid: int = 3,
                            num_splits_to_process_els: int = 5,
                            gSig_filt: tuple[float, float] = None,
                            save_movie: bool = True,
                            min_mov: Optional[float] = None,
                            template: Optional[np.ndarray] = None):
    """
    Runs the full motion correction pipeline (with the option to do rigid and piecewise rigid registration after)
    See documentation for parameter details
    """
    corrector = motion_correction.MotionCorrect(lazy_dataset, max_shifts=max_shifts, frames_per_split=frames_per_split,
                                                num_splits_to_process_rig=num_splits_to_process_rig,
                                                niter_rig=niter_rig, pw_rigid=pw_rigid, strides=strides,
                                                overlaps=overlaps, max_deviation_rigid=max_deviation_rigid,
                                                num_splits_to_process_els=num_splits_to_process_els, min_mov=min_mov,
                                                gSig_filt=gSig_filt)

    # Run MC, Always Saving Non-Final Outputs For Use In Next Iteration
    frame_corrector_obj, target_file = corrector.motion_correct(
        template=template, save_movie=save_movie
    )

    display("Motion correction completed.")

    # Save Frame-wise Shifts
    display(f"Saving computed shifts to ({outdir})...")
    np.savez(os.path.join(outdir, "shifts.npz"),
             shifts_rig=corrector.shifts_rig,
             x_shifts_els=corrector.x_shifts_els if pw_rigid else None,
             y_shifts_els=corrector.y_shifts_els if pw_rigid else None)
    display('Shifts saved as "shifts.npz".')

    return frame_corrector_obj, target_file


def main():
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

    registration_obj, registered_filename = motion_correct_pipeline(lazy_dataset, ".", max_shifts,
                                                                    frames_per_split=1000,
                                                                    num_splits_to_process_rig=5,
                                                                    niter_rig=4,
                                                                    pw_rigid=pw_rigid,
                                                                    strides=strides,
                                                                    overlaps=overlaps,
                                                                    max_deviation_rigid=3,
                                                                    num_splits_to_process_els=5,
                                                                    gSig_filt=None,
                                                                    save_movie=True
                                                                    )

if __name__ == "__main__":
    main()
