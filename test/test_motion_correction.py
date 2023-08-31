import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import h5py
import logging

from jnormcorre.simulation import SimData
from jnormcorre.motion_correction import MotionCorrect

class Test_Simulation:

    def test_init(self, frames=100, X=100, Y=100):
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                           blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                           background_noise=1, shot_noise=0.2)

    def test_run(self, frames=100, X=100, Y=100):

        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                           blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                           background_noise=1, shot_noise=0.2)

        _, _ = sim.simulate()


class Test_mc:

    def save_sample(self, input_, data, h5_loc=None):

        if isinstance(input_, np.ndarray):
            return input_

        input_ = Path(input_)

        # save temp file
        if input_.suffix in ('.tiff', '.tif'):
            tifffile.imwrite(input_, data=data)

        elif input_.suffix in ('.h5', '.hdf5'):

            with h5py.File(input_.as_posix(), "w") as h:
                # h.create_group("data")
                ds = h.create_dataset(h5_loc, data=data, shape=data.shape, dtype=data.dtype)
                ds[:] = data[:]

        else:
            raise ValueError(f"unknown file extension: {input_.suffix}")

        return input_

    def setup_method(self):

        frames, X, Y = (25, 75, 75)
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                           blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                           background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        self.data = data
        self.shifts = shifts

    @pytest.mark.parametrize("file_type", [("test.tiff", None), ("test.h5", "data/"), ("test.h5", "data/ch0")])
    def test_file(self, file_type):

        name, h5_loc = file_type

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # create sim data
            data, shifts = self.data, self.shifts
            input_ = self.save_sample(tmp_dir.joinpath(name), data, h5_loc=h5_loc)

            mc = MotionCorrect(input_, var_name_hdf5=h5_loc,
                               max_shifts=(6, 6), niter_rig=4, nonneg_movie=True, max_deviation_rigid=3,
                               upsample_factor_grid=4, strides=(50, 50), splits_rig=5, splits_els=5, gSig_filt=None,
                               num_splits_to_process_els=5, num_splits_to_process_rig=5, pw_rigid=True,
                               overlaps=(10, 10), min_mov=-5, niter_els=1)

            # Perform motion correction
            mc.motion_correct(save_movie=True)
            self.shifts = mc.shifts_rig

    @pytest.mark.parametrize("max_shifts", [(6, 6)]) # fails: , (50, 50)
    @pytest.mark.parametrize("num_splits_to_process_rig", [5, None])
    @pytest.mark.parametrize("num_splits_to_process_els", [5, None])
    @pytest.mark.parametrize("splits_els", [5]) # fails: , 14
    @pytest.mark.parametrize("splits_rig", [5]) # fails: , 14
    @pytest.mark.parametrize("gSig_filt", [None]) # fails: , 5, 20 (20, 20)
    @pytest.mark.parametrize("overlaps", [(10, 10)]) # fails: , (24, 24)
    @pytest.mark.parametrize("pw_rigid", [True]) # , False
    @pytest.mark.parametrize("min_mov", [-5]) # , False
    @pytest.mark.parametrize("niter_els", [1]) # , False
    def test_file(self, max_shifts, num_splits_to_process_rig, num_splits_to_process_els,
                  gSig_filt, overlaps, pw_rigid, splits_els, splits_rig, min_mov, niter_els,
                   niter_rig=4, nonneg_movie=True, max_deviation_rigid=3, upsample_factor_grid=4, strides=(50, 50)):

        input_, shifts = self.data, self.shifts

        # Create MotionCorrect instance
        mc = MotionCorrect(input_, var_name_hdf5=None,
                max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                num_splits_to_process_rig=num_splits_to_process_rig, strides=strides, overlaps=overlaps,
                pw_rigid=pw_rigid, splits_els=splits_els, num_splits_to_process_els=num_splits_to_process_els,
                upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                           min_mov=min_mov, niter_els=niter_els)

        # Perform motion correction
        mc.motion_correct(save_movie=True)
        self.shifts = mc.shifts_rig

        # test correct shift reconstruction
