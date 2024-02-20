import jnormcorre.motion_correction
from jnormcorre.utils.lazy_array import lazy_data_loader
import tifffile
import numpy as np
import h5py
from typing import *


class TiffArray(lazy_data_loader):

    def __init__(self, filename):
        """
        TiffArray data loading object. Supports loading data from multipage tiff files.

        Args:
            filename (str): Path to file

        """
        self.filename = filename

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
            x, y = page.shape
        return num_frames, x, y

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        if isinstance(indices, int):
            data = tifffile.imread(self.filename, key=[indices]).squeeze()
        else:
            indices_list = list(range(indices.start or 0, indices.stop or self.shape[0], indices.step or 1))
            data = tifffile.imread(self.filename, key=indices_list).squeeze()
        return data.astype(self.dtype)


class Hdf5Array(lazy_data_loader):

    def __init__(self, filename: str, field: str) -> None:
        """
        Generic lazy loader for Hdf5 files video files, where data is stored as (T, x, y). T is number of frames,
        x and y are the field of view dimensions (height and width).

        Args:
            filename (str): Path to filename
            field (str): Field of hdf5 file containing data
        """
        if not isinstance(field, str):
            raise ValueError("Field must be a string")
        self.filename = filename
        self.field = field
        with h5py.File(self.filename, 'r') as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]

            # Get the shape of the array
            self._shape = field_dataset.shape

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        with h5py.File(self.filename, 'r') as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]
            if isinstance(indices, int):
                data = field_dataset[indices, :, :].squeeze()
            else:
                indices_list = list(range(indices.start or 0, indices.stop or self.shape[0], indices.step or 1))
                data = field_dataset[indices_list, :, :].squeeze()
        return data.astype(self.dtype)

class RegistrationArray(lazy_data_loader):
    def __init__(self, registration_obj: jnormcorre.motion_correction.frame_corrector,
                 data_loader: jnormcorre.utils.lazy_array.lazy_data_loader):
        """
        Class for registering 2D functional imaging data on the fly. Useful for visualization libraries etc.

        Args:
            registration_obj (jnormcorre.motion_correction.frame_corrector). Object which can perform registration
            data_loader (jnormcorre.utils.lazy_array.lazy_data_loader). Data loading object
        """
        self.data_loader = data_loader
        self.registration_obj = registration_obj
        # Verify that the data and registration info align properly
        dim1_match = data_loader.shape[1] == registration_obj.template.shape[0]
        dim2_match = data_loader.shape[2] == registration_obj.template.shape[1]
        error_msg = "Dimension mismatch: FOV dims of dataset {} FOV dims\
            of template {}".format(data_loader.shape[1:], registration_obj.template.shape)
        if not (dim1_match and dim2_match):
            raise ValueError(error_msg)

    @property
    def dtype(self):
        return self.data_loader.dtype

    @property
    def shape(self):
        return self.data_loader.shape

    @property
    def ndim(self):
        return self.data_loader.ndim

    def _compute_at_indices(self):
        pass

    def __getitem__(
            self,
            item: Union[int, Tuple[Union[int, slice, range]]]
    ):
        frames = self.data_loader[item]
        if len(frames.shape) == 2:  # This means we just loaded 1 frame
            frames = frames[None, :, :]

        return self.registration_obj.register_frames(frames).astype(self.dtype).squeeze()
