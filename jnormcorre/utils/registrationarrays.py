from jnormcorre.utils.lazy_array import lazy_data_loader
import tifffile
import numpy as np
import h5py
from typing import *


class TiffArray(lazy_data_loader):

    def __init__(self, filename):
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
    def n_frames(self) -> int:
        """
        int
            number of frames
        """
        return self.shape[0]

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

    def __init__(self, filename, field):
        """
        Generic lazy loader for Hdf5 files
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
    def n_frames(self) -> int:
        """
        int
            number of frames
        """
        return self.shape[0]

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
