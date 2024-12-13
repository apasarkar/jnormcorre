import jnormcorre.motion_correction
from jnormcorre.utils.lazy_array import lazy_data_loader
import tifffile
import numpy as np
import h5py
from typing import *

class FilteredArray(lazy_data_loader):
    def __init__(self,
                 raw_data_loader: lazy_data_loader,
                 filter_function: Callable):
        """
        Class for loading and filtering data; this is broadly useful because we often want to spatially filter
        data to expose salient signals. We use this filtered version of the data to estimate shifts
        Args:
                raw_data_loader (lazy_data_loader): An object that supports the lazy_data_loader interface.
                    This can be for e.g. a custom object that reads data from disk, an array in RAM (like a numpy ndarray)
                    or anything else

                filter_function (Callable): A function that applies a spatial filter to every frame of a data array. It takes
                    an input movie of shape (frames, fov dim 1, fov dim 2) and returns a
                    filtered movie of the same shape. The type of the output is cast to numpy array in this function.
        """

        self._raw_data_loader = raw_data_loader
        self._filter = filter_function

    @property
    def raw_data_loader(self) -> lazy_data_loader:
        return self._raw_data_loader

    @property
    def filter_function(self) -> Callable:
        return self._filter

    @property
    def dtype(self) -> str:
        """
        data type
        """
        return self.raw_data_loader.dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self.raw_data_loader.shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)


    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Parameters
        ----------
        indices: Union[list, int, slice]
            the user's desired way of picking frames, either an int, list of ints, or slice
             i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        frames = self.raw_data_loader[indices]
        if frames.ndim == 2:
            frames = frames[None, :, :]
        return np.array(self.filter_function(frames))




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

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        if isinstance(indices, int):
            data = tifffile.imread(self.filename, key=[indices]).squeeze()
        elif isinstance(indices, list):
            data = tifffile.imread(self.filename, key=indices).squeeze()
        else:
            indices_list = list(
                range(
                    indices.start or 0, indices.stop or self.shape[0], indices.step or 1
                )
            )
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
        with h5py.File(self.filename, "r") as file:
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

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        with h5py.File(self.filename, "r") as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]
            if isinstance(indices, int):
                data = field_dataset[indices, :, :].squeeze()
            elif isinstance(indices, list):
                data = field_dataset[indices, :, :].squeeze()
            else:
                indices_list = list(
                    range(
                        indices.start or 0,
                        indices.stop or self.shape[0],
                        indices.step or 1,
                    )
                )
                data = field_dataset[indices_list, :, :].squeeze()
        return data.astype(self.dtype)


class RegistrationArray(lazy_data_loader):
    def __init__(
        self,
        registration_obj: jnormcorre.motion_correction.FrameCorrector,
        data_to_register: jnormcorre.utils.lazy_array.lazy_data_loader,
        pw_rigid=False,
        reference_data: Optional[jnormcorre.utils.lazy_array.lazy_data_loader] = None
    ):
        """
        Class for registering 2D functional imaging data on the fly. Useful for visualization libraries etc.

        Args:
            registration_obj (jnormcorre.motion_correction.FrameCorrector): Object which can perform registration
            data_to_register (jnormcorre.utils.lazy_array.lazy_data_loader): Data loading object
            pw_rigid (bool): Indicates whether we apply rigid or piecewise rigid registration to frames
            reference_data [Optional(jnormcorre.utils.lazy_array.lazy_data_loader)]: A reference stack. If provided, the algorithm
                will find optimal alignment between template and each frame of this stack. It will then apply these shifts to "data_to_register"
        """
        self.reference_data = reference_data
        self.data_loader = data_to_register
        if self.reference_data is not None:
            if not (self.reference_data.shape == self.data_loader.shape):
                raise ValueError(f"The data to register and the reference data stack do not have the same shape.")
        self.registration_obj = registration_obj
        self._pw_rigid = pw_rigid
        # Verify that the data and registration info align properly
        dim1_match = data_to_register.shape[1] == registration_obj.template.shape[0]
        dim2_match = data_to_register.shape[2] == registration_obj.template.shape[1]
        error_msg = "Dimension mismatch: FOV dims of dataset {} FOV dims\
            of template {}".format(
            data_to_register.shape[1:], registration_obj.template.shape
        )
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

    @property
    def batching(self):
        return self.registration_obj.batching

    @batching.setter
    def batching(self, new_batch: int):
        self.registration_obj.batching = new_batch

    @property
    def template(self) -> np.ndarray:
        """
        The template used for registration
        """
        return self.registration_obj.template

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        # Use data loader to load the frames
        frames = self.data_loader[indices, :, :]
        if len(frames.shape) == 2:  # This means we loaded 1 frame only
            frames = frames[None, :, :]

        # Register the data
        if self.reference_data is None:
            return self.registration_obj.register_frames(
                frames, pw_rigid=self._pw_rigid
            ).squeeze()
        else:
            reference_frames = self.reference_data[indices, :, :]
            if len(reference_frames.shape) == 2:
                reference_frames = reference_frames[None, :, :]
            return self.registration_obj.register_frames_and_transfer(frames, reference_frames)
