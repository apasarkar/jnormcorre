from typing import *
from abc import ABC, abstractmethod
import numpy as np


class lazy_data_loader(ABC):
    '''
    This captures the numpy array-like functionality that all data loaders for motion correction need to contain

    Key: To implement support for a new file type, you just need to specify the key properties below (dtype, shape, ndim)
    and then implement the function _compute_at_indices.
    Adapted from mesmerize core: https://github.com/nel-lab/mesmerize-core/blob/master/mesmerize_core/arrays/_base.py
    '''

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def __getitem__(
            self,
            item: Union[int, Tuple[Union[int, slice, range]]]
    ):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of <{start}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")
            if stop is not None:
                if stop > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of <{stop}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Parameters
        ----------
        indices: Union[int, slice]
            the user's desired slice, i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        pass