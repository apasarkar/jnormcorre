"""
Functionality to deal with background (neuropil) contamination 

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import cv2
from typing import *


def get_kernel(gSig_filt: list[int]):
    if not isinstance(gSig_filt[0], int):
        raise TypeError(
            "gSig_filt is a list which must contain an integer,"
            " instead it contained{}".format(type(gSig_filt[0]))
        )
    if gSig_filt[0] < 1:
        raise ValueError("gSig_filt is a list which must contain a positive integer")
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return ker2D


def _high_pass_filter_img(img: np.ndarray, kernel: np.ndarray):
    """
    Filter img with kernel
    Args:
        img (np.ndarray): Shape (fov dim 1, fov dim 2). Image to be filtered
        kernel (np.ndarray): Shape (k1, k1). Kernel for high pass filtering
    Returns:
        filtered_img (np.ndarray): Shape (fov dim 1, fov dim 2).
    """
    img_padded = jnp.pad(img,
                         (((kernel.shape[0]) // 2, (kernel.shape[0]) // 2),
                          ((kernel.shape[1]) // 2, (kernel.shape[1]) // 2)),
                         mode='reflect')
    filtered_frame = jax.scipy.signal.convolve(img_padded, kernel, mode="valid")
    return filtered_frame

high_pass_filter_img = jit(_high_pass_filter_img)
high_pass_batch = jit(vmap(_high_pass_filter_img, in_axes=(0, None)))
