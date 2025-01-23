"""
Functionality to deal with background (neuropil) contamination 

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import cv2
from typing import *


def compute_highpass_filter_kernel(gaussian_sigma: list[int]):
    if not isinstance(gaussian_sigma[0], int):
        raise TypeError(
            "gSig_filt is a list which must contain an integer,"
            " instead it contained{}".format(type(gaussian_sigma[0]))
        )
    if gaussian_sigma[0] < 1:
        raise ValueError("gSig_filt is a list which must contain a positive integer")
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gaussian_sigma])
    ker = cv2.getGaussianKernel(ksize[0], gaussian_sigma[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return ker2D


def compute_gaussian_kernel(
    sigma_x: float, sigma_y: float, size: Optional[tuple] = None
):
    """
    Constructs a Gaussian kernel for image convolution.

    Args:
        sigma_x (float): Standard deviation of the Gaussian in the x dimension.
        sigma_y (float): Standard deviation of the Gaussian in the y dimension.
        size (tuple of int, optional): Dimensions of the kernel (height, width).
                                       If None, defaults to 6 * sigma_x and 6 * sigma_y.

    Returns:
        np.ndarray: 2D Gaussian kernel.
    """
    # Define the size of the kernel if not provided
    if size is None:
        size_x = int(np.ceil(6 * sigma_x))  # Covers ~99% of the Gaussian
        size_y = int(np.ceil(6 * sigma_y))
    else:
        size_x, size_y = size

    # Ensure the kernel size is odd
    size_x += size_x % 2 == 0
    size_y += size_y % 2 == 0

    # Create grid for the kernel
    x = np.linspace(-(size_x // 2), size_x // 2, size_x)
    y = np.linspace(-(size_y // 2), size_y // 2, size_y)
    x, y = np.meshgrid(x, y)

    # Compute the Gaussian function
    kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel


def _convolution_image_filter(img: np.ndarray, kernel: np.ndarray):
    """
    Filter img with kernel
    Args:
        img (np.ndarray): Shape (fov dim 1, fov dim 2). Image to be filtered
        kernel (np.ndarray): Shape (k1, k1). Kernel for filtering
    Returns:
        filtered_img (np.ndarray): Shape (fov dim 1, fov dim 2).
    """
    img_padded = jnp.pad(
        img,
        (
            ((kernel.shape[0]) // 2, (kernel.shape[0]) // 2),
            ((kernel.shape[1]) // 2, (kernel.shape[1]) // 2),
        ),
        mode="reflect",
    )
    filtered_frame = jax.scipy.signal.convolve(img_padded, kernel, mode="valid")
    return filtered_frame


convolution_image_filter = jit(_convolution_image_filter)
convolution_image_filter_batch = jit(vmap(_convolution_image_filter, in_axes=(0, None)))
