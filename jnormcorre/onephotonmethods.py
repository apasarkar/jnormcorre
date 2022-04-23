'''
Functionality to deal with background (neuropil) contamination 

'''

import numpy as np
import jax
from jax import jit, vmap
from functools import partial 
import cv2


def get_kernel(gSig_filt):
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return ker2D

def high_pass_filter_cv(kernel, img_orig):
    if img_orig.ndim == 2:  # image
        return cv2.filter2D(np.array(img_orig, dtype=np.float32),
                            -1, kernel, borderType=cv2.BORDER_REFLECT)
    else:  # movie
        return jnormcorre.utils.movies.movie(np.array([cv2.filter2D(np.array(img, dtype=np.float32),
                            -1, kernel, borderType=cv2.BORDER_REFLECT) for img in img_orig]))     

def high_pass_batch(kernel, imgs):
    return np.array([cv2.filter2D(np.array(img, dtype=np.float32),
                            -1, kernel, borderType=cv2.BORDER_REFLECT) for img in imgs])


