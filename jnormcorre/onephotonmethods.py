'''
Functionality to deal with background (neuropil) contamination 

'''

import numpy as np
import jax
from jax import jit, vmap
from functools import partial 
import cv2

@partial(jit)
def high_pass_filter_jax(full_kernel, img):
    orig_img = img + 0
    interm = jax.scipy.signal.convolve(img, full_kernel, mode='valid')
    return jax.image.resize(interm, orig_img.shape, method = "cubic")

def get_kernel(gSig_filt):
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return ker2D

