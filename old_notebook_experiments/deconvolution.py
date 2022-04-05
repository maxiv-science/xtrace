from cupy import array, linspace, meshgrid, reshape, concatenate, around
from cupy import zeros, vstack, tile, inner, floor, linalg, where, shape
import cupy as np
import cupy
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cusp
import cupyx
import time
from functools import reduce

def landweber(img, psf, w, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        pixel_arr_itr = pixel_arr_itr - w*psf.T@(psf@pixel_arr_itr - pixel_arr)
    return pixel_arr_itr.reshape(img.shape).get()

def regularized_richard_lucy(img, psf, smooth_coeff, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        grad = np.gradient(pixel_arr_itr.reshape(img.shape))
        divergence = np.gradient(grad[0], axis=0) + np.gradient(grad[1], axis=1)
        divergence_arr = divergence.reshape(-1)
        sm = psf.T@(pixel_arr/c)
        temp_1 = pixel_arr_itr*sm
        temp_2 = (1 - smooth_coeff*divergence_arr)
        pixel_arr_itr = temp_1/temp_2
    return pixel_arr_itr.reshape(img.shape).get()

def richard_lucy(img, psf, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        sm = psf.T@(pixel_arr/c)
        pixel_arr_itr = pixel_arr_itr*sm
    return pixel_arr_itr.reshape(img.shape).get()