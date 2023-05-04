from cupy import array, linspace, meshgrid, reshape, concatenate, around
from cupy import zeros, vstack, tile, inner, floor, linalg, where, shape
import cupy as np
import cupy
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cusp
import cupyx
import cupyx.scipy.sparse.linalg as cslinalg
import time
from functools import reduce
import scipy

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
    ipsf = psf.copy()
    ipsf = psf.T
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        sm = ipsf@(pixel_arr/c)
        pixel_arr_itr = pixel_arr_itr*sm
    return pixel_arr_itr.reshape(img.shape).get()

def richard_lucy_inv_factorized(img, psf, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    psf_factorized = factorized(psf)
    ipsf = psf_factorized( cupy.eye(psf.shape[0], dtype=psf.dtype) ).T
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        sm = ipsf@(pixel_arr/c)
        pixel_arr_itr = pixel_arr_itr*sm
    return pixel_arr_itr.reshape(img.shape).get()

def richard_lucy_inv_simple(img, psf, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    ipsf = cusp.diags(psf.diagonal())
    ipsf.data = 1/ipsf.data
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        sm = ipsf@(pixel_arr/c)
        pixel_arr_itr = pixel_arr_itr*sm
    return pixel_arr_itr.reshape(img.shape).get()

def factorized(G):
    return cslinalg.factorized(G)

def splu(G):
    return scipy.sparse.linalg.splu(G.get(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=1e-4)

def nneg_inv_factorized(img, fact_G, nneg=True):
    inv_img = fact_G(cupy.array(img).flatten())
    if (nneg):
        inv_img[inv_img<0] = 0
    return inv_img.get().reshape(img.shape)

def nneg_inv_solve(img, inv_A, nneg=True):
    inv_img = inv_A.solve(img.flatten())
    if (nneg):
        inv_img[inv_img<0] = 0
    return inv_img.reshape(img.shape)

def eye(m, **kwargs):
    return cupyx.scipy.sparse.eye(m, **kwargs)

def H4neigh(shape, dtype=np.float32):
    m, n = shape[0], shape[1]
    H = cusp.coo_matrix((m*n,m*n), dtype=dtype)
    # dx (img)
    nrows = m*(n-1)
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx%(n-1) + (idx//(n-1))*n, -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx%(n-1) + (idx//(n-1))*n + 1, 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += B.transpose()@B
    # dy (img)
    nrows = (m-1)*n
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx, -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx + n, 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += B.transpose()@B
    return H

def H8neigh(shape, dtype=np.float32):
    m, n = shape[0], shape[1]
    H = cusp.coo_matrix((m*n,m*n), dtype=dtype)
    # dx (img)
    nrows = m*(n-1)
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx%(n-1) + (idx//(n-1))*n, -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx%(n-1) + (idx//(n-1))*n + 1, 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += B.transpose()@B
    # dy (img)
    nrows = (m-1)*n
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx, -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx + n, 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += B.transpose()@B
    # +dx+dy (img)
    # ----------------------------------------------
    nrows = (m-1)*(n-1)
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx + idx//(n-1), -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx + (n+1) + idx//(n-1), 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += 0.5*B.transpose()@B
    # -dx+dy (img)
    # ----------------------------------------------
    nrows = (m-1)*(n-1)
    idx = cupy.arange(nrows)
    row, col, data = cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=np.int32), cupy.zeros((2*nrows,), dtype=dtype)
    row[:nrows], col[:nrows], data[:nrows] = idx, idx + 1 + idx//(n-1), -1
    row[nrows:], col[nrows:], data[nrows:] = idx, idx + n + idx//(n-1), 1
    B = cusp.coo_matrix((data, (row, col)), shape=(nrows,m*n))
    H += 0.5*B.transpose()@B
    return H
