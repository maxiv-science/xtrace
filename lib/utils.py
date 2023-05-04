import os, json
import numpy as np
#import hdf5plugin
import bitshuffle
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import math
import itertools
from cupy import meshgrid, linspace
import numpy as np
import cupy as cp
import xtrace
import skimage.morphology as morph
import cupyx.scipy.sparse as cusp
import scipy.sparse
import time
from scipy.ndimage import zoom
from scipy.signal import convolve2d
import copy
import xtrace
import deconvolution

def nvirpix(config):
    nmods_0 = config["detector"]["nmods"][0]
    nmods_1 = config["detector"]["nmods"][1]
    npixmod_0 = config["detector"]["npixmod"][0]
    npixmod_1 = config["detector"]["npixmod"][1]
    #It is supposed to be a squared matrix, any index should work
    nvirpixgap = np.shape(config["detector"]["el_corr"])[0]
    #I include the pixels that form the horizontal and vertical gaps
    return np.array([(nmods_0 * npixmod_0) + ((nmods_0 - 1) * nvirpixgap), (nmods_1 * npixmod_1) + ((nmods_1 - 1) * nvirpixgap)])

def mask(data, removehotspots=False):
    #Removing raw parts of image: 
    #-1 -> detector gaps, -2 -> hot pixels + extreme values
    mask = np.logical_or(data==np.uint32(-1),data==np.uint32(-2))
    img = data.copy()
    if removehotspots:
        peaks = img > 100_000
        peaks = morph.binary_dilation(peaks)
        mask = np.logical_or(mask, peaks)
    img[mask] = 0
    mask = ~mask
    return img, mask

def deconvolve(config, img, samp_density):
    r_grid = ray_grid(config["dimensions"], samp_density)
    G = xtrace.depth_spill_psf(config, *r_grid)
    recovered_img = deconvolution.richard_lucy(img, G, 10)
    return recovered_img, G

def load_stack(
    data_filepath, 
    poni_filepath,
    detector={"pixel_dims": np.array([75.0, 75.0, 450.0]), "mu":  3.445930 * 10**-3},
    h5_path="/entry/data/data",
    ):
    with h5py.File(data_filepath,'r') as h5f:
        data = h5f[h5_path]
        data = np.squeeze(data) 
    poni = load_poni(poni_filepath)
    config = {
        "detector": detector,
        #Detector dimentions
        "dimensions": np.array(data[0].shape),
        "ray_origin": np.array([poni[v]*1e6 for v in ["Poni1", "Poni2", "Distance"]]),
        "wavelength": poni["Wavelength"],
        "ray_rotations": np.array([poni[v] for v in ['Rot1', 'Rot2', 'Rot3']]),
    }
    return config, data

def load_poni(filepath):
        with open(filepath, 'r') as f:
            poni_dict = {}
            for l in f:
                if '#' not in l:
                    try:
                        k, v = l.split(":")
                        poni_dict[k] = float(v)
                    except:
                        pass#print(f"didn't parse {l}")
            return poni_dict

def ray_grid(dimensions, samp_density=1):
    eps = 1/samp_density/2
    hit_row, hit_col = meshgrid(
        linspace(eps, dimensions[0] - eps, samp_density*dimensions[0]),
        linspace(eps, dimensions[1] - eps, samp_density*dimensions[1])
    )
    hit_row = hit_row.flatten()
    hit_col = hit_col.flatten()
    return hit_row, hit_col

def random_ray_grid(dimensions):
    hit_row, hit_col = ray_grid(dimensions)
    hit_row += cp.random.random(hit_row.shape) - 0.5
    hit_col += cp.random.random(hit_col.shape) - 0.5
    return hit_row, hit_col

#redo these
def local_transform(config, img, area):
    px, py, pz = config["detector"]["pixel_dims"]
    ylim, xlim = area
    xdiff = px*xlim.start
    ydiff = py*ylim.start
    img_area = img[area]
    config_cp = copy.deepcopy(config)
    config_cp["ray_origin"][:2] -= np.array(xdiff, ydiff)*1e-6
    config_cp["dimensions"] = np.array(img_area.shape)
    return config_cp, img_area

def upsample_transform(config, img, z=2, order=0):
    u_img = zoom(img, z, order=order)
    config_cp = copy.deepcopy(config)
    config_cp["detector"]["pixel_dims"][:2] /=z
    config_cp["dimensions"] *= z
    return config_cp, u_img
    
def downsample_img(img, z=2):
    kernel = np.ones((z, z))/(z*z)
    convolved = convolve2d(img, kernel, mode='valid')
    return convolved[::z,::z]

#Use this like this (pyFAI library)
#integrator = utils.azimutal_integrator(config)
#integrated_img = integrator.integrate2d(img, az_npt, az_npt, radial_range=radial)[0]
#integrated_recovered = integrator.integrate2d(recovered_img, az_npt, az_npt, radial_range=radial)[0]
def azimutal_integrator(config):
    px, py, _ = config["detector"]["pixel_dims"]
    poni1, poni2, dist = config["ray_origin"]*1e-6
    rot1, rot2, rot3 = config["ray_rotations"]
    return AzimuthalIntegrator(
        pixel1=px*1e-6, 
        pixel2=py*1e-6,
        dist=dist,poni1=poni1, poni2=poni2,
        rot1=rot1, rot2=rot2, rot3=rot3,
        wavelength=config['wavelength']
    )

def pad(a, pad_rows, pad_cols = None, pad_val = 0):
    if pad_cols is None:
        pad_cols = pad_rows

    sh_ = a.shape
    sh = (int(np.prod(sh_[:-2])),)+sh_[-2:] # flatten image stack shape

    a_pad = np.full(sh[:1]+(sh[1]+2*pad_rows,sh[2]+2*pad_cols), pad_val, dtype=a.dtype)
    a_pad[:,pad_rows:(pad_rows+sh[1]),pad_cols:(pad_cols+sh[2])] = a.reshape(sh)
    a_pad = a_pad.reshape(sh_[:-2]+a_pad.shape[-2:])

    return a_pad

def pad_sparse(G, pad_rows, pad_cols = None, pad_val = None):
    if pad_cols is None:
        pad_cols = pad_rows

    sh_ = G.shape
    sh = (sh_[0]+2*pad_rows, sh_[1]+2*pad_cols)

    rows = np.empty_like(G.indices)
    scipy.sparse._sparsetools.expandptr(G.shape[0], G.indptr, rows)
    cols = G.indices

    if pad_val is None or pad_val==0:
        rows += pad_rows
        cols += pad_cols
        G_pad = scipy.sparse.csr_matrix((G.data,(rows,cols)),shape=sh)
    else:
        rows += pad_rows
        cols += pad_cols
        _cols, _rows = np.meshgrid(np.arange(sh[1]),np.arange(pad_rows))
        _data = np.full(_cols.shape, pad_val, dtype=G.dtype)
        rows = np.concatenate((rows,_rows.flatten()))
        cols = np.concatenate((cols,_cols.flatten()))
        data = np.concatenate((G.data,_data.flatten()))
        _cols, _rows = np.meshgrid(np.arange(sh[1]),np.arange(sh_[0]+pad_rows,sh_[0]+2*pad_rows))
        rows = np.concatenate((rows,_rows.flatten()))
        cols = np.concatenate((cols,_cols.flatten()))
        data = np.concatenate((data,_data.flatten()))
        _cols, _rows = np.meshgrid(np.arange(pad_cols),np.arange(pad_rows,sh_[0]+pad_rows))
        _data = np.full(_cols.shape, pad_val, dtype=G.dtype)
        rows = np.concatenate((rows,_rows.flatten()))
        cols = np.concatenate((cols,_cols.flatten()))
        data = np.concatenate((data,_data.flatten()))
        _cols, _rows = np.meshgrid(np.arange(sh_[1]+pad_cols,sh_[1]+2*pad_cols),np.arange(pad_rows,sh_[0]+pad_rows))
        rows = np.concatenate((rows,_rows.flatten()))
        cols = np.concatenate((cols,_cols.flatten()))
        data = np.concatenate((data,_data.flatten()))
        G_pad = scipy.sparse.csr_matrix((data,(rows,cols)),shape=sh)

    return G_pad


def pad_psf(G, img_shape, pad_rows, pad_cols = None, pad_val = None):
    if pad_cols is None:
        pad_cols = pad_rows

    rows = np.empty_like(G.indices)
    scipy.sparse._sparsetools.expandptr(G.shape[0], G.indptr, rows)
    cols = G.indices

    # convert to image indices
    rows_i, rows_j = np.unravel_index(rows, img_shape)
    cols_i, cols_j = np.unravel_index(cols, img_shape)
    # padding
    rows_i += pad_rows
    rows_j += pad_cols
    cols_i += pad_rows
    cols_j += pad_cols
    # convert to flatten indices
    rows = rows_i * (img_shape[1]+2*pad_cols) + rows_j
    cols = cols_i * (img_shape[1]+2*pad_cols) + cols_j

    # new shape
    sh = (img_shape[0]+2*pad_rows)*(img_shape[1]+2*pad_cols)
    sh = (sh,sh)

    if pad_val is None or pad_val==0:
        return scipy.sparse.csr_matrix((G.data,(rows,cols)),shape=sh)
    else:
        _cols, _rows = np.meshgrid(np.arange(img_shape[1]+2*pad_cols),np.arange(pad_rows))
        _data = np.full(_cols.shape, pad_val, dtype=G.dtype)
        rowsp = _rows.ravel()
        colsp = _cols.ravel()
        datap = _data.ravel()
        _cols, _rows = np.meshgrid(np.arange(img_shape[1]+2*pad_cols),np.arange(img_shape[0]+pad_rows,img_shape[0]+2*pad_rows))
        rowsp = np.concatenate((rowsp,_rows.ravel()))
        colsp = np.concatenate((colsp,_cols.ravel()))
        datap = np.concatenate((datap,_data.ravel()))
        _cols, _rows = np.meshgrid(np.arange(pad_cols),np.arange(pad_rows,img_shape[0]+pad_rows))
        _data = np.full(_cols.shape, pad_val, dtype=G.dtype)
        rowsp = np.concatenate((rowsp,_rows.ravel()))
        colsp = np.concatenate((colsp,_cols.ravel()))
        datap = np.concatenate((datap,_data.ravel()))
        _cols, _rows = np.meshgrid(np.arange(img_shape[1]+pad_cols,img_shape[1]+2*pad_cols),np.arange(pad_rows,img_shape[0]+pad_rows))
        rowsp = np.concatenate((rowsp,_rows.ravel()))
        colsp = np.concatenate((colsp,_cols.ravel()))
        datap = np.concatenate((datap,_data.ravel()))
        idx = rowsp * (img_shape[1]+2*pad_cols) + colsp
        rows = np.concatenate((rows,idx.ravel()))
        cols = np.concatenate((cols,idx.ravel()))
        data = np.concatenate((G.data,datap.ravel()))
        return scipy.sparse.csr_matrix((data,(rows,cols)),shape=sh)

def csr_to_cupy(G):
    rows = np.empty_like(G.indices)
    scipy.sparse._sparsetools.expandptr(G.shape[0], G.indptr, rows)
    cols = G.indices
    return cusp.csr_matrix((cp.array(G.data),(cp.array(rows),cp.array(cols))),shape=G.shape)
