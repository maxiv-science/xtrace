import os, json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
import hdf5plugin
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import math
import itertools
from cupy import meshgrid, linspace
import numpy as np
import cupy as cp
import xtrace
import skimage.morphology as morph
import cupyx.scipy.sparse as cusp
import time
from scipy.ndimage import zoom
from scipy.signal import convolve2d
import copy
import xtrace
import deconvolution

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