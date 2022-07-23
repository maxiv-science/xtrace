# coding: utf-8

# # Removing Effect of Sensor Depth Spill
# This code is meant to illustrate the current state of trying to remove the effect of sensor depth spill. 
# Authors: Zdenek Matej and Samuel Selleck

import os, json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
import hdf5plugin
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import maxpy_erda as mp
import math
from fit_asym_voigt import fit
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

def plot_compare_images(img1, img2, area=(slice(550,650), slice(100, 200)), title1="First", title2="Second"):
    perc = np.percentile(img1, 99.8)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].set_title(title1)
    axs[0, 0].imshow(img1, vmax=perc)
    axs[0, 1].set_title(title2)
    axs[0, 1].imshow(img2, vmax=perc)
    axs[1, 0].set_title(f"{title1} Zoom")
    axs[1, 0].imshow(img1[area], vmax=perc)
    axs[1, 1].set_title(f"{title2} Zoom")
    axs[1, 1].imshow(img2[area], vmax=perc)
    plt.show()

    #plt.figure(figsize=(20, 20))
    #plt.imshow(img1, vmax=perc)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(img2, vmax=perc)

    
def azimutal_fit(config, icsdfilepath):
    pl, ph, pw = config["detector"]["pixel_dims"]
    # ## Azimuthal Integration
    # Comparison between raw and reconstructed
    #add options
    # we actually see 6 LaB6 lines and so we take only them
    # note: some lines still may be there multiple times (in case they overlap)
    nhkl = 11
    icsd_tbl = np.loadtxt(icsdfilepath, skiprows=1)
    hkl = icsd_tbl[:,0:3].astype(np.int32)
    alat = 4.156468 # Å (tabulated)
    # interplanar distances
    d_hkl = alat/np.sqrt((hkl**2).sum(axis=1))
    # peak positions in 2Theta (Bragg's law)
    arg = config['wavelength']*1e10/2./d_hkl
    lidx = arg<=1. # some reflections may be unreachable for our wavelength
    d_hkl = d_hkl[lidx]
    arg = arg[lidx]
    
    tth_hkl = 2.*np.arcsin(config['wavelength']*1e10/2./d_hkl)
    # we actually see 6 LaB6 lines and so we take only them
    # note: some lines still may be there multiple times (in case they overlap)
    hkl = hkl[:nhkl,:]
    d_hkl = d_hkl[:nhkl]
    tth_hkl = tth_hkl[:nhkl]
    hkl = hkl[:nhkl,:]
    d_hkl = d_hkl[:nhkl]
    tth_hkl = tth_hkl[:nhkl]

    deg2rad = np.pi/180.
    rad2deg = 180./np.pi
    poni1, poni2, dist = config["ray_origin"]*1e-6
    rot1, rot2, rot3 = config["ray_rotations"]
    ai = AzimuthalIntegrator(
        pixel1=ph*1e-6, 
        pixel2=pl*1e-6,
        dist=dist,poni1=poni1, poni2=poni2,
        rot1=rot1, rot2=rot2, rot3=rot3,
        wavelength=config['wavelength']
    )

    # convert detector parameters from Carmen's (pyFAI-like) notation to bli711 (Fit2D-like) notation
    pf2d = ai.getFit2D()
    
    #help(mp)
    det_params_b711 = {
        'n0': pf2d['centerY'],
        'm0': pf2d['centerX'],
        'wn': (pf2d['pixelY']*1e-3/pf2d['directDist']),
        'wm': (pf2d['pixelX']*1e-3/pf2d['directDist']),
        'phi': 0*np.pi/2,
        'n': config["dimensions"][0], 'm': config["dimensions"][1],
        'rot': pf2d['tiltPlanRotation']*deg2rad ,
        'tilt': pf2d['tilt']*deg2rad
    }
    return ai, det_params_b711, tth_hkl

# def plot_fit(img, poni_dict, det_params_b711, tth_hkl, pixel_dimensions):
#     pl, pw, ph = pixel_dimensions
#     deg2rad = np.pi/180.
#     rad2deg = 180./np.pi
    
#     plt.figure(figsize=(8,8))
#     plt.imshow(img, vmax=np.percentile(img,98))
#     # plot a cross at a beam center
#     plt.plot(poni_dict['Poni2']*1e6/pl,poni_dict['Poni1']*1e6/ph,'rx')

#     # mark pixels in the image that are on diffraction lines
#     # note: there is no function 2theta to pixel coords in maxpy_erda as it was not needed for integration

#     N,M = np.meshgrid(np.linspace(0, det_params_b711['n'], 200), np.linspace(0, det_params_b711['m'], 200))
#     tth = mp.tth2DwithTilt(0.0,N,M,det_params_b711)

#     dtth = 0.1 * deg2rad
#     for line in tth_hkl:
#         lidx = np.logical_and(tth>=(line-dtth), tth<=(line+dtth))
#         plt.plot(M[lidx],N[lidx],'r.',ms=2)

#     plt.title('image with beam center and line labels');


# def plot_compare_peaks(az_before, az_after, deg_xtth, tth_hkl): #add options
#     peak_width = 2
#     peaks = { str(n):(v*60 - peak_width/2, v*60 + peak_width/2)
#              for n, v in enumerate(tth_hkl) if n <= 5}
#     peak_slices = {name:np.logical_and(b1 < deg_xtth, deg_xtth < b2) 
#                    for name, (b1, b2) in peaks.items()}

#     # show integrated data
#     plt.figure(figsize=(14,3))
#     plt.plot(deg_xtth, az_after, "r:x")
#     plt.plot(deg_xtth, az_before, "b-*")

#     fig, axs = plt.subplots(2, 3, figsize=(14,10))

#     for i, (name, slc) in enumerate(peak_slices.items()):
#         j, k = (math.floor(i/3), i%3)
#         axs[j,k].plot(deg_xtth[slc], az_after[slc], "r:x")
#         axs[j,k].plot(deg_xtth[slc], az_before[slc], "b-*")
#         axs[j,k].set_title(name)


# def plot_compare_peak_fit(az_before, az_after, deg_xtth, tth_hkl): #add options
#     peak_width = 2
#     peaks = { str(n):(v*60 - peak_width/2, v*60 + peak_width/2)
#              for n, v in enumerate(tth_hkl) if n <= 5}
#     peak_slices = {name:np.logical_and(b1 < deg_xtth, deg_xtth < b2) 
#                    for name, (b1, b2) in peaks.items()}
    
#     deg2rad = np.pi/180.
#     rad2deg = 180./np.pi
    
#     xtth = deg_xtth*deg2rad
#     stages = {"original": az_before, "recovered": az_after}
#     fig, axs = plt.subplots(2, 6, figsize=(20,6))
#     for p, (peak_name, mask) in enumerate(peak_slices.items()):
#         for i, (src_name, az_src) in enumerate(stages.items()):
#             peak_fit, p1 = fit(xtth[mask], az_src[mask])
#             axs[i,p].plot(deg_xtth[mask], az_src[mask], 'b:.')
#             axs[i,p].plot(deg_xtth[mask], peak_fit, 'r')
#             axs[i,p].set_title(f'{src_name} - {peak_name}')
#             axs[i,p].annotate(f'intensity:\n{p1[0]:.5f}\npos:\n{p1[1]*rad2deg:.3f}\nfwhm:\n'              f'{p1[2]*rad2deg:.5f}\nshape:\n{p1[3]:.5f}\nasym:\n{p1[4]:.5f}',
#                     xy=(0.05, 0.2),
#                     xycoords='axes fraction')
