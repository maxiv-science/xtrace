#from numpy import array, linspace, meshgrid, reshape, concatenate, around
#from numpy import zeros, vstack, tile, inner, floor, linalg, where, shape
#import numpy as np
from cupy import array, linspace, meshgrid, reshape, concatenate, around
from cupy import zeros, vstack, tile, inner, floor, linalg, where, shape
import cupy as np
import cupy
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cusp
import cupyx
import time
from functools import reduce

#                                                ^
#       .    /------/------/------/------/------/
#      /    /      /      /      /      /      /
#     /    ^------/------/------/------/------/
#    / py /      /      /      /      /      /
#   /    o------>------/------/------/------/
#  /|   /   px /      /      /      /      / ny (count)
# . |  /------/------/------/------/------/
# pz| <---------o   /      /      /      /
#   |/---sx-/--/|--/------/------/------/
#   /      /sy/ | /      /      /      /
#  <------/--v--|/------/------/------o
#               |    nx (count)
#             d |
#               |
#               v
#               @
#           RAY ORIGIN
#
#

def get_config(poni, detector_info):
    return {
        
    }
def depth_spill_psf(params, xp, yp, G=None):
    px = params["pl"]
    py = params["ph"]
    pz = params["pw"]
    sx = params["sz"]
    sy = params["sy"]
    d = -params["sx"]
    mu = params["mu"]
    nx, ny = params["det_numpixels"]
    
    raycount = len(xp)
    npix = nx*ny
    
    if G is None:
        G = cusp.coo_matrix((npix,npix))
    
    origin = np.array([sx, sy])[:,np.newaxis]
    pixel_dims = np.array([px, py])[:,np.newaxis]
    bounds = np.array([nx, ny])[:,np.newaxis]
    
    hitpoints_pspace = np.stack((xp, yp))
    hitpoints = hitpoints_pspace*pixel_dims
    rays = hitpoints - origin
    ray_lengths = np.sqrt(rays[0]**2 + rays[1]**2 + d**2)
    rays /= ray_lengths
    quadrant_shift = hitpoints > origin
    plane_dirs = 2*quadrant_shift - 1
    original_cells = np.floor(hitpoints_pspace) 
    back_dists = pz/d*ray_lengths
    
    current_cells = original_cells.copy()
    energies = np.ones(raycount)*npix/raycount
    last_dists = np.ones(raycount)
    
    while True:
        keep = ~np.logical_or(np.logical_or(
            np.any(current_cells >= bounds, axis=0),
            np.any(current_cells < 0, axis=0)),
            last_dists == back_dists
        )
        
        if not keep.any():
            break
        
        current_cells = current_cells[:, keep]
        original_cells = original_cells[:, keep]
        plane_dirs = plane_dirs[:,keep]
        hitpoints = hitpoints[:,keep]
        rays = rays[:,keep]
        quadrant_shift = quadrant_shift[:,keep]
        energies = energies[keep]
        ray_lengths = ray_lengths[keep]
        last_dists = last_dists[keep]
        back_dists = back_dists[keep]
        
        candidate_dists = ((current_cells + quadrant_shift)*pixel_dims - hitpoints)/rays
        candidate_dists[~np.isfinite(candidate_dists)] = np.inf
        closest_ind = (np.argmin(candidate_dists, axis=0), np.arange(candidate_dists.shape[1]))
        dists = np.minimum(candidate_dists[closest_ind], back_dists)
        d_dists = dists - last_dists
        absorbed = energies*(1 - np.exp(-mu*d_dists))
        G += cusp.coo_matrix((
            absorbed,(
            current_cells[0]*ny + current_cells[1],
            original_cells[0]*ny + original_cells[1])),
            shape=G.shape
        )
        last_dists = dists
        energies -= absorbed
        current_cells[closest_ind] += plane_dirs[closest_ind]
    
    return G


def landweber_deconvolution(img, psf, w, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        pixel_arr_itr = pixel_arr_itr - w*psf.T@(psf@pixel_arr_itr - pixel_arr)
    return pixel_arr_itr.reshape(img.shape).get()

def regularized_richard_lucy_deconv(img, psf, smooth_coeff, iters):
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
    return pixel_arr_itr.reshape(img.shape)

def richard_lucy_deconv(img, psf, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        sm = psf.T@(pixel_arr/c)
        pixel_arr_itr = pixel_arr_itr*sm
    return pixel_arr_itr.reshape(img.shape)