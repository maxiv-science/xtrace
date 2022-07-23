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
#  /    /   px /      /      /      /      / ny (count)
# o pz /------/------/------/------/------/
#  \  <---------o   /      /      /      /
#   \/---sx-/--/|--/------/------/------/
#   /v     /sy/ | /      /      /      /
#  <------/--v--|/------/------/------o
#               |    nx (count)
#             d |
#               |
#               v
#           RAY ORIGIN
#
#

def depth_spill_psf(config, xp, yp, energies=None, G=None):
    nx, ny = config["dimensions"]
    sx, sy, d = config["ray_origin"]
    px, py, pz = config["detector"]["pixel_dims"]
    mu = config["detector"]["mu"]
    
    raycount = len(xp)
    npix = nx*ny
    
    if G is None:
        G = cusp.coo_matrix((npix,npix))
    if energies is None:
        energies = np.ones(raycount)*npix/raycount
        
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
    last_dists = np.zeros(raycount)
    
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

def depth_offsets(config, xp, yp):
    nx, ny = config["dimensions"]
    sx, sy, d = config["ray_origin"]
    px, py, pz = config["detector"]["pixel_dims"]
    mu = config["detector"]["mu"]
    
    raycount = len(xp)
    npix = nx*ny
    
    origin = np.array([sx, sy])[:,np.newaxis]
    pixel_dims = np.array([px, py])[:,np.newaxis]
    bounds = np.array([nx, ny])[:,np.newaxis]
    
    hitpoints_pspace = np.stack((xp, yp))
    original_cells = np.floor(hitpoints_pspace).astype(int)
    rays = hitpoints_pspace - origin/pixel_dims
    offsets = rays/d*pz
    
    offsets /= npix/raycount
    
    xoffsets = np.zeros((nx, ny))
    xoffsets[original_cells[0],original_cells[1]] += offsets[0]
    yoffsets = np.zeros((nx, ny))
    yoffsets[original_cells[0],original_cells[1]] += offsets[1]
    
    return xoffsets.get(), yoffsets.get()