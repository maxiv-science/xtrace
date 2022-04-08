import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

import examplesetup
import lib.deconvolution as deconvolution
import lib.utils as utils
import lib.xtrace as xtrace

from numpy.random import default_rng
rng = default_rng()

import tensorflow as tf
from tensorflow import keras
import numpy as np

from scipy.signal import convolve2d


global_config = {
    'detector': {
        'pixel_dims': np.array([ 75.,  75., 450.]),
        'mu': 0.0034459300000000003
    },
    'dimensions': np.array([100, 100]),
    'ray_origin': np.array([24552.85736032, 86782.16334339, 80442.15320462]),
    'wavelength': 1.0332016536100021e-10,
    'ray_rotations': np.array([0., 0., 0.])
}
G_glob = xtrace.depth_spill_psf(global_config, *utils.ray_grid(global_config["dimensions"]))
      
    
class SyntheticDepthBlur(keras.utils.Sequence):

    #could include options to configure config params (or there ranges)
    def __init__(self, batch_size, batches, img_shape):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.batches = batches

    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        
        data = np.array([
            get_synthetic_data_pair()
            for i in range(self.batch_size)
        ])
        distorted_imgs = data[:,0, :, :, np.newaxis]
        images = data[:,1, :, :, np.newaxis]
        return (distorted_imgs, images)


def get_synthetic_data_pair(config=None, randgrid=True):
    if config is None:
        config = global_config
    #ray_gen_func = utils.random_ray_grid if randgrid else utils.ray_grid
    #ay_grid = ray_gen_func(config["dimensions"])
    #G = xtrace.depth_spill_psf(config, *ray_grid)
    img = random_image(config["dimensions"])
    distorted_img = apply_blur(img, G_glob, noise=True)
    return (distorted_img, img)

def apply_blur(img, G, noise):
    distorted_img = (G.get()@img.flatten()).reshape(img.shape)
    if noise:
        distorted_img += 0.00004*(rng.random()*0.5 + 1)*rng.poisson(100,img.shape)/100
    return distorted_img

def random_image(dimensions, density=0.08):
    img = np.zeros(dimensions)
    hits = rng.random(dimensions) <= density*rng.random()
    img[hits] = rng.exponential( size=np.count_nonzero(hits))#0.001*rng.zipf(1.7, size=np.count_nonzero(hits))
    s = 15
    c = np.floor(s/2)
    def eval(x, y):
        r_d_sq = ((x - c)**2 + (y - c)**2)*(20*(rng.random() + 0.5))**2
        r_d_sq[r_d_sq == 0] = 0.1
        res = 1/r_d_sq
        return res
    kernel = np.fromfunction(eval,(s, s))
    kernel -= kernel.min()
    kernel /= kernel.max()
    img = convolve2d(img, kernel, mode='same')
    return img

def random_depth_blur_psf(dimensions, randgrid=True):
    dim = np.max(dimensions) 
    config = { #RANDOM CONFIG
    "detector": {
            "pixel_dims": np.array([1.0, 1.0, 6]), #randomize this?
            "mu":  rng.random()
        },
    "dimensions": np.array(dimensions),
    "ray_origin": np.array([
            dim*rng.normal() + dimensions[0]/2,
            dim*rng.normal() + + dimensions[1]/2,
            dim*rng.exponential() + dim/3
        ]),
    }
    N = np.prod(dimensions)
    if randgrid:
        ray_grid = utils.random_ray_grid(dimensions)
    else:
        ray_grid = utils.ray_grid(dimensions)
    G = xtrace.depth_spill_psf(config, *ray_grid)
    return G
