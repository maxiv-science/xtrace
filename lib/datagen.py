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

from scipy.signal import fftconvolve


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

class SyntheticDepthBlurPositional(keras.utils.Sequence):

    #could include options to configure config params (or there ranges)
    def __init__(self, batch_size, batches, img_shape, start_spread):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.batches = batches
        self.spread_mod = start_spread

    def on_epoch_end(self):
        if self.spread_mod < 1:
            self.spread_mod += 0.01
        
    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        
        train, truth = list(zip(*[
            get_synthetic_data_pair_positional(spread=self.spread_mod)
            for i in range(self.batch_size)
        ]))
        distorted_imgs = np.transpose(np.array(train, dtype=object), (0, 2, 3, 1)).astype('float32') 
        images = np.stack(truth)[:,:,:, np.newaxis].astype('float32') 
        #print(distorted_imgs.shape)
        #print(images.shape)
        return (distorted_imgs, images)


class SuperRes(keras.utils.Sequence):

    def __init__(self, batch_size, batches, img_shape, factor=4):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.batches = batches
        self.factor = factor
        config, _ = utils.upsample_transform(global_config, np.zeros((1, 1)), z=4)
        self.G = xtrace.depth_spill_psf(config, *utils.ray_grid(config["dimensions"]))
    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        
        images = [apply_blur(random_image(self.img_shape, 0.05, sc=4), self.G, True) for i in range(self.batch_size)]
        downsampled = [utils.downsample_img(img, 4) for img in images]
        images = np.array(images)[:,:,:,np.newaxis]
        downsampled = np.array(downsampled)[:,:,:, np.newaxis]
        return (downsampled, images)
    
def get_synthetic_data_pair(config=None, randgrid=True):
    if config is None:
        config = global_config
    #ray_gen_func = utils.random_ray_grid if randgrid else utils.ray_grid
    #ay_grid = ray_gen_func(config["dimensions"])
    #G = xtrace.depth_spill_psf(config, *ray_grid)
    img = random_image(config["dimensions"])
    distorted_img = apply_noise(apply_blur(img, G_glob, noise=True))
    return (distorted_img, img)

def get_synthetic_data_pair_positional(config=None, randgrid=False, randconfig=True, spread=1.0):
    if config is None:
        config = global_config
    if randconfig:
        config = random_config(config["dimensions"], spread=spread)
    ray_gen_func = utils.random_ray_grid if randgrid else utils.ray_grid
    ray_grid = ray_gen_func(config["dimensions"])
    G = xtrace.depth_spill_psf(config, *ray_grid)
    img = random_image(config["dimensions"])
    distorted_img = apply_noise(apply_blur(img, G, noise=True))
    xdiff, ydiff = xtrace.depth_offsets(config, *ray_grid)
    observed = np.stack((distorted_img, xdiff, ydiff))
    return (observed, img)

def apply_blur(img, G, noise):
    return (G.get()@img.flatten()).reshape(img.shape)

def apply_noise(img):
    img += 0.00008*rng.random()*rng.poisson(100,img.shape)/100
    return img

def _get_hitnoise(dimensions, density):
    hits = rng.random(dimensions) <= density
    noise = np.zeros(dimensions)
    noise[hits] = rng.exponential(0.01, size=np.count_nonzero(hits))
    hits = rng.random(dimensions) <= density*0.1*rng.random()
    noise[hits] = rng.exponential(0.7,size=np.count_nonzero(hits))
    return noise
    
def random_image(dimensions, density=0.02, sc=1):
    
    t_density = density/(sc*sc)
    #single_hits = randomly hitting rays
    single_hits = _get_hitnoise(dimensions, t_density*rng.random())
    
    s = 14*sc + 1
    c = np.floor(s/2)
    def eval(x, y):
        r_d_sq = ((x - c)**2 + (y - c)**2)*(40*(rng.random() + 0.2))**2/sc
        r_d_sq[r_d_sq == 0] = 0.1
        res = 1/r_d_sq
        return res
    kernel = np.fromfunction(eval,(s, s))
    kernel -= kernel.min()
    kernel /= kernel.max()
    
    single_hits = fftconvolve(single_hits, kernel, mode='same')
    
    #Gaussian hits = randomly hitting gaussians (made of multiple rays)
    gaussian_hits = _get_hitnoise(dimensions, t_density*rng.random()*0.1)
    s = 14*sc + 1
    kernel = makeGaussian(s, (0.1 + 1.5*rng.random())*sc)
    kernel -= kernel.min()
    kernel /= kernel.max()
    gaussian_hits = fftconvolve(gaussian_hits, kernel, mode='same')
    
    return single_hits + gaussian_hits #+ noise_image(dimensions, density*0.2, sc)

def noise_image(dimensions, density=0.02, sc=1):
    img = 0.00008 + 3*rng.random(dimensions)
    img[rng.random(dimensions) > density/(sc*sc)] = 0
    return img

#Source: https://gist.github.com/andrewgiessel/4635563
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

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

def random_config(dimensions, spread=1.0):
    dim = np.max(dimensions) 
    return { #RANDOM CONFIG
    "detector": {
            "pixel_dims": np.array([1.0, 1.0, 6]), #randomize this?
            "mu":  0.3
        },
    "dimensions": np.array(dimensions),
    "ray_origin": np.array([
            spread*dim*rng.normal() + dimensions[0]/2,
            spread*dim*rng.normal() + + dimensions[1]/2,
            5*dim - spread*4*dim*rng.random()
        ]),
    }