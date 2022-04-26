# Removing Detector Depth Blur in Near Field Scattering - An Initial Exploration

This project expores methods of removing sensor depth spill for X-ray crystallography applications. Two primary
methods were tested: non-blind spatially variant Richard-Lucy Deconvolution and deep learning deconvolution using
synthetic data.

# Setup

- install tensorflow, h5py, numpy, cupy, matplotlib
- for evaluation of networks, they need to be placed in a folder "trained networks".

# Project Strucutre

- **lib** - common code throughout the project
  - **xtrace.py** - estimation of spatially variant point spread function (and depth field estimates)
  - **deconvolution.py** - deconvolution method implementations
  - **datagen.py** - utilities to generate synthetic data for training the deep learning models
  - **mlmodels.py** - keras network models
  - **utils** - deconvolution utilites
- **psf_experiments** - illustration/explanation of point spread behaviour
- **deconv_richard_lucy_experiments** - evaluation of richard lucy deconvolution
- **denconv_ml_experiments** - training and evaluation of different deep learning models (both deconvolution and upscaling + deconvolution tests)
- **old_notebook_experiments** - collection of old code/past experiments, warning very messy