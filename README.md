# Removing Detector Depth Blur in Near Field Scattering - An Initial Exploration

This project expores methods of removing sensor depth spill for X-ray crystallography applications (and may be applicable to other fields!). Two primary
methods were tested: non-blind spatially variant Richard-Lucy Deconvolution and deep learning deconvolution with networks trained on
synthetic data.

# Setup

- Install tensorflow, numpy, cupy, matplotlib, scipy, pyFAI, h5py, hdf5plugin
- For evaluation of networks, they need to be placed in a folder "trained networks".
- Trained models (if you are a MAX IV employee) are in: /data/staff/common/data/pd/xtrace

# Project Structure

- **lib** - common code throughout the project
  - **xtrace.py** - estimation of spatially variant point spread function (and depth field estimates) and construction of detector gap transformation matrix
  - **deconvolution.py** - deconvolution method implementations
  - **datagen.py** - utilities to generate synthetic data for training the deep learning models
  - **mlmodels.py** - keras network models
  - **utils.py** - deconvolution utilites
- **psf_experiments** - illustration/explanation of point spread behaviour
  - **4_psf_example.ipynb** - illustrates point spread application
  - **4_psf_example_nonsq.ipynb** - illustrates point spread simulation with non-square detector
  - **4_psf_example_gaps.ipynb** - illustrates point spread application with detector gaps
- **deconv_richard_lucy_experiments** - evaluation of richard lucy deconvolution
  - **1_comparison_areas.ipynb** - deconvolution/raw data comparison for a number of areas of interest
  - **2_comparison_psf_methods.ipynb** - comparison of different deconvolution methods (local/updownsampled/normal)
  - **3_synthetic_example.ipynb** - synthetic example of convoluted and then deconvoluted image
- **denconv_ml_experiments** - training and evaluation of different deep learning models (both deconvolution and upscaling + deconvolution tests)
  - **eval_gyro.ipynb** - evaluation of gyro network (see next section)
  - **eval_meta.ipynb** - comparison between different methods (gyro, ruchard-lucy, static)
  - **eval_static.ipynb** - evaluation of static network
  - **eval_superres.ipynb** - evaluation of superres network 
  - **learning_gyroadapted.ipynb** - training of gyro network
  - **learning_static.ipynb** - training of static network
  - **learning_syperres.ipynb** - training of superres network
- **old_notebook_experiments** - collection of old code/past experiments, warning very messy

# ML Models

- **static** - network trained to deconvolve an image always affected by the same point spread in each location.
- **gyro** - repurposed network originally performing deconvolution of blurred images with added gyro information, gyro layers now replaced by depth field estimates to allow the network to learn deconvolution with different point spreads.
- **superres** - network performing both super resolution and deconvolution.
