# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import bayesian
import dataprep


# -

def create_images(length, height, nframes, a, c, w, alpha, c_add, x): 
    data = np.zeros((height, length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    for h in range(0, height):
        data[h, :, :] = bayesian.model_sample(a, cc+c_add[h], w[h], alpha, xx).eval().T
            
    return data


# +
length = 512
height = 40
nframes = 500

a = 1
w_base = 10
alpha = 0

c_init = -100

v = 230e-6

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

x = np.linspace(0, length, length)

n = np.arange(0, nframes)     
c = c_init+n*v/(fps*px)
 
h = np.linspace(0, height, height)
w = (w_base+(h-height/2)**2/5)
c_add = -(h-height/2)**2/5

data_raw = create_images(length, height, nframes, a, c, w, alpha, c_add, x)
# -

data = dataprep.averageoverheight(data_raw)
data = dataprep.standardize(data)

fig, ax = plt.subplots(4,1, sharex=True)
ax[0].imshow(data_raw[:,:,50], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[1].imshow(data_raw[:,:,100], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[2].imshow(data_raw[:,:,150], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].imshow(data_raw[:,:,200], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].set_xticks(np.linspace(0, length, 5))
fig.tight_layout()

# +
snr = 0.5 # a/sigma
a = data_raw.max() # standardized
sigma = a/snr

data_noisy = data_raw + np.random.normal(0,sigma,data_raw.shape)
data_noisy = dataprep.standardize(data_noisy)
# -

fig, ax = plt.subplots(4,1, sharex=True)
ax[0].imshow(data_noisy[:,:,50], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[1].imshow(data_noisy[:,:,100], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[2].imshow(data_noisy[:,:,150], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].imshow(data_noisy[:,:,200], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].set_xticks(np.linspace(0, length, 5))
fig.tight_layout()

plt.plot(data_noisy[20,:,150]);
plt.plot(data_raw[20,:,150]);
