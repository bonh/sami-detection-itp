# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2
# -

import numpy as np
import matplotlib.pyplot as plt
import dataprep
import utilities
from matplotlib.patches import Rectangle
import arviz as az
from sklearn import preprocessing
import pymc3 as pm
import matplotlib.animation as animation
from IPython.display import HTML

# IDEA: Use the correlation function of the frames to decide which frame might contain relevant data and which not. This is necessary to reduce the number of frames that only contain noise which would increase the detection rate/snr.

# +
inname = "../../../../03_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"

# to cut images to channel
channel_lower = 27
channel_upper = 27

# to cut images to the one containing the sample
startframe = 200
endframe = 300

# from experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)
# -

data_raw = dataprep.load_nd_data(inname, verbose=False)
data_raw = dataprep.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = dataprep.substractbackground(data_raw, background)

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))

tmp = dataprep.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)

x = np.linspace(0, length*px*1e3, length)

time = 180
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(x, data[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[1].imshow(data.T, aspect="auto", origin="lower", extent=(0, length*px*1e3, 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");


# Now, the data is pre-processed but it still contains frames with noise only. Here, we define a criterium that allows to decide which frames are relevant and which not. This is based on the cross-correlation function of the data.

# The whole data (frames) is divided into sets of frames (here each set contains 10 frames; see nset). The cross correlation with lagstep=1 is calculated and a running average is used to smooth out random peaks. Whenever a clear peak is visible in the correlation function, the frames might be relevant for the analysis. When there is no peak (only noise), this frames can savely be neglected. 

def runningmean(data, nav, mode='valid'):
    return np.convolve(data, np.ones((nav,)) / nav, mode='valid')


# +
lagstep = 1
nset = 10
nav = 10

fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(12,30))
plt.subplots_adjust(wspace=0.3, hspace=0.4)
axs = axs.flatten()

for i in range(len(axs)):
    start = i*20
    corr = np.mean(dataprep.correlate_frames(data[:,start:start+nset], lagstep), axis=1)
    axs[i].plot(runningmean(corr, nav))
    axs[i].set_title('start frame = %s' %start)

plt.show()
# -








