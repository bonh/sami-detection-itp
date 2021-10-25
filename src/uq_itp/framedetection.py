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
import bayesian
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
# To automize the frame detection, a Gaussian (signal model) is fitted to the correlation functions to determine whether there is a peak or not. Additionally, decision criterion is added by comparing the signal model with the noise-only model. The idea is, to keep frames, in which the signal model is more likely  to describe the data than the noise-only model. Additionally, a correlation between neighboring intervals is used, i.e. when the previous and the following interval are "informative" the tagged is also assumed to be "informative".

# +
lagstep = 1
# each delta frames use nset to calculate the correlation function
nset = 10
delta = 20

fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(12,30))
plt.subplots_adjust(wspace=0.3, hspace=0.4)
axs = axs.flatten()
keep = np.empty(len(axs), dtype=object)
results = np.empty(len(axs), dtype=object)

for i in range(len(axs)):
    start = i*delta
    
    # calculate correlation function
    corr = np.mean(dataprep.correlate_frames(data[:,start:start+nset], lagstep), axis=1)
    corr = corr.reshape(-1,1)
    
    # normalize data
    scale = preprocessing.StandardScaler().fit(corr)
    cor_scaled = scale.transform(corr).reshape((1,-1))[0]
    
    # fit model
    x = np.linspace(0, cor_scaled.shape[0], cor_scaled.shape[0])
    with bayesian.create_signalmodel(cor_scaled, x) as model:
        trace_mean = pm.sample(2000, return_inferencedata=True, cores=4)
    with bayesian.create_model_noiseonly(cor_scaled) as model:
        trace_noiseonly = pm.sample(2000, return_inferencedata=True, cores=4)
    
    # compare both models
    dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
    results[i] = dfwaic
    print(dfwaic)
    az.plot_compare(dfwaic, insample_dev=False);
    
    summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
    map_estimate = summary_mean.loc[:, "mean"]
    model_mean = bayesian.model_signal(map_estimate["amplitude"], map_estimate["centroid"],\
                                       map_estimate["sigma"], map_estimate["baseline"], x)
    
    # decision criterion
    keep[i] = dfwaic['waic']['noiseonly']+dfwaic['se']['noiseonly'] < dfwaic['waic']['sample']-dfwaic['se']['sample']
    print(keep[i])
    
    # plot fit and data
    axs[i].plot(cor_scaled, label='data')
    axs[i].set_title('start frame = %s' %start)
    axs[i].plot(x, model_mean, label="fit, mean", alpha=0.5)
    
    axs[i].legend()

plt.show()

# +
for i,d in enumerate(results):
    keep[i] = d['waic']['noiseonly']+d['se']['noiseonly'] < d['waic']['sample']-d['se']['sample']

# check neighboring intervals - add correlation ...
if np.all(keep==False):
    keep.fill(True)
else:
    for i in range(len(keep)):
        if i > 0 and i < len(keep)-2:
            if keep[i-1] == True and keep[i+1] == True:
                keep[i] = True
                
print(keep)

# calculate frames to keep
keep_frames = np.empty(data.shape[1], dtype=object)
keep_frames.fill(False)

for i,k in enumerate(keep):
    keep_frames[int(i*20):int((i+1)*20)] = k
    
#print(keep_frames)
minframe = np.min(np.where(keep_frames == True)[0])
maxframe = np.max(np.where(keep_frames == True)[0])
print('Frames to keep: %s - %s' %(minframe,maxframe))
# -









