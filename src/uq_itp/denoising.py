# -*- coding: utf-8 -*-
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

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# +
import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import dataprep
import bayesian
# -

plt.rcParams['animation.html'] = "jshtml"#'html5'
plt.rcParams['figure.dpi'] = 72

# from experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"
channel = [27, 27]
lagstep = 30
frames = [100,300]

# +
data_raw = dataprep.load_nd_data(inname, verbose=False)
data_raw = dataprep.cuttochannel(data_raw, channel[0], channel[1])
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = dataprep.substractbackground(data_raw, background)
data = dataprep.averageoverheight(data_raw)
    
corr = dataprep.correlate_frames(data, lagstep)
    
corr = corr[:,frames[0]:frames[1]]
    
corr_mean = np.mean(corr, axis=1).reshape(-1, 1)
    
x = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])
corr_mean[int(corr_mean.shape[0]/2)] = 0
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
x = x[0:int(corr_mean.shape[0])]
x *= -1
    
scaler = preprocessing.MinMaxScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()
    
with bayesian.create_signalmodel(corr_mean, x) as model:
    trace_mean = pm.sample(20000, return_inferencedata=True, cores=4)
# -

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])

# +
map_estimate = summary_mean.loc[:, "mean"]
corr_mean_fit = bayesian.model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x)

dx = map_estimate["centroid"]*px #0.15e-3 # m
dt = lagstep/fps # s

v = dx/dt # mm/s
print("sample velocity = {} mm/s".format(v*1e3))

# +
data_shifted = dataprep.shift_data(data, v, fps, px)

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].imshow(data.T, aspect="auto", origin="lower")
axs[0].set_ylabel("frame (-)");
axs[0].set_title("raw data")
axs[1].imshow(data_shifted.T, aspect="auto", origin="lower")
axs[1].set_xlabel("length (px)")
axs[1].set_ylabel("frame (-)");
axs[1].set_title("shifted");
# -

data_mean = np.mean(data_shifted, axis=1)

x = np.linspace(0, data_mean.shape[0], data_mean.shape[0])
with bayesian.create_signalmodel(data_mean, x) as model:
    trace_mean = pm.sample(20000, return_inferencedata=True, cores=4)

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])

fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, data_mean, label="mean", alpha=0.8)
map_estimate = summary_mean.loc[:, "mean"]
model_mean = bayesian.model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x)
axs.plot(x, 
    model_mean, label="fit, mean", alpha=0.5)
axs.legend();
axs.set_xlabel("lag (px)")
axs.set_ylabel("intensity (AU)");

# +
model_mean = bayesian.model_signal(map_estimate["amplitude"], length/2, map_estimate["sigma"], map_estimate["baseline"], x)

matched = np.zeros((data.shape[0], data.shape[1]))
for i in range(0,data.shape[1]):
    matched[:,i] = np.correlate(data[:,i], model_mean, "same")
# -

fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].imshow(data.T, aspect="auto", origin="lower")
axs[0].set_ylabel("frame (-)");
axs[0].set_title("raw data")
axs[1].imshow(matched.T, aspect="auto", origin="lower")
axs[1].set_xlabel("length (px)")
axs[1].set_ylabel("frame (-)");
axs[1].set_title("shifted");

# +
from scipy.signal import butter,filtfilt

# Filter requirements.
T = nframes/fps         # Sample Period
fs = fps       # sample rate, Hz
cutoff = 5     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

y = butter_lowpass_filter(matched, cutoff, fs, order)

plt.plot(y[:,130], label="cut");
plt.plot(matched[:,130]);
plt.legend()
# -

y = (y - np.mean(y)) / np.std(y)

matched_height = np.repeat(y[np.newaxis, :, :], data_raw.shape[0], axis=0)

# +
fig, ax = plt.subplots(2,1,figsize=(10, 20*height/length),sharex=True)

im0 = ax[0].imshow(
    data_raw[:,:,0], animated=True, origin="lower", extent=(0, length*px*1e3, 0, height*px*1e3),
    vmin=np.min(data_raw), vmax=np.max(data_raw))
im1 = ax[1].imshow(
    matched_height[:,:,0], animated=True, origin="lower", extent=(0, length*px*1e3, 0, height*px*1e3),
    vmin=np.min(matched_height), vmax=np.max(matched_height))

step = 5
def updatefig(n):
    im0.set_array(data_raw[:,:,n*step]) 
    im1.set_array(matched_height[:,:,n*step])    
    ax[0].set_title("frame = {}".format(n*step))
    ax[1].set_xlabel("length (mm)")
    ax[0].set_ylabel("height (mm)")
    ax[1].set_ylabel("height (mm)")
    return im0, im1

ani = animation.FuncAnimation(fig, updatefig, frames=int(nframes/step))
plt.close()
fig.tight_layout()
writer = animation.PillowWriter(fps=10)
ani.save('denoising.gif', writer=writer, dpi=150)
ani
# -


