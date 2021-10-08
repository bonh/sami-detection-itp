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

# from experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

512*px*fps/2.2e-4

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"
channel = [27, 27]
lagstep = 30
frames = [170,270]

# +
data_raw = dataprep.load_nd_data(inname, verbose=False)
#data_raw = dataprep.cuttochannel(data_raw, channel[0], channel[1])
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

dx = map_estimate["centroid"]*px
dt = lagstep/fps # s

v = dx/dt # mm/s
print("sample velocity = {} mm/s".format(v*1e3))
# -

data_shifted = dataprep.shift_data(data, v, fps, px)
plt.imshow(data_shifted.T,origin="lower")

# +
frames2try = np.arange(frames[0]+1, frames[0]+100, 10)
print(frames2try)
N = len(frames2try)

j = 0
a, sigma, nframes = np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(0,N):
    tmp = data_shifted[:, frames[0]:frames2try[i]]
    avg = np.mean(tmp, axis=1).reshape(-1,1)
    
    scaler = preprocessing.StandardScaler().fit(avg)
    avg = scaler.transform(avg).flatten()
    
    x = np.linspace(0, len(avg), len(avg))

    with bayesian.create_signalmodel(avg, x) as model:
        trace_mean = pm.sample(20000, return_inferencedata=True, cores=4, progressbar=False)
        
    summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
    map_estimate = summary_mean.loc[:, "mean"]
        
    print(frames2try[i]-frames[0], map_estimate["amplitude"], map_estimate["sigma_noise"])
    a[j]  = map_estimate["amplitude"]
    sigma[j] = map_estimate["sigma_noise"]
    
    j+=1
# -

x = frames2try-frames[0]
x_ = np.linspace(x[0], x[-1], 100)
snr = a/sigma
plt.plot(x, snr, "ro", label="$snr$")
plt.plot(x_, np.sqrt(x_), label="$\propto \sqrt{n}$") 
plt.xlabel("no. frames $n$")
plt.ylabel("signal-to-noise $snr=A/\sigma$")
plt.legend();


