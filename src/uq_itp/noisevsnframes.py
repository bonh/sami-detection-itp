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
import helper

# +
#inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_10ng_l/001.nd2"
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"

channel_lower = 27
channel_upper = 27

startframe = 150
endframe = 250

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

sigma_mean = 10
rope_sigma = (5,15)
rope_velocity = (200,250)

time = 150

# +
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))
# -

tmp = dataprep.averageoverheight(data_raw)
scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)

# +
lagstep = 30 
corr = dataprep.correlate_frames(data, lagstep)

scaler = preprocessing.StandardScaler().fit(corr)
corr = scaler.transform(corr)

corr_mean = np.mean(corr[:,startframe:endframe], axis=1).reshape(-1, 1)
x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

# clean the correlation data
# remove peak at zero lag
corr_mean[int(corr_mean.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

x_lag = x_lag[0:int(corr_mean.shape[0])]

scaler = preprocessing.StandardScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()
# -

with bayesian.signalmodel_correlation(corr_mean, -x_lag, px, lagstep, fps) as model:
    trace = pm.sample(return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc = pm.fast_sample_posterior_predictive(trace, model=model)
    idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
    summary = az.summary(idata, var_names=["sigma_noise", "sigma", "centroid", "amplitude", "c", "b", "velocity"])
    
    hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)

v = summary["mean"]["velocity"]*1e-6
print("Mean velocity of the sample is v = {} $microm /s$.".format(v*1e6))
data_shifted = dataprep.shift_data(data, v, fps, px)

# +
frames2try = np.linspace(startframe+10, endframe, 6, dtype=int)
print(frames2try)
N = len(frames2try)

j = 0

idatas = []
snr = np.zeros((4,N))
for i in range(0,N):
    j = 0
    while True:
        try:
            print(frames2try[i]-startframe)
            tmp = data_shifted[:, startframe:frames2try[i]]
            avg = np.mean(tmp, axis=1).reshape(-1,1)
    
            scaler = preprocessing.StandardScaler().fit(avg)
            avg = scaler.transform(avg).flatten()
    
            x = np.linspace(0, len(avg), len(avg))
            with bayesian.signalmodel(avg, x) as model:
                trace = pm.sample(2000, return_inferencedata=False, cores=4, target_accept=0.9)
    
                ppc = pm.fast_sample_posterior_predictive(trace, model=model)
                idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
                summary = az.summary(idata, var_names=["sigma_noise", "sigma", "centroid", "amplitude", "c", "fmax", "snr", "alpha"])
        
            mean = summary.loc[:, "mean"]
            hdi_low = summary.loc[:, "hdi_3%"]
            hdi_high = summary.loc[:, "hdi_97%"]

            snr[:,i] = [frames2try[i]-startframe, mean["snr"], hdi_low["snr"], hdi_high["snr"]]
            idatas.append(idata)
        except:
            if j<3:
                print("retry")
                j+=1
                continue
            else:
                break
        break

# +
fig, ax1 = plt.subplots()

x = snr[0,:]
y = snr[1,:]
yerr = np.abs(snr[2:4,:]-y)
ax1.plot(x, y, marker="o", color="red", linestyle="None", label="snr and .97 HDI")
x_ = np.linspace(0, x[-1], 100)
ax1.plot(x_, y[-3]*np.sqrt(x_/x[-3]), label="$\propto \sqrt{n}$") 
ax1.annotate(r"$\mathit{snr} \propto N$", xy=(50, 11), xytext=(40, 15), arrowprops=dict(arrowstyle="->"))
ax1.set_xlabel("number of frames $N$")
ax1.set_ylabel("signal-to-noise $snr$")

def bla(endframe):
    tmp = data_shifted[:, startframe:endframe]
    avg = np.mean(tmp, axis=1).reshape(-1,1)
    
    scaler = preprocessing.StandardScaler().fit(avg)
    return scaler.transform(avg).flatten()

x = np.linspace(0, len(avg), len(avg))

left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x[::5], bla(frames2try[0])[::5], alpha=0.5)
ax2.plot(x, idatas[0].posterior_predictive.mean(("chain", "draw"))["y"], color="red")
ax1.annotate("", xy=(10, 10.5), xytext=(10, 5), arrowprops=dict(arrowstyle="<-"))
ax2.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([])

left, bottom, width, height = [0.67, 0.2, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x[::5], bla(frames2try[-1])[::5], alpha=0.5)
ax2.plot(x, idatas[-1].posterior_predictive.mean(("chain", "draw"))["y"], color="red")
ax1.annotate("", xy=(90, 6), xytext=(100, 14), arrowprops=dict(arrowstyle="<-"))
ax2.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([]);

left, bottom, width, height = [0.4, 0.2, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x[::5], bla(frames2try[-4])[::5], alpha=0.5)
ax2.plot(x, idatas[-4].posterior_predictive.mean(("chain", "draw"))["y"], color="red")
ax1.annotate("", xy=(50, 6), xytext=(47, 9.5), arrowprops=dict(arrowstyle="<-"))
ax2.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([]);
