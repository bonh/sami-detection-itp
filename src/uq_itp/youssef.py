# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# + tags=[]
import numpy as np
import matplotlib.pyplot as plt
import input
import utilities
from matplotlib.patches import Rectangle
import arviz as az
from sklearn import preprocessing
import pymc3 as pm

# + tags=[]
plt.rcParams['figure.figsize'] = [10, 5]
# -

# 1. Load the raw microscopy images.
# 2. Cut the height of the image to the microchannel.
# 3. Substract a mean background image

# + tags=[]
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_10ng_l/001.nd2"

channel_lower = 27
channel_upper = 27

startframe = 100
endframe = 300

# + tags=[]
data_raw = input.load_nd_data(inname)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = input.substractbackground(data_raw, background)

# + tags=[]
plt.imshow(data_raw[:,100], origin="lower");
plt.xlabel("Length")
plt.ylabel("Height")

currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((80, 0), 40, data_raw[:,100].shape[0]-5, linewidth=2, edgecolor='r', facecolor='none'))

plt.text(130,30, "sample\nmoves from the left to the right", color="red",fontsize=14);
# -

# 4. Average signal over height to reduce noise

tmp = input.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)

time = 100
fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(data[:,time])
axs[0].set_ylabel("Intensity");
axs[1].imshow(data.T, aspect="auto",origin="lower")
axs[1].set_xlabel("Length")
axs[1].set_ylabel("Time");
axs[1].text(130,200, "sample moves with constant velocity", color="red",fontsize=14);


# 5. Shift every image pixel by the constant velocity (which is a good assumption) of the sample (Galilean transformation)

def shift_data(data, v, fps, px):
    dx =  v/fps
    data_shifted = np.zeros(data.shape)
    for i in range(0, data.shape[1]):
        shift = data.shape[0] - int(i*dx/px)%data.shape[0]
        data_shifted[:,i] = np.roll(data[:,i], shift)
    
    #data_shifted = np.roll(data_shifted, int(data_shifted.shape[1]/2), axis=1)
    
    return data_shifted


# +
v = 0.4e-3
fps = 46
px = 1.6e-6

data_shifted = shift_data(data, v, fps, px)
# -

time = 100
fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(data[:,time], label="raw")
axs[0].plot(data_shifted[:,time], label="shifted")
axs[0].plot(np.mean(data_shifted, axis=1), label="shifted/averaged")
axs[0].plot(np.mean(data_shifted[:,startframe:endframe], axis=1), label="shifted/cut/averaged")
axs[0].set_ylabel("Intensity");
axs[0].legend()
axs[1].imshow(data_shifted.T, aspect="auto",origin="lower")
axs[1].set_xlabel("Length")
axs[1].set_ylabel("Time");
axs[1].text(130,250, "Samples shifted with v={}mm/s, so it's not perfect.".format(v*1000), color="red",fontsize=14);


# 6. Create Bayesian model of the sample distribution at a specific time (almost Gaussian). Find the velocity of the sample by optimization. In a way, we treat the velocity as a hyperparameter.

# +
def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

def create_model(data, x):
    model = pm.Model()
    with model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 0.5)
        cent = pm.Normal('centroid', 250, 200)
        base = pm.Normal('baseline', 0, 0.5)
        sig = pm.HalfNormal('sigma', 50)

        # forward model signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, base, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1)

        # likelihood
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)

        return model


# -

def functional(v):
    data_shifted = shift_data(data, v, fps, px)
    
    avg = np.mean(data_shifted[:,startframe:endframe], axis=1)
    
    x = np.linspace(0, len(avg), len(avg))
    with create_model(avg, x) as model:
        trace = pm.sample(10000, return_inferencedata=False, cores=4, progressbar=False)
        
    alpha = 1e-5
    hdi_centroid = az.hdi(trace["centroid"])
    hdi_sigma = az.hdi(trace["sigma"])
    result = 1/2*np.sqrt(
        (hdi_centroid[1] - hdi_centroid[0])**2 \
        + (hdi_sigma[1] - hdi_sigma[0])**2) \
        + alpha/2*np.sqrt(v**2)
    
    return result


# +
from gradient_free_optimizers import *

def functional2(para):
    return -functional(para["v"])

search_space = {"v": np.arange(2e-4, 4e-4, 1e-6)}

#opt = RandomSearchOptimizer(search_space)
#opt = BayesianOptimizer(search_space)
opt = EvolutionStrategyOptimizer(search_space)
#opt = SimulatedAnnealingOptimizer(search_space)
opt.search(functional2, n_iter=20, early_stopping={"n_iter_no_change":3}, max_score=-0.1)
# -

v = 2.39e-4#opt.best_para["v"]
data_shifted = shift_data(data, v, fps, px) 
avg = np.mean(data_shifted[:,startframe:endframe], axis=1)

time = 100
fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(data[:,time], label="raw")
axs[0].plot(data_shifted[:,time], label="shifted")
axs[0].plot(np.mean(data_shifted, axis=1), label="shifted/averaged")
axs[0].plot(np.mean(data_shifted[:,startframe:endframe], axis=1), label="shifted/cut/averaged")
axs[0].set_ylabel("Intensity");
axs[0].legend()
axs[1].imshow(data_shifted.T, aspect="auto",origin="lower")
axs[1].set_xlabel("Length")
axs[1].set_ylabel("Time");
axs[1].text(130,250, "Samples shifted with v={}mm/s. It's way better.".format(v*1000), color="red",fontsize=14);

x = np.linspace(0, len(avg), len(avg))
with create_model(avg, x) as model:
    trace = pm.sample(10000, return_inferencedata=True)
    
    #map_estimate = pm.find_MAP()
    #print(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], map_estimate["sigma_noise"])

summary = az.summary(trace, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary

az.plot_posterior(trace, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"]);

plt.plot(avg, label="shifted/cut/averaged")
plt.plot(data[:,time], label="raw")
map_estimate = summary.loc[:, "mean"]
plt.plot(
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="model");
plt.legend();


