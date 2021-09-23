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

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import input
import utilities

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"

# #### Read the raw images and transform them into an numpy array
# #### Cut the image to channel
# #### Subtract the background mean from the raw images
# #### Average the images along the y-axis (height)
def rawimages2heightaveraged(inname):
    data_raw = input.load_nd_data(inname)
    data_raw = input.cuttochannel(data_raw, 27, 27)
    background = np.mean(data_raw[:,:,0:10],axis=2)
    data_raw = input.substractbackground(data_raw, background)
    return input.averageoverheight(data_raw)

time = 200
fig,axs = plt.subplots(2,1,sharex=True, figsize=(7,5))
axs[0].plot(data[:,time])
axs[1].imshow(np.transpose(data), aspect="auto",origin="lower")
axs[1].hlines(time, 0, 500, "r")
axs[1].set_xlabel("x")
axs[1].set_ylabel("t");

# Perform averaging

def shift_data(data, v, fps, px):
    dx =  v/fps
    data_shifted = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        shift = data.shape[1] - int(i*dx/px)%data.shape[1]
        data_shifted[i,:] = np.roll(data[i,:], shift)
    
    data_shifted = np.roll(data_shifted, int(data_shifted.shape[1]/2), axis=1)
    
    return data_shifted


# Perform Bayesian inference

# +
import pymc3 as pm

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


# +
data_in = np.transpose(data)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data_in)
data_in = scaler.transform(data_in)

#data_in = data_in[100:150,:]
#print(data_in.shape)

#plt.plot(data_in[2,:]);

# +
v = 2.3e-4#2.75e-4
fps = 46
px = 1.6e-6

j=0
a, sigma = np.zeros(10), np.zeros(10)
for i in np.linspace(100, 300, 10, dtype=int):

    data_in_ = data_in[100:i,:]
    
    data_shifted = shift_data(data_in_, v, fps, px)

    avg = np.mean(data_shifted, axis=0)
    
    x = np.linspace(0, len(avg), len(avg))

    with create_model(avg, x) as model:
        map_estimate = pm.find_MAP(model=model)
    a[j]  = map_estimate["amplitude"]
    sigma[j] = map_estimate["sigma_noise"]
    j+=1
# -

sigma

x = np.linspace(0, len(sigma), len(sigma))
n = np.linspace(100,300,10)
plt.plot(n, sigma**2, label="estimate")
plt.plot(n, 1/np.sqrt(n-100)/5)
#plt.plot(n, 1/np.sqrt(n))
plt.legend()

# +
v = 2.3e-4#2.75e-4
fps = 46
px = 1.6e-6

data_shifted = shift_data(data_in, v, fps, px)

avg = np.mean(data_shifted, axis=0)
# -

plt.plot(avg)

# +
x = np.linspace(0, len(avg), len(avg))

with create_model(avg, x) as model:
    map_estimate = pm.find_MAP(model=model)
    print(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], map_estimate["sigma_noise"])
    
plt.plot(avg)
plt.plot(model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x));
# -

with create_model(avg, x) as models:
    trace = pm.sample(10000, return_inferencedata=True)

import arviz as az
az.plot_posterior(trace, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"]);


# Hyperparameter optimization (velocity)

def functional(v):
    data_shifted = shift_data(data_in, v, fps, px)
    
    avg = np.mean(data_shifted, axis=0)
    avg = (avg - np.min(avg))/(np.max(avg)-np.min(avg))
    
    with create_model(avg, x) as model:
        trace = pm.sample(10000, return_inferencedata=False, cores=4, progressbar=False)
        
    alpha = 1e-3
    hdi_centroid = az.hdi(trace["centroid"])
    hdi_sigma = az.hdi(trace["sigma"])
    result = 1/2*np.sqrt(
        (hdi_centroid[1] - hdi_centroid[0])**2 \
        + (hdi_sigma[1] - hdi_sigma[0])**2) \
        + alpha/2*np.sqrt(v**2)
    
    return result


# +
# import skopt
# from skopt.space import Real
# from skopt import gp_minimize

# def functional2(para):
#     return functional(para[0])

# search_space = [Real(1e-4, 9e-4, name='v')]
# result = gp_minimize(functional2, search_space)

from gradient_free_optimizers import *

def functional2(para):
    return -functional(para["v"])

search_space = {"v": np.arange(2e-4, 4e-4, 1e-5)}

#opt = RandomSearchOptimizer(search_space)
#opt = BayesianOptimizer(search_space)
opt = EvolutionStrategyOptimizer(search_space)
#opt = SimulatedAnnealingOptimizer(search_space)
opt.search(functional2, n_iter=20, early_stopping={"n_iter_no_change":3}, max_score=-1)
# -

# #### Shift data with optimal velocity

# +
v = 2.3e-4#opt.best_para["v"]
data_shifted = shift_data(data_in, v, fps, px)
    
avg = np.mean(data_shifted, axis=0)

# +
with create_model(avg, x) as model:
    map_estimate = pm.find_MAP(model=model)
    print(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], map_estimate["sigma_noise"])
    
plt.plot(avg)
plt.plot(model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x));

# +
with create_model(avg, x) as models:
    trace = pm.sample(10000, return_inferencedata=False, cores=4)
    
import arviz as az
az.plot_posterior(trace, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"]);


# -

# #### Detection of sample

def create_model_noiseonly(data):
    model = pm.Model()
    with model:
        # baseline only
        base = pm.Normal('baseline', 0.5, 0.5)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1)

        # likelihood
        likelihood = pm.Normal('y', mu = base, sd=sigma_noise, observed = data)

        return model


# +
from collections import OrderedDict
models = OrderedDict()

with create_model(avg, x) as models[0]:
    #trace = pm.sample(10000, return_inferencedata=True, cores=4)
    trace = pm.sample_smc(1000, random_seed=42, parallel=True)
  
with create_model_noiseonly(avg) as models[1]:
    #trace_noiseonly = pm.sample(10000, return_inferencedata=True, cores=4)
    trace_noiseonly = pm.sample_smc(1000, random_seed=42, parallel=True)
# -

dfloo = pm.compare({"sample":trace, "noiseonly":trace_noiseonly}, ic="loo")
dfloo

az.plot_compare(dfloo, insample_dev=False);

BF_smc = np.exp(trace.report.log_marginal_likelihood - trace_noiseonly.report.log_marginal_likelihood)
BF_smc
#np.round(BF_smc)



data_model = model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x)

plt.plot(data_model)
tmp = (data_in[100,:]-np.min(data_in[100,:]))/(np.max(data_in[100,:])-np.min(data_in[100,:]))
plt.plot(tmp)

plt.plot(np.correlate(tmp, data_model, mode="same"))


