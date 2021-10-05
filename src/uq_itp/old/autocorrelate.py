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

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import input
import utilities
import data
from matplotlib.patches import Rectangle
import arviz as az
from sklearn import preprocessing
import pymc3 as pm
from collections import OrderedDict

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/005.nd2"

channel_lower = 27
channel_upper = 27

startframe = 150
endframe = 200

fps = 46
px = 1.6e-6
# -

data_raw = input.load_nd_data(inname)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:50],axis=2)
data_raw = input.substractbackground(data_raw, background)
data = input.averageoverheight(data_raw)

plt.imshow(data.T);

corrs = []
vs = np.linspace(1e-4, 5e-4, 200)
for v in vs:
    data_shifted = utilities.shift_data(data, v, fps, px) 
    avg = np.mean(data_shifted[:,startframe:endframe], axis=1)
    corrs.append(np.correlate(avg, avg, mode='valid'))
plt.plot(vs, corrs);
plt.vlines(vs[np.argmax(corrs)], np.min(corrs), np.max(corrs), "r");
#plt.text(0.00015, 50, "v={}mm/s".format(vs[np.argmax(corrs)]*1e3));

# +
def functional(v):
    data_shifted = utilities.shift_data(data, v, fps, px) 
    avg = np.mean(data_shifted[:,startframe:endframe], axis=1)
    return np.correlate(avg, avg, mode='valid')

from gradient_free_optimizers import *

def functional2(para):
    return functional(para["v"])

# take the v calculated before
search_space = {"v": np.arange(1e-4, 5e-4, 1e-6)}

opt = RandomSearchOptimizer(search_space)
opt.search(functional2, n_iter=2000, verbosity=["print_results"])
# -

v = opt.best_para["v"]
data_shifted = utilities.shift_data(data, v, fps, px) 
avg = np.mean(data_shifted[:,startframe:endframe], axis=1)


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
models = OrderedDict()

x = np.linspace(0, len(avg), len(avg))
with create_model(avg, x) as models[0]:
    #trace = pm.sample(10000, return_inferencedata=True, cores=4)
    trace = pm.sample_smc(1000, random_seed=42, parallel=True)
  
with create_model_noiseonly(avg) as models[1]:
    #trace_noiseonly = pm.sample(10000, return_inferencedata=True, cores=4)
    trace_noiseonly = pm.sample_smc(1000, random_seed=42, parallel=True)
# -

summary = az.summary(trace, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary

summary_noiseonly = az.summary(trace_noiseonly, var_names=["baseline", "sigma_noise"])
summary_noiseonly

dfwaic = pm.compare({"sample":trace, "noiseonly":trace_noiseonly}, ic="waic")
dfwaic

az.plot_compare(dfwaic, insample_dev=False);

BF_smc = np.exp(trace.report.log_marginal_likelihood - trace_noiseonly.report.log_marginal_likelihood)
BF_smc

plt.plot(avg, label="shifted/cut/averaged")
map_estimate = summary.loc[:, "mean"]
plt.plot(
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="signal");
plt.plot(x*0+summary_noiseonly.loc[:,"mean"]["baseline"], label="no signal");
plt.legend();


