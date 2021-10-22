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

import helper
import dataprep
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
from PIL import Image

# +
inname_base = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/"

channel_lower = 27
channel_upper = 27

startframe = 150
endframe = 170

fps = 46 # frames per second (1/s)
px = 1.6e-6# size of pixel (m/px)

lagstep = 30

rope_sigma = (5,15)
rope_velocity = (200,250)
rope = {'sigma': [{'rope': rope_sigma}]
        , 'snr': [{'rope': (0, 1)}]
        , 'velocity': [{'rope': rope_velocity}]}
ref_val = {"sigma": [{"ref_val":20}]
           , "snr": [{"ref_val":3}]
           , "velocity": [{"ref_val":230}]}


# -

def raw2corrmean(number):
    inname = inname_base + "00{}.nd2".format(number)
    data = helper.raw2images(inname, (channel_lower, channel_upper))
    
    data = np.mean(data, axis=0)
    
    length_px, nframes = data.shape
    corr = dataprep.correlate_frames(data, lagstep)
    
    corr_mean = np.mean(corr[:,startframe:endframe], axis=1).reshape(-1, 1)

    #corr_mean[int(corr_mean.shape[0]/2)] = 0
    corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
    
    scaler = preprocessing.MinMaxScaler().fit(corr_mean)
    corr_mean = scaler.transform(corr_mean).flatten()
    
    return corr_mean


def signalmodel(data, x):
    with pm.Model() as model:
        # background
        # f = b*x + c
        b = pm.Normal('b', 0, 1)
        c = pm.Normal('c', 0, 1)

        background = pm.Deterministic("background", b*x+c)

        # peak
        amp = pm.Uniform('amplitude', 0, 2) 
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.Uniform('sigma', 0, 100) # TODO: calculate from physics?

        def model_signal(amp, cent, sig, x):
            return amp*np.exp(-1*(cent - x)**2/2/sig**2)

        # signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = background+signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        velocity = pm.Deterministic("velocity", (len(data)-cent)*px/(lagstep/fps)*1e6)
        snr = pm.Deterministic("snr", amp/sigma_noise)

        # sample the model
        tracesignal = pm.sample(15000, return_inferencedata=False, cores=4)
        modelsignal = model
        
        return modelsignal, tracesignal


x = np.linspace(0, len(corr_mean), len(corr_mean))
for number in [1]:
    corr_mean = raw2corrmean(number)
    
    modelsignal, tracesignal = signalmodel(corr_mean, x)

ppc = pm.fast_sample_posterior_predictive(tracesignal, model=modelsignal)
idata = az.from_pymc3(trace=tracesignal, posterior_predictive=ppc, model=modelsignal) 

fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, corr_mean, label="averaged", alpha=0.8, color="black")
axs.plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], color="red", label="signal fit")
axs.legend();
axs.set_xlabel("length (px)")
axs.set_ylabel("intensity (AU)");

az.summary(idata, var_names=["snr", "sigma_noise", "amplitude", "sigma", "velocity", "centroid", "b", "c"])

az.plot_posterior(idata, var_names=["sigma", "velocity"], rope=rope, hdi_prob=.95);

hdi = az.hdi(idata.posterior, hdi_prob=.95, var_names=["sigma", "velocity"])
hdi_sigma = hdi["sigma"].data
hdi_velocity = hdi["velocity"].data

values = idata.posterior["sigma"]
vals = rope_sigma
prob = ((values > vals[0]) & (values <= vals[1])).mean()
prob.data

values = idata.posterior["velocity"]
vals = rope_velocity
prob = ((values > vals[0]) & (values <= vals[1])).mean()
prob.data


