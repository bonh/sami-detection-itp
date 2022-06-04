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

import numpy as np
import matplotlib.pyplot as plt
import input
import utilities
from matplotlib.patches import Rectangle
import arviz as az
from sklearn import preprocessing
import pymc3 as pm

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/003.nd2"

channel_lower = 27
channel_upper = 27

startframe = 100
endframe = 300
# -

data_raw = input.load_nd_data(inname)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = input.substractbackground(data_raw, background)
data = input.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)

# +
step = 30

corr = np.zeros(data.shape)
for i in range(0,data.shape[1]-step):
    corr[:,i] = np.correlate(data[:,i], data[:,i+step], "same")

# +
tmp = np.mean(corr[:,150:300], axis=1).reshape(-1, 1)

# clean the correlation data
# remove peak at zero lag
tmp[int(tmp.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
#tmp = tmp[0:int(tmp.shape[0]/2)]

#scaler = preprocessing.StandardScaler().fit(tmp)
scaler = preprocessing.MinMaxScaler().fit(tmp)
tmp = scaler.transform(tmp).flatten()


# +
def model_signal(amp, cent, sig, base, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + base

def create_model(data, x):
    model = pm.Model()
    with model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 1)
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

def create_model_nosignal():
    model = pm.Model()
    with model:
        # prior peak
        base = pm.Normal('baseline', 0, 0.5)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1)

        # likelihood
        likelihood = pm.Normal('y', mu = base, sd=sigma_noise, observed = data)

        return model


# +
x = np.linspace(0, tmp.shape[0], tmp.shape[0])
with create_model(tmp, x) as model:
    trace_signal = pm.sample(return_inferencedata=True, cores=4)
    
with create_model_nosignal() as model:
    trace_nosignal = pm.sample(return_inferencedata=True, cores=4)
# -

rating = pm.compare({"sample":trace_signal, "noiseonly":trace_nosignal}, ic="loo")
rating

# +
tmp = np.mean(data[:,50:200], axis=1)

fps = 46
px = 1.6e-6

d = 512/2-np.argmax(tmp)
dx = d*px
t = step
dt = t/fps

v = dx/dt
print(v*1e3)
# -

plt.plot(tmp);
map_estimate = summary.loc[:, "mean"]
plt.plot(
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="model");
plt.legend();
