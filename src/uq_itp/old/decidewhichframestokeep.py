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

plt.rcParams['figure.figsize'] = [10, 5]

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"

channel_lower = 27
channel_upper = 27


# -

def model_signal(amp, cent, sig, base, v, x, t):
    x_hat = x - v*t
    return amp*np.exp(-1*(cent - x_hat)**2/2/sig**2) + base


data_raw = input.load_nd_data(inname)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = input.substractbackground(data_raw, background)

data = input.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)

plt.imshow(data.T, origin="lower");

# +
step = 1

corr = np.zeros(data.shape)
for i in range(0,data.shape[1]-step):
    corr[:,i] = np.correlate(data[:,i], data[:,i+step], "same")
# -

plt.imshow(corr.T)

fig, axs = plt.subplots(2, 1)
axs[0].plot(corr[:,210])
axs[1].plot(corr[:,400])

frame = 110
plt.plot(np.correlate(corr[:,frame], corr[:,frame], "same"))
plt.plot(np.correlate(corr[:,400], corr[:,400], "same"),alpha=0.5)

plt.xcorr(corr[:,frame],corr[:,frame], maxlags=100);




