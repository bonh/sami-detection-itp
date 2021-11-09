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

# +
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

import helper
import dataprep
import bayesian
# -

#mpl.rcParams['figure.dpi'] = 100
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathtools}'
plt.style.use(['science', 'vibrant'])

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_10ng_l/001.nd2"
#inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"

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

# +
v = summary["mean"]["velocity"]*1e-6
print("Mean velocity of the sample is v = {} $microm /s$.".format(v*1e6))
data_shifted = dataprep.shift_data(data, v, fps, px)

data_mean = np.mean(data_shifted[:,startframe:endframe], axis=1).reshape(-1, 1)

scaler = preprocessing.StandardScaler().fit(data_mean)
data_mean = scaler.transform(data_mean).flatten()
# -

x = np.linspace(0, len(data_mean), len(data_mean))
with bayesian.signalmodel(data_mean, x) as model:
    trace2 = pm.sample(4000, return_inferencedata=False, cores=4, target_accept=0.9)
    
    ppc2 = pm.fast_sample_posterior_predictive(trace2, model=model)
    idata2 = az.from_pymc3(trace=trace2, posterior_predictive=ppc2, model=model) 
    summary2 = az.summary(idata2, var_names=["sigma_noise", "sigma", "centroid", "amplitude", "c", "b", "snr"])
    
    hdi2 = az.hdi(idata2.posterior_predictive, hdi_prob=.95)

with bayesian.signalmodel(data[:,time], x) as model:
    trace3 = pm.sample(return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc3 = pm.fast_sample_posterior_predictive(trace3, model=model)
    idata3 = az.from_pymc3(trace=trace3, posterior_predictive=ppc3, model=model) 
    summary3 = az.summary(idata3)

# +
fig = plt.figure(constrained_layout=True, figsize=(15,7))
gs = GridSpec(3, 6, figure=fig)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.imshow(data_raw[:,:,time], origin="lower", extent=(0, length, 0, height),aspect="auto")
ax1.set_yticks(np.linspace(0, height, 3))
ax1.set_ylabel("height (px)")
ax1.set_title("A (cut to channel, background subtracted)", loc="left")

#ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
#ax2.plot(data[:,time])
#ax2.set_ylabel("intensity (AU)");
#ax2.set_title("B (height averaged)", loc="left")

ax3 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
ax3.imshow(data.T, aspect="auto", origin="lower", extent=(0, length, 0, nframes))
ax3.set_xticks(np.linspace(0, length, 5))
ax3.set_xlabel("length (px)");
ax3.set_ylabel("frame (-)");
ax3.set_title("B (height averaged)", loc="left")
ax3.set_yticks(np.linspace(0, nframes, 3))


plt.setp(ax1.get_xticklabels(), visible=False);
#plt.setp(ax2.get_xticklabels(), visible=False);

#
#ax4 = fig.add_subplot(gs[0, 1])
#ax4.plot(x_lag, corr[0:int(corr.shape[0]/2),time])
#ax4.set_ylabel("corr.\nintensity (AU)", va='center');

ax5 = fig.add_subplot(gs[0, 2:4])
ax5.imshow(corr[0:int(corr.shape[0]/2),:].T, aspect="auto", origin="lower", extent=[x_lag[0],x_lag[-1],0,corr[0:int(corr.shape[0]/2),:].T.shape[0]])
ax5.set_ylabel("corr. frame");
ax5.set_title("C (correlated frames)", loc="left")
#
ax5.set_yticks(np.linspace(0, corr[0:int(corr.shape[0]/2),:].T.shape[0], 3))

ax6 = fig.add_subplot(gs[1, 2:4], sharex=ax5)
ax6.plot(x_lag, corr_mean, alpha=0.8)
ax6.plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit")
ax6.fill_between(x_lag, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI")
ax6.set_ylabel("avg. corr. intensity (AU)")
ax6.set_xlabel("lag (px)");
ax6.set_title("D (avg. corr. frames, fit, .95 HDI)", loc="left")
ax6.set_xticks(np.linspace(-length/2, 0, 5))
ax6.set_yticks(np.linspace(0, np.ceil(np.max(corr_mean)), 3))

#plt.setp(ax4.get_xticklabels(), visible=False);
plt.setp(ax5.get_xticklabels(), visible=False);

#
ax7 = fig.add_subplot(gs[2, 2:3])
rope = {'sigma': [{'rope': rope_sigma}]
        , 'velocity': [{'rope': rope_velocity}]}
axs = az.plot_posterior(idata, var_names=["sigma"], rope=rope, kind="hist", point_estimate='mean', hdi_prob=.95, ax=ax7, textsize=10);
axs.set_title("")
ax7.set_title("E (marginal posterior+ROPE)", loc="left")
axs.set_xlabel("spread")


ax8 = fig.add_subplot(gs[2, 3:4])
axs = az.plot_posterior(idata, var_names=["velocity"], rope=rope, kind="hist", point_estimate='mean', hdi_prob=.95, ax=ax8, textsize=10);
axs.set_title("")
axs.set_xlabel("velocity");

#
ax8 = fig.add_subplot(gs[0, 4:6])
ax8.imshow(data_shifted.T, aspect="auto", origin="lower", extent=(0, length, 0, nframes))
ax8.set_ylabel("shifted frame (-)");
ax8.set_title("F (shifted frames)", loc="left")

ax9 = fig.add_subplot(gs[1, 4:6])
ax9.plot(data[:,time], alpha=0.4, label="single frame")
ax9.plot(data_mean, label="shift+avg", alpha=0.8)
ax9.plot(idata2.posterior_predictive.mean(("chain", "draw"))["y"], label="fit")
ax9.plot(hdi2["y"][:,0], "r-", alpha=0.4)
ax9.plot(hdi2["y"][:,1], "r-", alpha=0.4, label=".95 HDI")
#ax9.fill_between(x, hdi2["y"][:,0], hdi2["y"][:,1], alpha=0.2, label=".95 HDI")
#axs[1].legend();
ax9.set_xticks(np.linspace(0, length, 5))
ax9.set_yticks(np.linspace(0, np.ceil(np.max(data_mean)), 3))
ax9.set_xlabel("length (px)")
ax9.set_ylabel("avg. intensity (AU)");
ax9.set_title(r"G (shifted \& avg. frames)", loc="left")
ax9.set_xlim(0, length)
ax9.set_ylim(-2.5,7)
ax9.legend()

plt.setp(ax8.get_xticklabels(), visible=False);

#
ax10 = fig.add_subplot(gs[2, 4:5])
axs = az.plot_posterior(idata3, var_names=["snr"], kind="hist", point_estimate='mean', hdi_prob=.95, ax=ax10, textsize=10);
axs.set_title("")
ax10.set_title("H (signal-to-noise)", loc="left")
axs.set_xlabel("single frame")

ax11 = fig.add_subplot(gs[2, 5:6])
axs = az.plot_posterior(idata2, var_names=["snr"], kind="hist", point_estimate='mean', hdi_prob=.95, ax=ax11, textsize=10);
axs.set_title("")
axs.set_xlabel("avg. frames")

#
fig.align_ylabels()
# -
summary
