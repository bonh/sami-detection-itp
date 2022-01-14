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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np
import pymc3 as pm
import arviz as az

import helper
import dataprep
import bayesian

# +
mpl.style.use(['science'])

mpl.rcParams['figure.dpi'] = 150
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
figsize = np.array([500*px,400*px])
mpl.rcParams["figure.figsize"] = figsize

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathtools}'

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['axes.titlesize'] = 10
# -

labeller = az.labels.MapLabeller(var_name_map={"sigma":"spread", "centroid":"position 1", "deltac":"shift", "sigmanoise":r"$\sigma_n$"})

# +
basepath = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"
concentration = "AF647_1pg_l/"

#basepath = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/"
#concentration = "AF_10ng_l/"
number = "003.nd2"
inname = basepath + concentration + number

channel_lower = 27
channel_upper = 27

startframe = 100
endframe = 300#startframe+200

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

sigma_mean = 10
rope_sigma = (5,17)
rope_velocity = (117,184)
snr_ref = 3

time = 150

rope = {'sigma': [{'rope': rope_sigma}]
        , 'velocity': [{'rope': rope_velocity}]}
# -

data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))

data = dataprep.averageoverheight(data_raw)
data = dataprep.standardize(data, axis=0)

# +
fig, axs = plt.subplots(2,1, figsize=(figsize[0]*2, figsize[1]), sharex=True)
axs[0].imshow(data_raw[:,:,time]);
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("height (px)");

axs[1].plot(data[:,time])
axs[1].set_xlabel("length (px)")
axs[1].set_ylabel("intensity (-)")
                        
fig.tight_layout();
# -

if False:
    x = np.arange(0, d.shape[0])
    with bayesian.signalmodel(data[:,time], x) as model:
        trace = pm.sample(10000, tune=4000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata_single = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 

        hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

    fig = plt.figure(constrained_layout=True, figsize=(figsize[0]*2, figsize[1]))
    gs = GridSpec(1, 4, figure=fig)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(x, data[:,time], "b", alpha=0.5, label="data");
    ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
    ax.set_xlabel("length (px)")
    ax.set_ylabel("intensity (-)")
    ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color="r");
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    az.plot_posterior(idata_single, "sigma", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax)
    ax.set_title("")
    ax.set_xlabel(r"spread (px)");

    ax = fig.add_subplot(gs[0, 3])
    az.plot_posterior(idata_single, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax, ref_val=3)
    ax.set_title("")
    ax.set_xlabel(r"snr");

    #fig.tight_layout()

data_fft, mask, ff = dataprep.fourierfilter(data, 100, 40/4, -45, True, True)
data_fft = dataprep.standardize(data_fft, axis=0)

# +
fig, axs = plt.subplots(1,4, figsize=2*figsize, sharey=True)

axs[0].imshow(data.T)
axs[0].set_xlabel("length (px)")
axs[0].set_ylabel("frame (-)");

axs[1].imshow(np.log(np.abs(ff.T)), cmap="gray")

axs[2].imshow(np.log(mask.T), cmap="gray");

axs[3].imshow(data_fft.T)
axs[3].set_xlabel("length (px)")

fig.tight_layout();
# -

N = 8
deltalagstep = 5
lagstepstart = 30
x_lag, corr_mean_combined = dataprep.correlation(data_fft, startframe, endframe, lagstepstart=lagstepstart, deltalagstep=deltalagstep, N=N)

# +
fig, axs = plt.subplots(1,2, figsize=(figsize[0]*2, figsize[1]), sharey=True)

axs[0].plot(x_lag, corr_mean_combined.T[:,0]);
axs[0].set_xlabel("lag (px)")
axs[0].set_ylabel("intensity (-)");

axs[1].plot(x_lag, corr_mean_combined.T);
axs[1].set_xlabel("lag (px)");

fig.tight_layout()
# -

with bayesian.signalmodel_correlation(corr_mean_combined.T, -np.array([x_lag,]*N).T, px, deltalagstep, lagstepstart, fps, artificial=True) as model:
    trace = pm.sample(5000, tune=4000, return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc = pm.fast_sample_posterior_predictive(trace, model=model)
    idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
    #idata = az.from_pymc3(trace=trace, model=model)
    #summary = az.summary(idata, var_names=["b", "c", "amplitude", "sigma", "centroid", "deltac", "sigmanoise", "velocity"])
    
    hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)

az.plot_pair(
    idata
    ,var_names=["b", "c", "amplitude", "sigma", "centroid", "sigmanoise"]#, "deltac"
    ,kind="kde"
    ,figsize=2*figsize
    ,labeller=labeller
    ,marginals=True
    #,point_estimate="mode",
);

# +
fig, axs = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
axs[0].plot(x_lag, corr_mean_combined.T, "b", alpha=0.5, label="data");
axs[0].plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
axs[0].set_xlabel("lag (px)")
axs[0].set_ylabel("intensity (-)")

for i in range(0,N):
    axs[0].fill_between(x_lag, hdi["y"][:,i,0], hdi["y"][:,i,1], alpha=0.2, label=".95 HDI", color="r");

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend([handles[0], handles[N], handles[-1]], [labels[0], labels[N], labels[-1]]);

az.plot_posterior(idata, ["velocity"], hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=axs[1])
axs[1].set_title("")
axs[1].set_xlabel(r"velocity ($\mu m$)");

fig.tight_layout()

# +
v = bayesian.get_mode(idata.posterior, ["velocity"])[0]*1e-6
print("Mode of velocity of the sample is v = {} $microm /s$.".format(v*1e6))
data_shifted = dataprep.shift_data(data, v, fps, px)

data_mean = np.mean(data_shifted[:,startframe:endframe], axis=1)

data_mean = dataprep.standardize(data_mean)
# -

data_fft_shifted, mask_shifted, ff_shifted = dataprep.fourierfilter(data_shifted, 30, 30, 0, True, False)
data_fft_shifted = dataprep.standardize(data_fft_shifted)

# +
fig, axs = plt.subplots(1,4, figsize=2*figsize, sharey=True)

axs[0].imshow(data_shifted.T)
axs[0].set_xlabel("length (px)")
axs[0].set_ylabel("frame (-)");

axs[1].imshow(np.log(np.abs(ff_shifted.T)), cmap="gray")

axs[2].imshow(np.log(mask_shifted.T), cmap="gray");

axs[3].imshow(data_fft_shifted.T)
axs[3].set_xlabel("length (px)")

fig.tight_layout();

# +
data_mean = np.mean(data_fft_shifted[:,startframe:endframe], axis=1)
data_mean = dataprep.standardize(data_mean)

plt.figure(figsize=figsize)
plt.plot(data_mean);
plt.xlabel("length (px)")
plt.ylabel("intensity (-)");
# -

x = np.arange(0, data_mean.shape[0])
with bayesian.signalmodel(data_mean, x) as model:
    trace3 = pm.sample(2000, tune=8000, return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc3 = pm.fast_sample_posterior_predictive(trace3, model=model)
    idata3 = az.from_pymc3(trace=trace3, posterior_predictive=ppc3, model=model) 
    
    hdi3 = az.hdi(idata3.posterior_predictive, hdi_prob=.95)

az.plot_pair(
    idata3
    ,var_names=["b", "c", "amplitude", "sigma", "centroid", "alpha", "sigmanoise"]
    ,kind="kde"
    ,figsize=figsize*2
    ,labeller=labeller
    ,marginals=True
    #,point_estimate="mode",
);

# +
fig = plt.figure(constrained_layout=True, figsize=(figsize[0]*2, figsize[1]))
gs = GridSpec(1, 4, figure=fig)

ax = fig.add_subplot(gs[0, 0:2])
ax.plot(x, data_mean, "b", alpha=0.5, label="data");
ax.plot(x, idata3.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_xlabel("length (px)")
ax.set_ylabel("intensity (-)")
ax.fill_between(x, hdi3["y"][:,0], hdi3["y"][:,1], alpha=0.2, label=".95 HDI", color="r");
ax.legend()

ax = fig.add_subplot(gs[0, 2])
az.plot_posterior(idata3, "sigma", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax)
ax.set_title("")
ax.set_xlabel(r"spread (px)");

ax = fig.add_subplot(gs[0, 3])
az.plot_posterior(idata3, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax, ref_val=3)
ax.set_title("")
ax.set_xlabel(r"snr");

#fig.tight_layout()

# +
fig, ax = plt.subplots(figsize=figsize/2)
ax.axis('off')
    
if bayesian.check_rope(idata.posterior["velocity"], rope_velocity)>.95 and bayesian.check_rope(idata3.posterior["sigma"], rope_sigma)>.95 and bayesian.check_refvalue(idata3.posterior["snr"], snr_ref)>.95:  
    rect = mpl.patches.Rectangle((0, 0), 1, 1, facecolor='lightgreen')
    ax.add_patch(rect)
    ax.text(0.5, 0.5, 'Sample present!', horizontalalignment='center', verticalalignment='center')
else:
    rect = mpl.patches.Rectangle((0, 0), 1, 1, facecolor='lightcoral')
    ax.add_patch(rect)
    ax.text(0.5, 0.5, 'Sample absent!', horizontalalignment='center', verticalalignment='center')

# +
columns = 6
rows = 5

fig = plt.figure(figsize=(figsize[0]*columns, figsize[1]*rows*1-2))
gs = GridSpec(rows, columns, figure=fig)

axmain = fig.add_subplot(gs[0, 0:3])
axmain.set_title("A: Cut to channel and background subtracted (frame = {})".format(time), loc="left")
axmain.imshow(data_raw[:,:,time]);
axmain.set_yticks(np.linspace(0, height, 3))
axmain.set_ylabel("height (px)");
axmain.tick_params(labelbottom = False)

#
ax = fig.add_subplot(gs[1, 0:3], sharex=axmain)
ax.set_title("B: Height averaged (frame = {})".format(time), loc="left")
ax.plot(data[:,time])
ax.set_ylabel("intensity (-)")

#
ax = fig.add_subplot(gs[2, 0])
ax.set_title("C: Fourier filter", loc="left")
ax.imshow(data.T, aspect="auto")
ax.set_ylabel("frame (-)");
ax.tick_params(labelbottom = False)

#
#ax = fig.add_subplot(gs[2, 1], sharey=ax)
#ax.imshow(np.log(np.abs(ff.T)), cmap="gray")
#ax.tick_params(labelleft = False)

#
#ax = fig.add_subplot(gs[2, 2], sharey=ax)
#ax.imshow(np.log(mask.T), cmap="gray");
#ax.tick_params(labelleft = False)

#
ax = fig.add_subplot(gs[2, 1], sharey=ax)
ax.imshow(data_fft.T, aspect="auto")
ax.tick_params(labelleft = False)
ax.tick_params(labelbottom = False)

#
ax = fig.add_subplot(gs[2, 2])
ax.set_title("D: Cross correlated, averaged, and fitted", loc="left")
ax.plot(x_lag, corr_mean_combined.T[:,0], "b", alpha=0.5, label="data");
ax.plot(x_lag, corr_mean_combined.T[:,1:], "b", alpha=0.3);
ax.plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_xlabel("lag (px)")
ax.tick_params(labelleft = False)

for i in range(0,N):
    ax.fill_between(x_lag, hdi["y"][:,i,0], hdi["y"][:,i,1], alpha=0.2, label=".95 HDI", color="r");

handles, labels = ax.get_legend_handles_labels()
#ax.legend([handles[0], handles[N], handles[-1]], [labels[0], labels[N], labels[-1]]);

#
ax = fig.add_subplot(gs[3, 0])
ax.set_title("E: Shifted data and Fourier filter", loc="left")
ax.imshow(data_shifted.T, aspect="auto")
ax.set_xlabel("length (px)")
ax.set_ylabel("frame (-)");

#
#ax = fig.add_subplot(gs[3, 1])
#ax.imshow(np.log(np.abs(ff_shifted.T)), cmap="gray")

#
#ax = fig.add_subplot(gs[3, 2])
#ax.imshow(np.log(mask_shifted.T), cmap="gray");

#
ax = fig.add_subplot(gs[3, 1])
ax.imshow(data_fft_shifted.T, aspect="auto")
ax.set_xlabel("length (px)")
ax.tick_params(labelleft = False)

#
ax = fig.add_subplot(gs[3, 2])
ax.set_title("F: Averaged and fitted", loc="left")
ax.plot(x, data_mean, "b", alpha=0.5, label="data");
ax.plot(x, idata3.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_xlabel("length (px)")
ax.fill_between(x, hdi3["y"][:,0], hdi3["y"][:,1], alpha=0.2, label=".95 HDI", color="r");
ax.tick_params(labelleft = False)
#ax.legend()

#
ax = fig.add_subplot(gs[4, 0])
ax.set_title("G: Marginal posteriors and detection", loc="left")
az.plot_posterior(idata, ["velocity"], hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax, textsize=8)
ax.set_title("")
ax.set_xlabel(r"velocity ($\mu m$)");

#
ax = fig.add_subplot(gs[4, 1])
az.plot_posterior(idata3, "sigma", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax,  textsize=8)
ax.set_title("")
ax.set_xlabel(r"spread (px)");

#
ax = fig.add_subplot(gs[4, 2])
az.plot_posterior(idata3, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", rope=rope, ax=ax, ref_val=snr_ref,  textsize=8)
ax.set_title("")
ax.set_xlabel(r"snr");


fig.align_ylabels()
fig.tight_layout()
