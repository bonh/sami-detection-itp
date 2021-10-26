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

# todo:
# * include (semi-)automatic endframe and startframe decision

# #### Steps:
# 0. Required input:
#     * Define the ROPEs (region of practical equivalence) around the Null hypothesis for the spread of the sample distribution and the sample velocity (depends on Physics). 
#     * Define position of the channel in the images.
#     * Define start and end frames for the averages (see todo above!).
#     * Define frames-per-second (fps) and resolution (px).
# 1. Load microscopy images, cut height of images to microchannel, and substract mean background image -> results in a 4-tuple (height, length, frame, intensity).
# 2. Average the intensity of the 4-tuple from step 1 over the height of the image -> results in a 3-tuple (length, frame, averaged intensity).
# 3. Calculate cross correlation between frame $n$ and frame $n+L$ -> results in a 3-tuple (lag, frame, correlated intensity).
# 4. Average the correlated intensity of the 3-tuple from step 3 over all the frames -> results in a 2-tuple (lag, averaged correlated intensity).
# 5. Fit a Gaussian function (defined as a Bayesian model) to the averaged correlated intensity of the 2-tuple from step 4.
# 6. Calculate the distribution of the sample velocity from the posterior of the centroid of the Gaussian estimated in step 9, the lag from step 3 as well as the fps and px.
# 7. Test if more than 95 % of the marginal posteriors of the spread and velocity calculated in step 5 are inside the ROPEs -> Detected (yes/no)?
# 8. If a sample is detected, use the mean of the velocity distribution calculated in step 4 to perform the Gallilei transformation of the original 3-tuple from step 2 -> (length; frame; shifted, averaged intensity).
# 9. Average the shifted and height averaged intensity of the 3-tuple from step 7 over all frames -> results in a 2-tuple (length; averaged, shifted, averaged intensity).
# 10. Fit a skewed Gaussian function (defined as a Bayesian model) to the frame averaged, shifted, height averaged intensity of the 2-tuple from step 8.
# 11. Calculate the distribution of the signal-to-noise ratio from the posterior of the amplitude and the noise standard deviation estimated in step 9.

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# + tags=[]
import numpy as np
import matplotlib.pyplot as plt
import dataprep
import helper
import utilities
import bayesian

import arviz as az
from sklearn import preprocessing
import pymc3 as pm

from IPython.display import HTML

import matplotlib as mpl

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
# -

mpl.rcParams['animation.html'] = "jshtml"#'html5'
#mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathtools}'
plt.style.use(['science', 'vibrant'])

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/002.nd2"

# to cut images to channel
channel_lower = 27
channel_upper = 27

# to cut images to the one containing the sample
startframe = 150
endframe = 250

# from experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

# rope
rope_sigma = (7,12)
rope_velocity = (210,230)
rope = {'sigma': [{'rope': rope_sigma}]
        , 'velocity': [{'rope': rope_velocity}]}
# -

# At first, we perform raw data processing (load microscopy images, cut height of images to microchannel, substract mean background image). This results in a 4-tuple (height, length, frame, intensity). This corresponds to step 1 in the flowsheet.

# + tags=[]
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))
# -

x_mm = np.linspace(0, length*px*1e3, length)

# +
fig, ax = plt.subplots(figsize=(10, 30*height/length))

im = plt.imshow(data_raw[:,:,0], animated=True, origin="lower", extent=(0, length*px*1e3, 0, height*px*1e3))
ax.set_xlabel("length (mm)")
ax.set_ylabel("height (mm)")

step = 10
def updatefig(n):
    im.set_array(data_raw[:,:,n*step])    
    ax.set_title("frame = {}".format(n*step))

    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=int(nframes/step), blit=True)
plt.close()
fig.tight_layout()
ani
# -

# Observe that the sample is almost perfectly rectangular (or constant over the image height). So we can reduce noise already by averaging the signal over the height of the image (step 2). This results in a 3-tuple (length, frame, intensity).

tmp = dataprep.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)


# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


time = 200
fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(x_mm, data[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[0].set_ylim(-5, 5)
axs[1].imshow(data.T, aspect="auto", origin="lower", extent=(x_mm[0], x_mm[-1], 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");
fig.align_ylabels()
set_size(6,4)
fig.savefig("step2.pdf", bbox_inches='tight')

# Observe that the sample moves through the channel with an (almost) constant velocity. However, we do not know this velocity, esspecially for low concentrations. But due to the velocity being constant, the cross-correlation between every frame $n$ and $n+L$, where $L$ is some specified lag, should be almost identical (up to the random noise of course). This calculation of the single cross correlation functions (step 3) and the subsequent averaging (step 4) results in a 2-tuple (lag, correlated intensity).

lagstep = 30 
corr = dataprep.correlate_frames(data, lagstep)

scaler = preprocessing.StandardScaler().fit(corr)
corr = scaler.transform(corr)

# +
corr_mean = np.mean(corr[:,startframe:endframe], axis=1).reshape(-1, 1)
x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

# clean the correlation data
# remove peak at zero lag
corr_mean[int(corr_mean.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

x_lag = x_lag[0:int(corr_mean.shape[0])]
# -

scaler = preprocessing.MinMaxScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()

fig,axs = plt.subplots(3,1,sharex=True)#, figsize=(10, 6)
axs[0].plot(x_lag, corr[0:int(corr.shape[0]/2),time])
#axs[0].set_title(r"$\text{{frame}}\[200\]*\text{{frame}}[${}$]$".format(lagstep))
axs[0].set_title(r"single corr (add better title)")
axs[0].set_ylabel("corr.\nintensity (AU)", va='center');
axs[0].set_ylim(-4, 4)
axs[1].imshow(corr[0:int(corr.shape[0]/2),:].T, aspect="auto", origin="lower", extent=[x_lag[0],x_lag[-1],0,corr.T.shape[1]])
#axs[1].set_ylabel(r"${{(I(n)\star I(n+{}))(\tau)}}$".format(lagstep));
axs[1].set_ylabel(r"corr");
axs[2].plot(x_lag, corr_mean)
axs[2].set_ylabel("averaged corr")
axs[-1].set_xlabel("lag $\\tau$ (px)");
fig.align_ylabels()
set_size(6,6)
fig.savefig("step34.pdf", bbox_inches='tight')

# Based on this 2-tuple we decide if a sample is present or not. Therefore we fit a Gaussian function to the averaged cross correlation functions (step 5). (Note, that the cross correlation of a skewed Gaussian with a shifted skewed Gaussian is a Gaussian(?).)

with bayesian.signalmodel_correlation(corr_mean, -x_lag, px, lagstep, fps) as model:
    trace = pm.sample(return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc = pm.fast_sample_posterior_predictive(trace, model=model)
    idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
    summary = az.summary(idata)

# + tags=[]
hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
fig,axs = plt.subplots(1,1,sharex=True)
axs.plot(x_lag, corr_mean, label="averaged", alpha=0.8)
axs.plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit")
axs.fill_between(x_lag, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI")
axs.legend();
axs.set_xlabel("lag (px)")
axs.set_ylabel("intensity (AU)");
set_size(6,2)
fig.savefig("step5.pdf", bbox_inches='tight');
# -

# To decide if a sample is present we define a ROPE (region of practical equivalence) around the Null hypothesis for the spread of the Gaussian fit as well as the velocity calculated from the maximum of the Gaussian function. The more tight the ROPEs the more accurate the estimation has to be (we are more certain). If more than 95 % of the marginal posteriors are inside the ROPEs we decide that a sample is present (step 6). In this way we include the full uncertainty of our estimation in our decision. To visualize this approach, we plot the marginal posteriors together with the ROPEs below.

axs = az.plot_posterior(idata, var_names=["sigma", "velocity"], rope=rope, kind="hist", point_estimate='mean', hdi_prob="hide");
axs[0].set_title("")
axs[1].set_title("")
axs[0].set_xlabel("$\sigma$")
axs[1].set_xlabel("$v$");
plt.savefig("step6.pdf", bbox_inches='tight');

detected = (bayesian.check_rope(idata.posterior["sigma"], rope_sigma) > .95) and (bayesian.check_rope(idata.posterior["velocity"], rope_velocity) > .95)
print("Sample detected: {}".format(detected))

# If we decide that a sample is present we can use the resulting mean of the velocity distribution to perform the Gallilei transformation of the original 3-tuple (length, frame, intensity) (step 7) and average over all frames (step 8). This results again in a 2-tuple (length, averaged intensity). We fit a skewed Gaussian to the averaged intensit (step 9).

v = summary["mean"]["velocity"]*1e-6
print("Mean velocity of the sample is v = {} $mum /s$.".format(v*1e6))

data_shifted = dataprep.shift_data(data, v, fps, px)

# +
data_mean = np.mean(data_shifted[:,startframe:endframe], axis=1).reshape(-1, 1)

scaler = preprocessing.MinMaxScaler().fit(data_mean)
data_mean = scaler.transform(data_mean).flatten()
# -

x = np.linspace(0, len(data_mean), len(data_mean))
with bayesian.signalmodel(data_mean, x) as model:
    trace = pm.sample(return_inferencedata=False, cores=4, target_accept=0.9)
      
    ppc = pm.fast_sample_posterior_predictive(trace, model=model)
    idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
    summary = az.summary(idata)

# +
fig, axs = plt.subplots(2,1,sharex=True)
axs[0].imshow(data_shifted.T, aspect="auto", origin="lower", extent=(x_mm[0], x_mm[-1], 0, nframes))
axs[0].set_ylabel("shifted frame (-)");

hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
axs[1].plot(x_mm, data_mean, label="averaged", alpha=0.8)
axs[1].plot(x_mm, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit")
axs[1].fill_between(x_mm, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI")
axs[1].legend();
axs[-1].set_xlabel("length (px)")
axs[1].set_ylabel("averaged intensity (AU)");
fig.align_ylabels()
set_size(6,4)
fig.savefig("step789.pdf", bbox_inches='tight')
# -

# Finally, we can examine the full distribution including all the uncertainty of the signal-to-noise ratio calculated from the amplitude and the noise standard deviation (step 10).

ax = az.plot_posterior(idata, var_names=["snr"], rope=rope, kind="hist", point_estimate='mean', hdi_prob="hide");
ax.set_title("")
ax.set_xlabel("$snr$")
plt.savefig("step10.pdf", bbox_inches='tight');
