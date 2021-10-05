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

# %matplotlib inline

# Questions: 
# * We often record images in which the sample has not reached the measurement window yet. Basically, we record only noise. The more images with pure noise (so without sample) we include in our algorithm the worse the result. How can we decide if we should include a specific image or not? This question corresponds to the `startframe` and `endframe` variables below.
# * We find the best velocity by minimization the high density interval of a Bayesian fit function. How can we include the velocity as an uncertain parameter into the Bayesian inference problem? The data we are using depends on the velocity?
#

# + tags=[]
import numpy as np
import matplotlib.pyplot as plt
import input
import utilities
from matplotlib.patches import Rectangle
import arviz as az
from sklearn import preprocessing
import pymc3 as pm
import matplotlib.animation as animation
from IPython.display import HTML

# + tags=[]
plt.rcParams['animation.html'] = "jshtml"#'html5'
plt.rcParams['figure.dpi'] = 72

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_10ng_l/001.nd2"

channel_lower = 27
channel_upper = 27

startframe = 200
endframe = 300

# experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)
# -

# At first, we perform raw data processing (load microscopy images, cut height of images to microchannel, substract mean background image). This results in a 4D data array (height, length, frame, intensity). This corresponds to step 1 in the flowsheet.

# + tags=[]
data_raw = input.load_nd_data(inname, verbose=False)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = input.substractbackground(data_raw, background)

# + tags=[]
height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))

# +
fig, ax = plt.subplots(figsize=(10, 20*height/length))

im = plt.imshow(data_raw[:,:,0], animated=True, origin="lower", extent=(0, length*px*1e3, 0, height*px*1e3))

step = 10
def updatefig(n):
    im.set_array(data_raw[:,:,n*step])    
    ax.set_title("frame = {}".format(n*step))
    ax.set_xlabel("length (mm)")
    ax.set_ylabel("height (mm)")
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=int(nframes/step), blit=True)
plt.close()
fig.tight_layout()
ani
# -

# Now observe that the sample is almost perfectly rectangular or constant over the channel height. So we can reduce noise already by averaging the signal over the height of the channel (step 2). This results in a 3D data array (length, frame, intensity)

tmp = input.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)

x = np.linspace(0, length*px*1e3, length)

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(x, data[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[1].imshow(data.T, aspect="auto", origin="lower", extent=(0, length*px*1e3, 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");


# Observe that the sample moves through the channel with an (almost) constant velocity. If we know this velocity we can shift the samples in the different frames to be perfectly aligned (Galilean transformation). This corresponds to a rotation of the image above. If the frames are aligned we can average over all frames and reduce the noise (step 3). This results in a 2D data array (length, intensity)
#
# Obtaining this velocity is critically and difficult, esspecially for very small concentrations. Here, we start by guessing a velocity from the slope in the figure above: $v \approx \frac{\Delta x}{\Delta t}$ (because the sample movement is clearly visible here. We will see later that for lower concentrations we do not see any samples at all).

def shift_data(data, v, fps, px):
    dx =  v/fps
    data_shifted = np.zeros(data.shape)
    for i in range(0, data.shape[1]):
        shift = data.shape[0] - int(i*dx/px)%data.shape[0]
        data_shifted[:,i] = np.roll(data[:,i], shift)
    
    return data_shifted


# +
v = 2.3e-4 # this is a good guess

data_shifted = shift_data(data, v, fps, px)
data_mean = np.mean(data_shifted, axis=1).reshape(-1,1)

scaler = preprocessing.StandardScaler().fit(data_mean)
data_mean = scaler.transform(data_mean).flatten()
# -

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(x, data[:,time], label="input", alpha=0.8)
#axs[0].plot(x, data_shifted[:,time], label="shifted", alpha=0.8)
axs[0].plot(x, data_mean, label="shifted/averaged/rescaled", alpha=0.8)
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[0].legend()
axs[1].imshow(data_shifted.T, aspect="auto", origin="lower", extent=(0, length*px*1e3, 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");


# If we compare the blue signal with the orange signal we observe that the noise is greatly reduced. Let's quantify the improvement by fitting a Gaussian function to the raw and shifted/averaged signal (we know from physics that the sample distribution in the channel should be similar to a Gaussian).

# +
def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

def create_signalmodel(data, x):
    with pm.Model() as model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 5)
        cent = pm.Normal('centroid', 0.5, 0.2)
        base = pm.Normal('baseline', 0, 0.1)
        sig = pm.HalfNormal('sigma', 0.1)

        # forward model signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, base, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        return model


# +
with create_signalmodel(data_mean, x) as model:
    trace_mean = pm.sample(10000, return_inferencedata=True)

with create_signalmodel(data[:,time], x) as model:
    trace_raw = pm.sample(10000, return_inferencedata=True)
# -

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary_raw = az.summary(trace_raw, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
#summary

time = 100
fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, data[:,time], label="input, frame = {}".format(time), alpha=0.8)
axs.plot(x, data_mean, label="shifted/averaged/rescaled", alpha=0.8)
map_estimate = summary_raw.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit", alpha=0.5)
map_estimate = summary_mean.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit", alpha=0.5)
axs.set_ylabel("intensity (AU)");
axs.legend();

# +
map_estimate = summary_raw.loc[:, "mean"]
snr_before = map_estimate["amplitude"]/map_estimate["sigma_noise"]
map_estimate = summary_mean.loc[:, "mean"]
snr_after = map_estimate["amplitude"]/map_estimate["sigma_noise"]

print("improvement = snr_after/snr_before = {}".format(snr_after/snr_before))


# -

# As a result we have a reduced the noise and achieved a great improvement of the signal-to-noise ratio. We can even include a Bayesian hypothesis test or model comparison to derive at a conclusion for "sample present/not present". Therefore, we fit a noise-only model to the data too and compare the results using the Widely-applicable Information Criterion (step 4).

# +
def create_model_noiseonly(data):
    model = pm.Model()
    with model:
        # baseline only
        base = pm.Normal('baseline', 0, 0.1)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = base, sd=sigma_noise, observed = data)

        return model
    
with create_model_noiseonly(data_mean) as model:
    trace_noiseonly = pm.sample(10000, return_inferencedata=True)
# -

dfwaic = pm.compare({"sample":trace_raw, "noiseonly":trace_noiseonly}, ic="waic")
az.plot_compare(dfwaic, insample_dev=False);

dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
print(dfwaic)
az.plot_compare(dfwaic, insample_dev=False);

# As a final result, we can be very certain that a sample is present.
#
# Now let's try the same on a sample with a very low concentration.

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/003.nd2"

data_raw = input.load_nd_data(inname, verbose=False)
data_raw = input.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = input.substractbackground(data_raw, background)

# + tags=[]
height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))
# -

tmp = input.averageoverheight(data_raw)

scaler = preprocessing.StandardScaler().fit(tmp)
data = scaler.transform(tmp)

x = np.linspace(0, length*px*1e3, length)

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(x, data[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[1].imshow(data.T, aspect="auto", origin="lower", extent=(0, length*px*1e3, 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");


# We observe that the line representing the sample being transported through the channel is very weak. Now imaging that we are not able to see the line at all. Still if we would know the correct velocity of the sample we could again shift and average our frames. To find the correct velocity we now optimize over the velocity (in some way we treat the velocity as an hyperparameter). The optimization goes in the following way: choose an initial velocity, perform the shifting/averaging and fit a Gaussian to the resulting data. Calculate the high density intervals (HDI) of the fit parameters. Repeat until the HDI is small.

def functional(v):
    data_shifted = shift_data(data, v, fps, px)
    
    avg = np.mean(data_shifted, axis=1)
    
    with create_signalmodel(avg, x) as model:
        trace = pm.sample(10000, return_inferencedata=False, cores=4, progressbar=False)
        
    hdi_centroid = az.hdi(trace["centroid"])
    hdi_sigma = az.hdi(trace["sigma"])
    result = 1/2*np.sqrt(
        (hdi_centroid[1] - hdi_centroid[0])**2 \
        + (hdi_sigma[1] - hdi_sigma[0])**2) \
    
    return result


# +
from gradient_free_optimizers import *

def functional2(para):
    return -functional(para["v"])

search_space = {"v": np.arange(2e-4, 4e-4, 1e-5)} # we have some ideas about where to look for the correct velocity from physics

opt = EvolutionStrategyOptimizer(search_space)
#opt = SimulatedAnnealingOptimizer(search_space)
opt.search(functional2, n_iter=20, early_stopping={"n_iter_no_change":3})

# +
v = opt.best_para["v"]
data_shifted = shift_data(data, v, fps, px)
data_mean = np.mean(data_shifted, axis=1).reshape(-1,1)

scaler = preprocessing.StandardScaler().fit(data_mean)
data_mean = scaler.transform(data_mean).flatten()
# -

with create_signalmodel(data_mean, x) as model:
    trace_mean = pm.sample(10000, return_inferencedata=True)

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary_mean

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(x, data[:,time], label="input", alpha=0.8)
#axs[0].plot(x, data_shifted[:,time], label="shifted", alpha=0.8)
axs[0].plot(x, data_mean, label="shifted/averaged/rescaled", alpha=0.8)
map_estimate = summary_mean.loc[:, "mean"]
axs[0].plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit", alpha=0.8)
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[0].legend()
axs[1].imshow(data_shifted.T, aspect="auto", origin="lower", extent=(0, length*px*1e3, 0, nframes))
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");

with create_model_noiseonly(data_mean) as model:
    trace_noiseonly = pm.sample(10000, return_inferencedata=True)

dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
print(dfwaic)
az.plot_compare(dfwaic, insample_dev=False);









# How can we find a better velocity value?

# +
step = 20

corr = np.zeros(data.shape)
for i in range(0,data.shape[1]-step):
    corr[:,i] = np.correlate(data[:,i], data[:,i+step], "same")
    
corr[int(512/2),:] = 0 # remove auto-correlation
scaler = preprocessing.StandardScaler().fit(corr)
corr = scaler.transform(corr)
# -

time = 100
x = np.linspace(-corr.shape[0]/2, corr.shape[0]/2, corr.shape[0])
fig,axs = plt.subplots(2,1,sharex=True)
#axs[0].plot(x, corr[:,time], label="correlated")
axs[0].plot(x, np.mean(corr, axis=1), label="correlated/averaged")
axs[0].plot(x, np.mean(corr[:,startframe:endframe], axis=1), label="correlated/cut/averaged")
axs[0].set_ylabel("corr");
axs[0].legend()
axs[1].imshow(corr.T, aspect="auto",origin="lower", extent=[x[0],x[-1],0,corr.shape[1]])
axs[1].set_xlabel("Length")
axs[1].set_ylabel("Time");

# +
tmp = np.mean(corr[:,startframe:endframe], axis=1)

d = 512/2-np.argmax(tmp)
dx = d*px
t = step
dt = t/fps

v = dx/dt
print(v*1e4)
# -

data_shifted = shift_data(data, v, fps, px)

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
axs[1].text(130,250, "Samples shifted with v={}mm/s, way better.".format(v*1000), color="red",fontsize=14);


# 6. Create Bayesian model of the sample distribution at a specific time (almost Gaussian). Find the velocity of the sample by optimization. In a way, we treat the velocity as a hyperparameter (?).

# +
def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

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

def functional(v):
    data_shifted = shift_data(data, v, fps, px)
    
    avg = np.mean(data_shifted[:,startframe:endframe], axis=1)
    
    x = np.linspace(0, len(avg), len(avg))
    with create_model(avg, x) as model:
        trace = pm.sample(10000, return_inferencedata=False, cores=4, progressbar=False)
        
    hdi_centroid = az.hdi(trace["centroid"])
    hdi_sigma = az.hdi(trace["sigma"])
    result = 1/2*np.sqrt(
        (hdi_centroid[1] - hdi_centroid[0])**2 \
        + (hdi_sigma[1] - hdi_sigma[0])**2)
    
    return result


# +
from gradient_free_optimizers import *

def functional2(para):
    return -functional(para["v"])

# take the v calculated before
search_space = {"v": np.arange(v-1e-4/2, v+1e-4/2, 1e-6)}

#opt = RandomSearchOptimizer(search_space)
#opt = BayesianOptimizer(search_space)
opt = EvolutionStrategyOptimizer(search_space)
#opt = SimulatedAnnealingOptimizer(search_space)
opt.search(functional2, n_iter=20, early_stopping={"n_iter_no_change":3}, max_score=-0.1)
# -

v = opt.best_para["v"]
data_shifted = shift_data(data, v, fps, px) 
avg = np.mean(data_shifted[:,startframe:endframe], axis=1)

time = 100
fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(data[:,time], label="raw")
axs[0].plot(data_shifted[:,time], label="shifted")
axs[0].plot(np.mean(data_shifted, axis=1), label="shifted/averaged")
axs[0].plot(avg, label="shifted/cut/averaged")
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
#plt.plot(data[:,time], label="raw")
map_estimate = summary.loc[:, "mean"]
plt.plot(
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="model");
plt.legend();


