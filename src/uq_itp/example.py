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

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2
# -

# Questions: 
# * We often record images in which the sample has not reached the measurement window yet. Basically, we record only noise. The more images with pure noise (so without sample) we include in our algorithm the worse the result. How can we decide if we should include a specific image or not? This question corresponds to the `startframe` and `endframe` variables below.
# * To decide if a sample is present or not (detected yes/no?) we fit a signal model and a pure noise model and calcualte the waic (widely applicable information criterion). Is this a convincing and robust criterion for our decision?

# + tags=[]
import numpy as np
import matplotlib.pyplot as plt
import dataprep
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

# to cut images to channel
channel_lower = 27
channel_upper = 27

# to cut images to the one containing the sample
startframe = 200
endframe = 300

# from experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)
# -

# At first, we perform raw data processing (load microscopy images, cut height of images to microchannel, substract mean background image). This results in a 4D data array (height, length, frame, intensity). This corresponds to step 1 in the flowsheet.

# + tags=[]
data_raw = dataprep.load_nd_data(inname, verbose=False)
data_raw = dataprep.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = dataprep.substractbackground(data_raw, background)

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

tmp = dataprep.averageoverheight(data_raw)

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

# Observe that the sample moves through the channel with an (almost) constant velocity. If we know this velocity we can shift the samples in the different frames to be perfectly aligned (Galilean transformation). This corresponds to a rotation of the image above. If the frames are aligned we can average over all frames and reduce the noise. This results in a 2D data array (length, intensity).
#
# However, obtaining this velocity is critically and difficult, esspecially for very small concentrations. Therefore, we further process the data by taking the cross correlation of each frame with a frame a few steps further (step 3).

lagstep = 30 
corr = dataprep.correlate_frames(data, lagstep)

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(corr[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[1].imshow(corr.T, aspect="auto", origin="lower")
axs[1].set_xlabel("lag ???")
axs[1].set_ylabel("???");

#  As the velocity of the sample is pratically constant, the correlation between the frames should be almost identical (except for the noise). Then we average all the cross correlation functions to reduce the noise (step 4). 

# +
corr_mean = np.mean(corr[:,50:200], axis=1).reshape(-1, 1)
x = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

# clean the correlation data
# remove peak at zero lag
corr_mean[int(corr_mean.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
x = x[0:int(corr_mean.shape[0])]
# -

corr_singleframe = corr[0:int(corr.shape[0]/2),time]

# +
scaler = preprocessing.MinMaxScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()

scaler = preprocessing.MinMaxScaler().fit(corr_singleframe.reshape(-1,1))
corr_singleframe = scaler.transform(corr_singleframe.reshape(-1,1)).flatten()
# -

plt.plot(x, corr_singleframe, label="single frame")
plt.plot(x, corr_mean, label="averaged")
plt.xlabel("lag (pixel)")
plt.ylabel("intensity (AU)")
plt.legend()


# If we compare the blue signal with the orange signal we observe that the noise is greatly reduced. Let's quantify the improvement by fitting a Gaussian function to the single frame signal and the averaged signal (we know from physics that the sample distribution in the channel should be similar to a Gaussian so the correlation should follow a Gaussian tooplt.plot(corr_singleframe)).

# +
def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

def create_signalmodel(data, x):
    with pm.Model() as model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 1)
        cent = pm.Normal('centroid', 125, 50)
        base = pm.Normal('baseline', 0, 0.5)
        sig = pm.HalfNormal('sigma', 10)

        # forward model signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, base, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        return model


# +
x *= -1
with create_signalmodel(corr_mean, x) as model:
    trace_mean = pm.sample(20000, return_inferencedata=True, cores=4)

with create_signalmodel(corr_singleframe, x) as model:
    trace_singleframe = pm.sample(20000, return_inferencedata=True, cores=4)
# -

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary_singleframe = az.summary(trace_singleframe, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])

fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, corr_mean, label="averaged", alpha=0.8)
axs.plot(x, corr_singleframe, label="single", alpha=0.8)
map_estimate = summary_singleframe.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit, single", alpha=0.5)
map_estimate = summary_mean.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit, mean", alpha=0.5)
axs.legend();
axs.set_xlabel("lag (px)")
axs.set_ylabel("intensity (AU)")

# +
map_estimate = summary_singleframe.loc[:, "mean"]
snr_before = map_estimate["amplitude"]/map_estimate["sigma_noise"]
map_estimate = summary_mean.loc[:, "mean"]
snr_after = map_estimate["amplitude"]/map_estimate["sigma_noise"]

print("snr_before = {}\nsnr_after = {}".format(snr_before, snr_after))
print("improvement = snr_after/snr_before = {}".format(snr_after/snr_before))
# -

# As a result we have a reduced the noise and achieved a great improvement of the signal-to-noise ratio.
#
# Let's calculate the velocity of the sample from the fit of the averaged cross corelation functions.

# +
map_estimate = summary_mean.loc[:, "mean"]
corr_mean_fit = model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x)

dx = map_estimate["centroid"]*px #0.15e-3 # m
dt = lagstep/fps # s

v = dx/dt # mm/s
print("sample velocity = {} mm/s".format(v*1e3))
# -

# We use the resulting velocity to perform the Gallilei transformation.

# +
data_shifted = dataprep.shift_data(data, v, fps, px)

time = 100
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].imshow(data.T, aspect="auto", origin="lower")
axs[0].set_ylabel("frame (-)");
axs[0].set_title("raw data")
axs[1].imshow(data_shifted.T, aspect="auto", origin="lower")
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");
axs[1].set_title("shifted");


# -

# Observe that now all the samples are aligned.

# We now include a Bayesian hypothesis test or model comparison to derive at a conclusion for "sample present/not present". Therefore, we fit a noise-only model to the data too and compare the results using the Widely-applicable Information Criterion (step 4).

# +
def create_model_noiseonly(data):
    model = pm.Model()
    with model:
        # baseline only
        base = pm.Normal('baseline', 0, 0.5)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = base, sd=sigma_noise, observed = data)

        return model
    
with create_model_noiseonly(corr_mean) as model:
    trace_noiseonly = pm.sample(10000, return_inferencedata=True)
# -

dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
print(dfwaic)
az.plot_compare(dfwaic, insample_dev=False);

# As a final result, we can be very certain that a sample is present.

# Now let's try the same on a sample with a very low concentration.

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/003.nd2"

data_raw = dataprep.load_nd_data(inname, verbose=False)
data_raw = dataprep.cuttochannel(data_raw, channel_lower, channel_upper)
background = np.mean(data_raw[:,:,0:10],axis=2)
data_raw = dataprep.substractbackground(data_raw, background)

# + tags=[]
height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))
# -

tmp = dataprep.averageoverheight(data_raw)

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

# We observe that the line representing the sample being transported through the channel is very weak. Now imaging that we are not able to see the line at all. Still if we would know the correct velocity of the sample we could again shift and average our frames. Otherwise, we again have to calculate the crosscorrelation and average the correlation functions.

corr = dataprep.correlate_frames(data, 30)

time = 200
fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].plot(corr[:,time])
axs[0].set_title("frame = {}".format(time))
axs[0].set_ylabel("intensity (AU)");
axs[1].imshow(corr.T, aspect="auto", origin="lower")
axs[1].set_xlabel("lag ???")
axs[1].set_ylabel("???");

# +
corr_mean = np.mean(corr[:,100:250], axis=1).reshape(-1, 1)
x = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

# clean the correlation data
# remove peak at zero lag
corr_mean[int(corr_mean.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
x = x[0:int(corr_mean.shape[0])]
# -

x *= -1

corr_singleframe = corr[0:int(corr.shape[0]/2),time]

# +
scaler = preprocessing.MinMaxScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()

scaler = preprocessing.MinMaxScaler().fit(corr_singleframe.reshape(-1,1))
corr_singleframe = scaler.transform(corr_singleframe.reshape(-1,1)).flatten()
# -

plt.plot(x, corr_singleframe, label="single frame")
plt.plot(x, corr_mean, label="averaged")
plt.legend();

# +
with create_signalmodel(corr_mean, x) as model:
    trace_mean = pm.sample(20000, return_inferencedata=True, cores=4)

with create_signalmodel(corr_singleframe, x) as model:
    trace_singleframe = pm.sample(20000, return_inferencedata=True, cores=4)
# -

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])
summary_singleframe = az.summary(trace_singleframe, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise"])

fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, corr_mean, label="averaged", alpha=0.8)
axs.plot(x, corr_singleframe, label="single", alpha=0.8)
map_estimate = summary_singleframe.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit, single", alpha=0.5)
map_estimate = summary_mean.loc[:, "mean"]
axs.plot(x, 
    model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x), label="fit, mean", alpha=0.5)
axs.legend();

# +
map_estimate = summary_singleframe.loc[:, "mean"]
snr_before = map_estimate["amplitude"]/map_estimate["sigma_noise"]
map_estimate = summary_mean.loc[:, "mean"]
snr_after = map_estimate["amplitude"]/map_estimate["sigma_noise"]

print("snr_before = {}\nsnr_after = {}".format(snr_before, snr_after))
print("improvement = snr_after/snr_before = {}".format(snr_after/snr_before))

# +
map_estimate = summary_mean.loc[:, "mean"]
corr_mean_fit = model_signal(map_estimate["amplitude"], map_estimate["centroid"], map_estimate["sigma"], map_estimate["baseline"], x)

dx = map_estimate["centroid"]*px #0.15e-3 # m
dt = lagstep/fps # s

v = dx/dt # mm/s
print("sample velocity = {} mm/s".format(v*1e3))
# -

with create_model_noiseonly(corr_mean) as model:
    trace_noiseonly = pm.sample(10000, return_inferencedata=True)

dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
print(dfwaic)
az.plot_compare(dfwaic, insample_dev=False);
