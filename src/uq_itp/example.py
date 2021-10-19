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
import helper
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
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/003.nd2"

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
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

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

time = 200
x = np.linspace(0, length*px*1e3, length)
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
axs[1].set_xlabel("lag (px)")
axs[1].set_ylabel("intensity (AU)");

#  As the velocity of the sample is pratically constant, the correlation between the frames should be almost identical (except for the noise). So we can average all the cross correlation functions to reduce the noise (step 4). 

# +
corr_mean = np.mean(corr[:,startframe:endframe], axis=1).reshape(-1, 1)
x = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

# clean the correlation data
# remove peak at zero lag
corr_mean[int(corr_mean.shape[0]/2)] = 0
#cut everything right of the middle (because we know that the velocity is positiv)
corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

x = x[0:int(corr_mean.shape[0])]

corr_singleframe = corr[0:int(corr.shape[0]/2),time]

# +
scaler = preprocessing.MinMaxScaler().fit(corr_mean)
corr_mean = scaler.transform(corr_mean).flatten()

scaler = preprocessing.MinMaxScaler().fit(corr_singleframe.reshape(-1,1))
corr_singleframe = scaler.transform(corr_singleframe.reshape(-1,1)).flatten()


# -

# To find the velocity of the sample we now fit a Gaussian function to the averaged cross correlation functions (we know from physics that the sample distribution in the channel should be similar to a Gaussian so the correlation should follow a Gaussian too).

# +
def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

def create_signalmodel(data, x):
    with pm.Model() as model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 1)
        cent = pm.Normal('centroid', 125, 50)
        base = pm.Normal('baseline', 0, 0.5)
        sig = pm.HalfNormal('sigma', 20)

        # forward model signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, base, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        # calculate the velocity and the signal-to-noise distributions from the uncertain parameters of the Gaussian
        velocity = pm.Deterministic("velocity", cent*px/(lagstep/fps)*1e6)
        snr = pm.Deterministic("snr", amp/sigma_noise)
        
        return model


# -

x *= -1
models = []
with create_signalmodel(corr_mean, x) as model:
    trace_mean = pm.sample(10000, return_inferencedata=True, cores=4)
    models.append(model)

ppc = pm.sample_posterior_predictive(trace_mean, model=models[0])
idata = az.from_pymc3(
        posterior_predictive=ppc,
        model=models[0]
    )

# Now we can display the averaged cross correlation functions together with the Gaussian fit and the .95 credible interval.

# + tags=[]
hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, corr_mean, label="averaged", alpha=0.8, color="black")
axs.plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], color="red", label="fit")
axs.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, color="blue", label=".95 HDI")
axs.legend();
axs.set_xlabel("lag (px)")
axs.set_ylabel("intensity (AU)");
# -

# Futhermore, let's have a look at the velocity and the signal-to-noise ratio calculated from the posterior distribution. The HDI interval of the velocity is narrow and centered around a physical reasonable value, and the snr is always greater than one. Taking both factors into mind we conclude that there might be a sample inside the channel and the estimated mean velocity is an accurate value (you can do the same analysis using a no-sample image). Therefore we continue the analysis.

summary_mean = az.summary(trace_mean, var_names=["amplitude", "centroid", "sigma", "baseline", "sigma_noise","velocity", "snr"])

axs = az.plot_posterior(trace_mean, var_names=["velocity", "snr"], figsize=(6,3));
axs[0].set_title("")
axs[1].set_title("")
axs[0].set_xlabel("v ($\mu m/s$)")
axs[1].set_xlabel("snr")

# Next, we use the resulting mean of the velocity distribution to perform the Gallilei transformation.

v = summary_mean["mean"]["velocity"]*1e-6
print("Mean velocity of the sample is v = {} $mum /s$.".format(v*1e6))

data_shifted = dataprep.shift_data(data, v, fps, px)

fig,axs = plt.subplots(2,1,sharex=True, figsize=(10,5))
axs[0].imshow(data.T, aspect="auto", origin="lower")
axs[0].set_ylabel("frame (-)");
axs[0].set_title("raw data")
axs[1].imshow(data_shifted.T, aspect="auto", origin="lower")
axs[1].set_xlabel("length (mm)")
axs[1].set_ylabel("frame (-)");
axs[1].set_title("shifted");

# Observe that now all the samples are aligned. As a result we can average the frames to reduce the noise.

# +
data_mean = np.mean(data_shifted[:,startframe:endframe], axis=1).reshape(-1, 1)

scaler = preprocessing.MinMaxScaler().fit(data_mean)
data_mean = scaler.transform(data_mean).flatten()
# -

# Again, we fit a Gaussian function to the intensity.

x = np.linspace(0, len(data_mean), len(data_mean))
model_wsignal = None
with create_signalmodel(data_mean, x) as model:
    trace_mean = pm.sample(10000, return_inferencedata=False, cores=4)
    model_wsignal = model

ppc = pm.sample_posterior_predictive(trace_mean, model=model_wsignal)
idata = az.from_pymc3(
        posterior_predictive=ppc,
        model=models[0]
    )

# And display the fit together with the averaged intensity data and the .95 HDI.

hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
fig,axs = plt.subplots(1,1,sharex=True, figsize=(10,5))
axs.plot(x, data_mean, label="averaged", alpha=0.8, color="black")
axs.plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], color="red", label="fit")
axs.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, color="blue", label=".95 HDI")
axs.legend();
axs.set_xlabel("length (px)")
axs.set_ylabel("intensity (AU)");

# Furthermore, let's have a look at the snr:

ax = az.plot_posterior(trace_mean, var_names=["snr"], figsize=(6,3));
ax.set_title("")
ax.set_xlabel("snr")


# Note that our calculation of the snr includes the uncertainty of the data and the estimation. The HDI of the estimated signal-to-noise ratio is between 9 and 11, which is enough to conclude that a sample is present. 

# We now include a Bayesian hypothesis test or model comparison to derive at a conclusion for "sample present/not present". Therefore, we fit a noise-only model to the data too and compare the results using the Widely-applicable Information Criterion.

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
    
model_noise = None
with create_model_noiseonly(data_mean) as model:
    trace_noiseonly = pm.sample(10000, return_inferencedata=False, cores=4)
    model_noise = model
# -

dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="loo")
print(dfwaic)
az.plot_compare(dfwaic, insample_dev=False);

# Furthermore, we calculate the Bayes factor using bridge sampling:

# +
import bayesian
ml_noiseonly = bayesian.Marginal_llk(trace_noiseonly, model_noise)
ml_signal = bayesian.Marginal_llk(trace_mean, model_wsignal)

print("Bayes factor: {}".format(np.exp(ml_signal["logml"]-ml_noiseonly["logml"])))
# -

# As a final result, we can be very certain that a sample is present.
