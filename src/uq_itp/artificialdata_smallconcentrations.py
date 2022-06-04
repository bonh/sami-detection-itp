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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm
import arviz as az
import pymc3 as pm

import bayesian
import dataprep
import idata_crosscorrelation
import idata_sample


# -

def create_data(length, nframes, a, c, w, alpha, x, height=None): 
    data = np.zeros((length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    if height:
        for h in range(0, height):
            data[h, :, :] = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
    else:
        data = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
            
    return data


def create_images(length, height, nframes, a, c, w, alpha, x): 
    data = np.zeros((height, length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    for h in range(0, height):
        data[h, :, :] = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
            
    return data


# +
snr = 0.01
data_raw.shape
a = 1
w = 10
alpha = 5
x = np.arange(0, 512)
   
nframes = 1000
chunks = 100

c = np.ones((nframes))*100

data_raw = create_data(512, nframes, a, c, w, alpha, x)
    
a = data_raw.max()
sigma = a/snr

chunks_list = [1, 10, 100, 1000, 10000]

results = np.zeros((len(chunks_list), 4), dtype=object)

for i, chunks in enumerate(chunks_list):
    sum_ = np.zeros((512))
    for j in range(0,chunks):
        data_noisy = data_raw + np.random.normal(0,sigma,data_raw.shape)

        tmp = np.sum(data_noisy, axis=1)

        sum_ = np.add(sum_, tmp)

    data_mean = sum_/(chunks*nframes)
    
    mean = np.mean(data_mean)
    std = np.std(data_mean)
    data_mean = (data_mean - mean)/std

    x = np.linspace(0, len(data_mean), len(data_mean))
    with bayesian.signalmodel(data_mean, x, artificial=True) as model:
        trace = pm.sample(8000, tune=4000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
        
    results[i,:] = np.array([chunks*nframes, idata, mean, std], dtype=object)

# +
interpolation="none"
aspect="equal"

width = 8

height = 40
length = 512

image = create_images(length, height, 1, a, c[0], w, alpha, x)
image_noisy = image + np.random.normal(0,sigma,image.shape)

##

fig = plt.figure(constrained_layout=True, figsize=(width, width*40/512*23))

spec2 = gridspec.GridSpec(ncols=5, nrows=3+len(chunks_list), figure=fig)

ax1 = fig.add_subplot(spec2[0, 0:3])
ax1.imshow(image, origin="lower", extent=(0, length/2, 0, height), aspect=aspect, interpolation=interpolation);
ax1.set_title("true distribution", loc="left")
ax1.set_yticks(np.linspace(0,height,3))
ax1.set_xticklabels('')

ax2 = fig.add_subplot(spec2[1, 0:3], sharex=ax1)
ax2.imshow(image_noisy, origin="lower", extent=(0, length/2, 0, height), aspect=aspect, interpolation=interpolation);
ax2.set_title("true distribution+noise ($\mathit{{snr}} ={}$)".format(snr), loc="left")
ax2.set_yticks(np.linspace(0,height,3))

ax3 = fig.add_subplot(spec2[2, 0:3], sharex=ax1)
bla = data_raw[:,0]
tmp = bla + np.random.normal(0,sigma,bla.shape)

mean = np.mean(tmp)
std = np.std(tmp)
bla = (bla-mean)/std
tmp = (tmp-mean)/std

ax3.plot(x, bla, label="true")
ax3.plot(x,tmp, label="noisy")

axins = ax3.inset_axes([0.5, 0.1, 0.2, 1.1])
axins.plot(x, bla)
# sub region of the original image
x1, x2, y1, y2 = 80, 140, 1.01*np.min(bla), np.max(bla)*1.01
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax3.indicate_inset_zoom(axins, edgecolor="black")
ax3.legend()

###
ax = fig.add_subplot(spec2[1:3, 3:5])
ax.imshow(data_noisy.T, aspect="auto", interpolation=interpolation)
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.set_xlabel("length (px)")
ax.set_ylabel("frames")
ax.set_xticks(np.linspace(0,512,5))

offset = 3
for i in range(0, len(chunks_list)):
    if i is not len(chunks_list)-1:
        ax = fig.add_subplot(spec2[i+3, 0:3], sharex=ax1)
    else:
        ax = fig.add_subplot(spec2[i+3, 0:3])
    
    ax.plot(x, results[i,1].posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red")
    ax.plot(x, results[i,1].observed_data.to_array().T, label="averaged", color="blue", alpha=0.3)
    
    hdi = az.hdi(results[i,1].posterior_predictive, hdi_prob=.95)
    ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")

    ax.set_title("{:.0e} frames included in average".format(results[i,0]), loc="left")
    
    ax.set_ylim(-2.5,8)
    ax.set_xlim(0,512/2)
    ax.set_xticks(np.linspace(0,512/2,5))
    
    rope_sigma = [7, 12]
    ref_snr = 3
    value2 = bayesian.check_rope(results[i,1].posterior["sigma"], rope_sigma)
    value3 = bayesian.check_refvalue(results[i,1].posterior["snr"], ref_snr)
    detected = (value2>.95) and (value3>.95)
    
    if detected:
        ax.text(40,5, "detected")
    else:
        ax.text(40,5, "not detected")
    
    ###
    ax2 = fig.add_subplot(spec2[i+3, 3])
    tmp = az.plot_posterior(results[i,1], "sigma", ax=ax2, kind="hist", point_estimate="mode", hdi_prob=.95, textsize=8)
    tmp.set_title("")
    if i == 0:
        ax2.set_title("Marginal posteriors", loc="left")
    
    ax3 = fig.add_subplot(spec2[i+3, 4])
    tmp = az.plot_posterior(results[i,1], "snr", ax=ax3, kind="hist", point_estimate="mode", hdi_prob=.95, textsize=8)
    tmp.set_title("")

# last axis
ax.plot(x, (data_raw[:,0]-results[-1,2])/results[-1,3], "--", color="black", label="true")
ax.set_xlim(0,512/2)
ax.set_xticks(np.linspace(0,512/2,5))
ax.legend();

ax2.set_xlabel("spread (px)")
ax3.set_xlabel("snr")

###

# -


