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
from tqdm.notebook import tqdm

import bayesian
import dataprep


# -

def create_images(length, height, nframes, a, c, w, alpha, c_add, alpha_add, x): 
    data = np.zeros((height, length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    for h in range(0, height):
        data[h, :, :] = bayesian.model_sample(a, cc+c_add[h], w[h], alpha+alpha_add[h], xx).eval().T
            
    return data


# +
length = 512
height = 40
nframes = 500

a = 1
w_base = 30
alpha = 5

c_init = -100

v = 230e-6

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

x = np.linspace(0, length, length)

n = np.arange(0, nframes)     
c = c_init+n*v/(fps*px)
 
h = np.linspace(0, height, height)
w = (w_base+(h-height/2)**2/20)
c_add = -(h-height/2)**2/10
alpha_add = -(h-height/2)**2/50

data_raw = create_images(length, height, nframes, a, c, w, alpha, c_add, alpha_add, x)
# -

data_raw = np.roll(data_raw, 10, axis=0)

fig, ax = plt.subplots(4,1, sharex=True)
ax[0].imshow(data_raw[:,:,50], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[1].imshow(data_raw[:,:,100], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[2].imshow(data_raw[:,:,150], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].imshow(data_raw[:,:,200], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].set_xticks(np.linspace(0, length, 5))
fig.tight_layout()

plt.plot(data_raw[30,:,150])
plt.plot(data_raw[20,:,150])
plt.plot(data_raw[10,:,150])

# +
snr = 0.5 # a/sigma
a = data_raw.max() # standardized
sigma = a/snr

data_noisy = data_raw + np.random.normal(0,sigma,data_raw.shape)
data_noisy = dataprep.standardize(data_noisy)
# -

fig, ax = plt.subplots(4,1, sharex=True)
ax[0].imshow(data_noisy[:,:,50], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[1].imshow(data_noisy[:,:,100], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[2].imshow(data_noisy[:,:,150], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].imshow(data_noisy[:,:,200], origin="lower", extent=(1, length, 1, height), aspect="auto");
ax[3].set_xticks(np.linspace(0, length, 5))
fig.tight_layout()

plt.plot(data_noisy[20,:,150]);
plt.plot(data_raw[20,:,150]);

data = data_noisy[10,:,:]
plt.imshow(data.T)

# +

lagstep = 30 
corr = dataprep.correlate_frames(data, lagstep)

plt.imshow(corr)
# -

corr_mean = np.mean(corr[:,0:300], axis=1)
plt.plot(corr_mean)

# +
import idata_crosscorrelation

fps = 46 # frames per second (1/s)
px = 1.6e-6# size of pixel (m/px)  

lagstep = 30
    
idata_cross, min_, max_ = idata_crosscorrelation.main(None, None, lagstep, px, fps, data_raw=data_noisy[30,:,:])
# -

import arviz as az
az.plot_posterior(idata_cross, "velocity")

# +
v = 230e-6#bayesian.get_mode(idata_cross.posterior, ["velocity"])[0]*1e-6
print(v*1e6)
min_, max_ = 30, 180

data_shifted = dataprep.shift_data(data, v, fps, px)

data_mean = np.mean(data_shifted[:,min_:max_], axis=1)
data_mean = dataprep.standardize(data_mean)
data_mean = np.roll(data_mean, 200)
# -

plt.plot(data_mean)

# +
import pymc3 as pm
import arviz as az

steps = np.arange(0,height,1)

idatas = np.zeros((2, len(steps)), dtype=object)
i = -1
for h in steps:
    i += 1
    j = 0
    print(h)
    data = data_noisy[h,:,:]
    data_shifted = dataprep.shift_data(data, v, fps, px)

    data_mean = np.mean(data_shifted[:,min_:max_], axis=1)
    data_mean = dataprep.standardize(data_mean)
    data_mean = np.roll(data_mean, 200)
    
    x = np.linspace(0, len(data_mean), len(data_mean))
    while True:
        try:
            with bayesian.signalmodel(data_mean, x) as model:
                trace = pm.sample(tune=2000, return_inferencedata=False, cores=4, target_accept=0.9)
    
                ppc = pm.fast_sample_posterior_predictive(trace, model=model)
                idatas[:,i] = np.array([h, az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model)], dtype=object)         
        except Exception as e:
            print(e)
            if j<3:
                print("retry")
                j+=1
                continue
            else:
                break
                
        break

# +
reconstructed = np.zeros((data_raw[:,:,0].shape))
hdi_low = np.zeros((data_raw[:,:,0].shape))
hdi_high = np.zeros((data_raw[:,:,0].shape))

for i in range(0, len(idatas[0,:])):
    h = idatas[0,i]
    d = idatas[1,i].posterior_predictive.mean(("chain", "draw")).y
    reconstructed[h,:] = d
    
    hdi = az.hdi(idatas[1,i].posterior_predictive, hdi_prob=.95)
    hdi_low[h,:] = hdi["y"][:,0]
    hdi_high[h,:] = hdi["y"][:,1]

# +
interpolation="none"
aspect="equal"


time = 119

reconstructed = dataprep.standardize(reconstructed)
data_raw_ = dataprep.standardize(data_raw[:,:,time])
data_noisy_ = dataprep.standardize(data_noisy[:,:,time])

lines = steps[5::15]
fig, axs = plt.subplots(3+len(lines),1, sharex=True, figsize=(8,3+(3+len(lines))*8*height/length))
axs[0].imshow(data_raw_, origin="lower", extent=(0, length, 0, height), aspect=aspect, interpolation=interpolation);
axs[0].set_title("true distribution for frame {}".format(time), loc="left")
axs[1].imshow(data_noisy_, origin="lower", extent=(0, length, 0, height), aspect=aspect, interpolation=interpolation);
axs[1].set_title("true distribution+noise for frame {} ($\mathit{{snr}} ={}$)".format(time, snr), loc="left")
axs[2].imshow(reconstructed, origin="lower", extent=(0, length, 0, height), aspect=aspect, interpolation=interpolation);
axs[2].set_title("reconstructed distribution using all frames", loc="left")
#pos = axs[3].imshow(dataprep.standardize(np.sqrt((data_raw_-reconstructed)**2)), origin="lower", extent=(1, length, 1, height), aspect="auto", interpolation=interpolation, );
#fig.colorbar(pos, ax=axs[3])

axs[0].set_yticks(np.linspace(0,height,3))
axs[1].set_yticks(np.linspace(0,height,3))
axs[2].set_yticks(np.linspace(0,height,3))

x = np.arange(1, length+1)
i=3
axs[i].set_title("comparison (slices)", loc="left")
for h in lines[::-1]:
    d = dataprep.standardize(reconstructed[h,:])
    axs[i].plot(x, d, label="true")
    d = dataprep.standardize(data_raw[h,:,time])
    axs[i].plot(x, d, label="reconstructed")
    d = dataprep.standardize(data_noisy[h,:,time])
    axs[i].plot(x, d, label="noisy (standardized)", alpha=0.5)
    
    axs[i].fill_between(x, hdi_low[h,:], hdi_high[h,:], alpha=0.2, label=".95 HDI")
    
    axs[i].set_yticks([])
    
    axs[i].text(120, 3, "at height {} px".format(h), horizontalalignment="center",);
    axs[i].set_ylim(-3,5)
    i+=1
    
axs[-1].set_xlim(0,512)
axs[-1].legend(loc="lower right")
axs[-1].set_xticks(np.linspace(0, length, 5))
axs[-1].set_xlabel("length (px)")
fig.tight_layout()
# -


