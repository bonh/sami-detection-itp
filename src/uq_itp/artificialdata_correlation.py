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


# +
a = 1
w = 10
alpha = 5

v_px = 2.31

l = 512
x = np.arange(0,l)
   
nframes = 500
frames = np.arange(0, nframes)

c0 = 2*w
c = c0 + v_px*frames

data_raw = create_data(512, nframes, a, c, w, alpha, x)
data_raw.shape
# -

plt.imshow(data_raw.T, origin="lower")

(np.argmax(data_raw[:,100])-np.argmax(data_raw[:,90]))/10

data = data_raw

# +
lagstep = 50

corr = np.zeros((data.shape[0], data.shape[1]-lagstep))
for i in range(0,data.shape[1]-lagstep):
    corr[:,i] = np.correlate(data[:,i], data[:,i+lagstep], "same")

corr = corr[0:int(l/2), :] 
print(corr.shape)
# -

x = np.arange(-256,0)
#x = np.linspace(-int(l/2), int(l/2), l)

plt.imshow(corr.T, origin="lower")

plt.plot(x, corr[:,100])

-x[np.argmax(corr[:,100])]/(lagstep)

# +
corr_mean = np.mean(corr[:,0:-lagstep], axis=1)
corr_mean[int(corr_mean.shape[0]/2)] = 0

x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
x_lag = x_lag[0:int(corr_mean.shape[0])]
# -

plt.plot(x_lag, corr_mean)

-x_lag[np.argmax(corr_mean)]/lagstep


