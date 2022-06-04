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
w = 5
alpha = 5

v_px = 0.1

l = 512
x = np.arange(0,l)
   
nframes = 2000
frames = np.arange(0, nframes)

c0 = 2*w
c = c0 + v_px*frames

data_raw = create_data(512, nframes, a, c, w, alpha, x)
data_raw.shape
# -

data_shifted = dataprep.shift_data(data_raw, v_px, None, None)
data_shifted.shape

# +
fig, axs = plt.subplots(2,1)

axs[0].plot(x, data_raw)

axs[1].plot(x, data_shifted)

fig.tight_layout()
# -

tmp = np.max(data_raw, axis=1)
plt.plot(tmp)

tmp = np.max(data_shifted, axis=1)
plt.plot(tmp)

plt.plot(data_raw[:,1000])
plt.plot(data_shifted[:,10])


