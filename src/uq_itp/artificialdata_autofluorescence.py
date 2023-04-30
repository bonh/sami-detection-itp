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

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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


# +
snr = 3
a = 1
w = 6 
alpha = 0
x = np.arange(0, 256)
   
nframes = 200

c = np.ones((nframes))*100

sigma = bayesian.fmax(a, c, w, alpha)/snr

sigma_f = sigma.eval()/10

#
sbr = 1.5
    
snr_auto = snr/sbr
a_auto = snr_auto*sigma # only true for alpha=0
w_auto = 6
alpha_auto = 0
c_auto = np.ones((nframes))*100


# +
def signal(snr, a, w, alpha, x, nframes, c, sigma_noise, sigma_f):
    data_raw = create_data(512, nframes, a, c, w, alpha, x)
    
    data_noisy = data_raw + np.random.normal(0, sigma_f, data_raw.shape) + np.random.normal(0,sigma_noise.eval(),data_raw.shape)
    return data_raw, data_noisy

data_raw, data_noisy = signal(snr, a, w, alpha, x, nframes, c, sigma, sigma_f)


# +
def auto(a_auto, w_auto, alpha_auto, x, nframes, c_auto, sigma_noise, sigma_f):
    data_raw_auto = create_data(512, nframes, a_auto, c_auto, w_auto, alpha_auto, x)
    
    data_noisy_auto = data_raw_auto + np.random.normal(0, sigma_f, data_raw_auto.shape) + np.random.normal(0,sigma_noise.eval(),data_raw_auto.shape)
    return data_raw_auto, data_noisy_auto

data_raw_auto, data_noisy_auto = auto(a_auto, w_auto, alpha_auto, x, nframes, c_auto, sigma, sigma_f)
# -

plt.plot(data_raw[:,100], label="signal with snr={}".format(snr))
plt.plot(data_raw_auto[:,100], label="autofluoresence with snr={}".format(snr_auto))
plt.plot(data_noisy[:,100], alpha=0.3)
plt.plot(data_noisy_auto[:,100], alpha=0.3);
plt.legend()

# +
from scipy.optimize import curve_fit
def sample(x, amp, cent, sig):
    return amp*np.exp(-(cent - x)**2/2/sig**2)

res = []
for i in range(0, data_noisy.shape[1]):
    popt, pcov = curve_fit(sample, x, data_noisy[:,i], p0=[1, 100, 10])
    res.append(popt[0])
    
res_auto = []
for i in range(0, data_noisy_auto.shape[1]):
    popt, pcov = curve_fit(sample, x, data_noisy_auto[:,i], p0=[1, 100, 10])
    res_auto.append(popt[0])
# -

plt.title("sbr={}, snr_s={}".format(sbr, snr))
plt.hist(res, bins="auto", alpha=0.4, label="Ampltiude of noisy signal");
plt.hist(res_auto, bins="auto", alpha=0.4, label="Ampltiude of noisy autofluoresence");
plt.legend();

# +
data_raw, data_noisy = signal(snr, a, w, alpha, x, nframes, c, sigma*1e-5, sigma_f)
data_raw_auto, data_noisy_auto = auto(a_auto, w_auto, alpha_auto, x, nframes, c_auto, sigma*1e-5, sigma_f)

res_sami = []
for i in range(0, data_noisy.shape[1]):
    popt, pcov = curve_fit(sample, x, data_noisy[:,i], p0=[1, 100, 10])
    res.append(popt[0])
    
res_sami_auto = []
for i in range(0, data_noisy_auto.shape[1]):
    popt, pcov = curve_fit(sample, x, data_noisy_auto[:,i], p0=[1, 100, 10])
    res_auto.append(popt[0])
# -

plt.title("sbr={}, snr_s={}, SAMI with images={}, samples={}".format(sbr, snr, nframes, samples))
plt.hist(res, bins="auto", alpha=0.4, label="Ampltiude of noisy signal");
plt.hist(res_auto, bins="auto", alpha=0.4, label="Ampltiude of noisy autofluoresence");
plt.hist(res_sami, bins="auto", alpha=0.4, label="Ampltiude of noisy signal after SAMI");
plt.hist(res_sami_auto, bins="auto", alpha=0.4, label="Ampltiude of noisy autofluoresence after SAMI");
plt.legend();
