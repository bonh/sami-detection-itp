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

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# +
import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

import dataprep
import bayesian
import helper

# +
#mpl.style.use(['science'])
#
#mpl.rcParams['figure.dpi'] = 300
#figsize = np.array([3.42,2.20])
#mpl.rcParams["figure.figsize"] = figsize
#
#mpl.rcParams["image.origin"] = "lower"
#
#mpl.rcParams['axes.titlesize'] = 9
#mpl.rcParams['axes.labelsize'] = 8
#mpl.rcParams['lines.markersize'] = 5
#
#mpl.use("pgf")
#
#pgf_with_latex = {                      # setup matplotlib to use latex for output
#    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#    "text.usetex": True,
#    "pgf.rcfonts": False,
#    "font.family": "sans-serif",
#    "font.sans-serif": ["Arial"],
#    "pgf.preamble": "\n".join([ # plots will use this preamble
#        r"\usepackage[utf8]{inputenc}",
#        r"\usepackage[T1]{fontenc}",
#        r"\usepackage[detect-all]{siunitx}",
#        r'\usepackage{mathtools}',
#        r'\DeclareSIUnit\pixel{px}'
#        ,r"\usepackage{sansmathfonts}"
#        ,r"\usepackage[scaled=0.95]{helvet}"
#        ,r"\renewcommand{\rmdefault}{\sfdefault}"
#        ])
#    }
#
#plt.rcParams.update(pgf_with_latex)


# +
basepath = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"
concentration = "AF647_10pg_l/"
number = "005.nd2"
inname = basepath + concentration + number

channel_lower = 27
channel_upper = 27

startframe = 100
endframe = 300

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

time = 200

# +
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

height = data_raw.shape[0]
length = data_raw.shape[1]
nframes = data_raw.shape[2]
print("height = {}, length = {}, nframes = {}".format(height, length, nframes))
# -

data = dataprep.averageoverheight(data_raw)
data = dataprep.standardize(data, axis=0)

# +
v = 140e-6
data_shifted = dataprep.shift_data(data, v, fps, px)

data_fft_shifted, mask_shifted, ff_shifted = dataprep.fourierfilter(data_shifted, 30, 30, 45, True, False)
#data_fft_shifted, mask_shifted, ff_shifted = dataprep.fourierfilter(data_shifted, 3000, 3000, 0, False, False)
#data_fft_shifted = dataprep.standardize(data_fft_shifted)
data_fft_shifted = data_shifted
# -

plt.imshow(data_fft_shifted)

everynth = np.array([20, 10, 4, 2, 1])
(endframe-startframe)/everynth

# +
fig, ax = plt.subplots(len(everynth))

for i, n in enumerate(everynth):
    d = data_fft_shifted[:,startframe:endframe:n]
    print(d.shape)
    data_mean = np.mean(d, axis=1)
    
    ax[i].plot(data_mean)

# +
N = len(everynth)

j = 0

idatas = []
snr = np.zeros((2,N))
for i in range(0,N):
    j = 0
    while True:
        try:
            d = data_fft_shifted[:,startframe:endframe:everynth[i]]
            n = d.shape[1]
            print(startframe, endframe, everynth[i], n)
            
            data_mean = np.mean(d, axis=1)
            data_mean = dataprep.standardize(data_mean)
    
            x = np.arange(0, data_mean.shape[0])
            with bayesian.signalmodel(data_mean, x, artificial=True) as model:
                trace = pm.sample(2000, tune=2000, return_inferencedata=False, cores=4, target_accept=0.9)

                idata = az.from_pymc3(trace=trace, model=model) 
        
                mode = bayesian.get_mode(idata.posterior, ["snr"])[0]

            snr[:,i] = [n, mode]
            idatas.append(idata)
        except:
            if j<3:
                print("retry")
                j+=1
                continue
            else:
                break
        break
# -

snr

factor = snr[1,:]/np.sqrt(snr[0,:])
print(np.mean(factor), np.std(factor))
factor = np.mean(factor)
std = np.std(factor)

# +
plt.figure(figsize=(figsize[0], figsize[1]))

x = np.logspace(0, 3, 2100)
plt.plot(x, factor*np.sqrt(x), color="black")#'#BBBBBB')
plt.annotate(r"$\mathclap{\sim \sqrt{N_n}}$", (100, factor*np.sqrt(100)), (25,-25), textcoords='offset points', arrowprops=dict(arrowstyle="->"),horizontalalignment='center')

plt.plot(snr[0,:], snr[1,:], "o", label=r"SAMI with \SI{0.1}{\nano\gram\per\liter}", color="#EE6677")

plt.legend(markerscale=1.0, labelspacing=0.25, handletextpad=0.05)

plt.xlabel("$N_n$ (-)")
plt.ylabel("$\mathit{SNR}$ (-)");
plt.xscale("log")
#plt.xlim(np.min(x),1000)
#plt.ylim(-0.25, 10.25)
#plt.xticks([1, 10, 100, 1000])
#plt.yticks([0, 4, 8])

plt.tight_layout()
plt.savefig("snrvsframes.pdf")
# -
plt.figure()
plt.plot(idatas[0].observed_data.to_array().T)
plt.plot(idatas[-1].observed_data.to_array().T)
plt.show()

p
