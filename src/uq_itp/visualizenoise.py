# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# File: visualizenoise.py
#
# Author:       Lukas Hecht (lukas.hecht@pkm.tu-darmstadt.de)
#
# Date:         March 10, 2022

# %%
import dataprep, helper
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats, signal

# %%
mpl.style.use(['science', "bright"])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([4.5,2.20])
mpl.rcParams["figure.figsize"] = figsize

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['lines.markersize'] = 5

mpl.use("pgf")

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ,r"\usepackage{sansmathfonts}"
        ,r"\usepackage[scaled=0.95]{helvet}"
        ,r"\renewcommand{\rmdefault}{\sfdefault}"
        ])
    }

plt.rcParams.update(pgf_with_latex)

# %% [markdown]
# Here, I use the data from the "ITP_AF647_5µA" experiments with zero concentration. All results are averaged over the five measurements. 
#
# NOTE: In the time correlation, a small "anti-correlation" can be observed, i.e., the time correlation function takes small negative values for time shifts larger than zero (not visible in the current axis scaling of the figure). This oberservation is consistent with the small skewness of the distribution of the pixel values towards negative values. Since this effect is really small, we can neglect it for the processing. However, it might be good to know where this comes from.

# %%
# load and prepare data
nr = [1,2,3,4,5]
data = {}
for n in nr:
    path = '/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/AF647_0ng_l/00%s.nd2' %n

    channel_lower = 27
    channel_upper = 27

    data_raw = helper.raw2images(path, (channel_lower, channel_upper))
    d = dataprep.standardize(data_raw)
    data['%s' %n] = d

# %%
# histogram of pixel values
processdata = np.concatenate([data['%s' %n].flatten() - np.mean(data['%s' %n]) for n in nr])
hist, binedges, _ = plt.hist(processdata, bins='auto', density=True, alpha=0.5)
bins = (binedges[1:]+binedges[:-1])/2
np.save('histogram.npy', np.array([hist, bins]))

# skew normal fit
p1, p2, p3 = stats.skewnorm.fit(processdata)
mean, var, skew, kurt = stats.skewnorm.stats(p1, p2, p3, moments='mvsk')
print(mean, var, skew, kurt)
np.save('skewnorm_params.npy', np.array([mean, var, skew, kurt]))
skewnorm = stats.skewnorm.pdf(bins, p1, p2, p3)
np.save('skewnorm.npy', np.array([skewnorm, bins]))

# normal fit
p1, p2 = stats.norm.fit(processdata)
mean, var = stats.norm.stats(p1, p2, moments='mv')
print(mean, var)
np.save('norm_params.npy', np.array([mean, var]))
norm = stats.norm.pdf(bins, p1, p2)
np.save('norm.npy', np.array([norm, bins]))

# %%
# spectrum
avgpsd = 0
for n in nr:
    processdata = (data['%s' %n]-np.mean(data['%s' %n], axis=2)[:,:,None])/np.std(data['%s' %n], axis=2)[:,:,None] # remove mean over time
    f, psd = signal.welch(processdata, axis=2, fs=46, nperseg=459) # fs: sampling frequency = frames per second

    # average over all pixels
    avgpsd += np.mean(np.mean(psd, axis=1), axis=0)/len(nr)
np.save('psd.npy', np.array([avgpsd,f]))
psd

# %%
# spatial correlation (only use each 20th frame to save time - still takes about 10 min!)
avgcor = 0
for n in nr:
    processdata = (data['%s' %n]-np.mean(np.mean(data['%s' %n], axis=0), axis=0)[None,None,:]) # remove mean of each frame
    cor = 0
    m = 0
    for i in range(0,data['%s' %n].shape[-1],20):
        cor += signal.correlate2d(processdata[:,:,i], processdata[:,:,i])
        m += 1
        print(m, end=' ')

    avgcor += cor/m/len(nr)

np.save('spatialcor.npy', avgcor)

# %%
# correlation in time averaged over all pixels
avgtimecor = 0
timecors = []
for n in nr:
    processdata = (data['%s' %n]-np.mean(data['%s' %n], axis=2)[:,:,None])/np.std(data['%s' %n], axis=2)[:,:,None] # remove mean over time
    timecor = 0
    for x in range(processdata.shape[1]):
        for y in range(processdata.shape[0]):
            c = signal.correlate(processdata[y,x,:], processdata[y,x,:], mode='same')
            timecor += c/np.max(c)

    avgtimecor += timecor[len(timecor)//2:]/processdata.shape[0]/processdata.shape[1]/len(nr)
    
np.save('timecor.npy', avgtimecor)

# %%
# plot
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(figsize[0], figsize[0]))

# histogram of pixel values (averaged over time)
skewnorm = np.load('skewnorm.npy')
norm = np.load('norm.npy')

processdata = np.concatenate([data['%s' %n].flatten() - np.mean(data['%s' %n]) for n in nr])
hist, binedges, _ = axs[0].hist(processdata, bins=150, density=True, alpha=0.5)
axs[0].set_title("A: Distribution of intensity values", loc="left", weight="bold")

#bins = (binedges[1:]+binedges[:-1])/2
#norm = stats.norm.pdf(bins, 0, 1)
axs[0].plot(norm[1], norm[0], label=r'Normal')
axs[0].plot(skewnorm[1], skewnorm[0], label='Skewed')

axs[0].legend(fontsize=8)
axs[0].set_xlabel('Intensity (-)')
axs[0].set_ylabel('Probability')
axs[0].tick_params(axis='both', which="both", labelleft=False)

# spectrum (averaged over all pixels)
psd = np.load('psd.npy')
axs[1].set_title("B: Power spectral density", loc="left", weight="bold")
axs[1].semilogy(psd[1], psd[0]) 
axs[1].set_xlabel('Frequency (\si{\hertz})')
axs[1].set_ylabel("Log PSD")
#axs[1].set_ylim(1e-2,1e-1)
axs[1].set_xlim(0,23)
axs[1].set_yticks([1e-2, 1e-1])
#axs[1].tick_params(axis='both', which="both", labelleft=False,labelbottom=False)

# time correlation averaged over all pixels
timecor = np.load('timecor.npy')


axs[2].set_title("C: Temporal autocorrelation", loc="left", weight="bold")
axs[2].plot(np.arange(len(timecor))/46, timecor)
axs[2].set_xlabel("Time shift (\si{\second})")
axs[2].set_ylabel("Correlation")
axs[2].axhline(0.0, linewidth=1, color='k')
axs[2].set_ylim(-0.01,0.1)
#axs[2].tick_params(axis='both', which="both", labelleft=False,labelbottom=False)


# spatial correlation (averaged over time)
cor = np.load('spatialcor.npy')
cor = cor/np.max(cor)
cor[cor<=0] = 1e-9

lagx = np.arange(-cor.shape[1]//2,cor.shape[1]//2)
lagy = np.arange(-cor.shape[0]//2,cor.shape[0]//2)
X,Y = np.meshgrid(lagx,lagy)

axs[3].set_title("D: Spatial autocorrelation", loc="left", weight="bold")
mappable = axs[3].imshow(cor, cmap="gray_r", norm=colors.LogNorm(vmin=1e-4, vmax=1),aspect="auto", extent=[lagx[0],lagx[-1],lagy[0],lagy[-1]])
axs[3].set_xticks([-256, 0, 256])
axs[3].set_yticks([-34, 0, 34])
axs[3].set_xlabel("Lengthwise shift (\si{\pixel})")
axs[3].set_ylabel("Heightwise shift (\si{\pixel})")
cb = fig.colorbar(mappable)
#axs[3].tick_params(axis='both', which="both", labelleft=False,labelbottom=False)
cb.set_label("Log correlation")
cb.set_ticks([1e-4,1])

fig.align_ylabels()
fig.tight_layout(h_pad=0.1)
fig.savefig("noise.pdf")

# %% [markdown]
# ### 

# %%

# %%

# %%
