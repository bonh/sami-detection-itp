# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
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
from scipy import stats, signal

# %% [markdown]
# Here, I use the data from the "ITP_AF647_5µA" experiments with zero concentration. All results are averaged over the five measurements. 
#
# NOTE: In the time correlation, a small "anti-correlation" can be observed, i.e., the time correlation function takes small negative values for time shifts larger than zero (not visible in the current axis scaling of the figure). This oberservation is consistent with the small skewness of the distribution of the pixel values towards negative values. Since this effect is really small, we can neglect it for the processing. However, it might be good to know where this comes from.

# %%
# load and prepare data
nr = [1,2,3,4,5]
data = {}
for n in nr:
    path = '../../../../03_data/ITP_AF647_5µA/AF_0ng_l/00%s.nd2' %n

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
    f, psd = signal.periodogram(processdata, axis=2, fs=46) # fs: sampling frequency = frames per second

    # average over all pixels
    avgpsd += np.mean(np.mean(psd, axis=1), axis=0)/len(nr)
np.save('psd.npy', np.array([avgpsd,f]))

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
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,12))
plt.subplots_adjust(hspace=0.35)
fig.patch.set_facecolor('white')

ticksize = 15
labelsize = 15


# histogram of pixel values (averaged over time)
skewnorm = np.load('skewnorm.npy')
norm = np.load('norm.npy')

processdata = np.concatenate([data['%s' %n].flatten() - np.mean(data['%s' %n]) for n in nr])
hist, binedges, _ = axs[0].hist(processdata, bins='auto', density=True, alpha=0.5)
axs[0].plot(skewnorm[1], skewnorm[0], label='skew normal')
axs[0].plot(norm[1], norm[0], label='normal')

axs[0].legend(fontsize=labelsize)
axs[0].set_xlabel('I', fontsize=labelsize)
axs[0].set_ylabel('P(I)', fontsize=labelsize)
axs[0].tick_params(axis='both', which='major', labelsize=ticksize)


# spectrum (averaged over all pixels)
psd = np.load('psd.npy')
axs[1].semilogy(psd[1][3:-2], psd[0][3:-2]) # remove zero frequency due to mean not exactly zero
axs[1].set_xlabel('f in Hz', fontsize=labelsize)
axs[1].set_ylabel(r'$I^2(\omega)$', fontsize=labelsize)
axs[1].set_ylim(1e-2,1e-1)
axs[1].tick_params(axis='both', which='major', labelsize=ticksize)


# time correlation averaged over all pixels
timecor = np.load('timecor.npy')

axs[2].plot(np.arange(len(timecor))/46, timecor)
axs[2].set_xlabel(r'$\Delta t$ in s', fontsize=labelsize)
axs[2].set_ylabel(r'$\langle I(0)I(\Delta t)\rangle/\langle I(0)^2\rangle$', fontsize=labelsize)
axs[2].axhline(0.0, linewidth=1, color='k')
axs[2].set_ylim(-0.01,0.1)
axs[2].tick_params(axis='both', which='major', labelsize=ticksize)


# spatial correlation (averaged over time)
cor = np.load('spatialcor.npy')
cor = cor/np.max(cor)
cor[cor<=0] = 1e-9

lagx = np.arange(-cor.shape[1]//2,cor.shape[1]//2)
lagy = np.arange(-cor.shape[0]//2,cor.shape[0]//2)
X,Y = np.meshgrid(lagx,lagy)

mappable = axs[3].pcolormesh(X, Y, cor, cmap='gray', shading='auto',\
                             norm=colors.LogNorm(vmin=1e-4, vmax=1))
axs[3].set_xlabel(r'$\Delta x$ in px', fontsize=labelsize)
axs[3].set_ylabel(r'$\Delta y$ in px', fontsize=labelsize)
cax = fig.add_axes([0.92, 0.13, 0.02, 0.14])
cb = fig.colorbar(mappable, cax=cax)
cb.set_label(r'$\langle I(\Delta x, \Delta y)I(0,0)\rangle/\langle I(0,0)^2\rangle$', fontsize=labelsize)
axs[3].tick_params(axis='both', which='major', labelsize=ticksize)


plt.savefig('noise_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('noise_v2.pdf', bbox_inches='tight')

plt.show()

# %%

# %%

# %%

# %%
