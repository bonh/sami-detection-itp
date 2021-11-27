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
# File: noise_analyis.py
#
# Author:       Lukas Hecht (hecht@fkp.tu-darmstadt.de)
#
# Date:         November 26, 2021

# %%
import helper, dataprep
import numpy as np
import matplotlib.pyplot as plt
import scipy

channel_lower = 27
channel_upper = 27

inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_10ng_l/001.nd2"
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))

fig, axs = plt.subplots(2,1)

s = dataprep.standardize(data_raw[:,:,:30].flatten())

x = np.linspace(0, len(s), len(s))

from scipy import signal
f, Pxx_den = signal.periodogram(s)
axs[0].semilogy(f[1:], Pxx_den[1:]) # remove zero frequency due to mean not exactly zero
axs[0].set_xlabel('frequency')
axs[0].set_ylabel('power')
axs[0].set_xticks([])
axs[0].set_yticks([])

_, bins, _ = axs[1].hist(s, bins="auto", density=1, alpha=0.5)
p1, p2, p3 = scipy.stats.skewnorm.fit(s)
mean, var, skew, kurt = scipy.stats.skewnorm.stats(p1, p2, p3, moments='mvsk')
#print(mean, var, skew, kurt)
best_fit_line = scipy.stats.skewnorm.pdf(bins, p1, p2, p3)
axs[1].plot(bins, best_fit_line);

#values = scipy.stats.skewnorm.rvs(p1, p2, p3, 100000)
#axs[1].hist(values, bins="auto", density=1, alpha=0.5)

p1, p2 = scipy.stats.norm.fit(s)
best_fit_line = scipy.stats.norm.pdf(bins, p1, p2)
axs[1].plot(bins, best_fit_line);

axs[1].set_xlabel('intensity')
axs[1].set_ylabel('density')
axs[1].set_xticks([]);
axs[1].set_yticks([]);

# %% [markdown] heading_collapsed=true
# ## Import Modules

# %% hidden=true
import os

import matplotlib.pyplot as plt
import numpy as np

from nd2reader import ND2Reader
from matplotlib import animation, rc
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import erf

import dataprep

# %% [markdown] heading_collapsed=true
# ## Load Data

# %% code_folding=[] hidden=true
#path = "/home/hecht/Documents/18_industrial/MerckLab_Isotachophoresis/03_data/BackgroundNoise/" 
path = "../../../../03_data/BackgroundNoise/"

nth = 2
fps = 46./nth # frames per second


#=== Leading Electrolyte (LE) data ===============================================
file = 'noise_LE_001'
name = os.path.join(path, 'LE/', file)
inname = "{}.nd2".format(name)

LEdata = dataprep.load_nd_data(inname, nth=nth)

#=== Trailing Electrolyte (TE) data ==============================================        
file = 'noise_TE_001'
name = os.path.join(path, 'TE/', file)
inname = "{}.nd2".format(name)

TEdata = dataprep.load_nd_data(inname, nth=nth)

# %% [markdown] heading_collapsed=true hidden=true
# ### Plot Raw Signal

# %% code_folding=[] hidden=true
# plot signal
fig = plt.figure(figsize=(10,4))

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

imLE = ax1.imshow(LEdata[:,:,0], cmap='Greys_r')
imTE = ax2.imshow(TEdata[:,:,0], cmap='Greys_r')

ax1.set_title('LE')
ax2.set_title('TE')

def init():
    imLE.set_data(LEdata[:,:,0])
    imTE.set_data(TEdata[:,:,0])
    return [imLE,imTE]
    
def animate(k):
    imLE.set_data(LEdata[:,:,k])
    imTE.set_data(TEdata[:,:,k])
    return [imLE,imTE]
    
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=LEdata.shape[2]//4, interval=200, blit=True)

rc('animation', html='jshtml')
anim

# %% [markdown] heading_collapsed=true
# ## Pre-Processing

# %% hidden=true
# cut to channel
channel = [27,27]

LEdata = dataprep.cuttochannel(LEdata, channel[0], channel[1])
TEdata = dataprep.cuttochannel(TEdata, channel[0], channel[1])

# %% hidden=true
# substract background
LEback = np.mean(LEdata, axis=2)
TEback = np.mean(TEdata, axis=2)

LEfinal = dataprep.substractbackground(LEdata, LEback)
TEfinal = dataprep.substractbackground(TEdata, TEback)

# %% [markdown] heading_collapsed=true hidden=true
# ### Plot Pre-Processed Signal

# %% hidden=true
# plot signal
fig = plt.figure(figsize=(14,2))

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

imLE = ax1.imshow(LEfinal[:,:,0], cmap='Greys_r')
imTE = ax2.imshow(TEfinal[:,:,0], cmap='Greys_r')

ax1.set_title('LE')
ax2.set_title('TE')

def init():
    imLE.set_data(LEfinal[:,:,0])
    imTE.set_data(TEfinal[:,:,0])
    return [imLE,imTE]
    
def animate(k):
    imLE.set_data(LEfinal[:,:,k])
    imTE.set_data(TEfinal[:,:,k])
    return [imLE,imTE]
    
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=LEdata.shape[2]//4, interval=200, blit=True)

rc('animation', html='jshtml')
anim

# %% hidden=true
print(np.min(LEfinal),np.max(LEfinal))
print(np.min(TEfinal),np.max(TEfinal))

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ## Statistics

# %% hidden=true
print('------- Mean values -------')
print('LE Mean: ', np.mean(LEfinal))
print('TE Mean: ', np.mean(TEfinal))

# %% hidden=true
print('------- Standard Deviation -------')
print('LE std: ', np.std(LEfinal))
print('TE std: ', np.std(TEfinal))


# %% [markdown] hidden=true
# ### Time-Averaged Histogram of Pixel Values

# %% code_folding=[0, 5, 8] hidden=true
def histogram(data, nbins):
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    bins = (bin_edges[1:]+bin_edges[:-1])/2
    return hist, bins

def gauss(x, a, sig, mu):
    return a/np.sqrt(2*np.pi*sig**2) * np.exp(-(x-mu)**2/(2*sig**2))

def skewGaussian(x, mu, sigma, alpha):
    '''
    Skew normal distribution.
    
    INPUT:
        mu : mean
        sigma: width parameter
        alpha: skewness parameter (simple Gaussian for alpha=0.0)
        
    RETURN:
        skew normal
    '''
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2) * (1 + erf(alpha*(x-mu)/np.sqrt(2)/sigma))


# %% code_folding=[] hidden=true
# histograms
LEhist, LEbins = histogram(LEfinal, nbins=5000)
TEhist, TEbins = histogram(TEfinal, nbins=5000)

# %% code_folding=[] hidden=true
# Gauss fit
LEpopt, pcov = curve_fit(skewGaussian, LEbins, LEhist, p0=[0.0, 20, 1.0], sigma=1/np.abs(LEbins))
LEperr = np.sqrt(np.diag(pcov))

TEpopt, pcov = curve_fit(skewGaussian, TEbins, TEhist, p0=[0.0, 20, 1.0])
TEperr = np.sqrt(np.diag(pcov))

# %% code_folding=[] hidden=true
# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,16), sharex=False)

axs[0].plot(LEbins, LEhist, label='LE histogram', color='gold')
axs[0].plot(LEbins, skewGaussian(LEbins, LEpopt[0], LEpopt[1], LEpopt[2]),\
            label='LE skew normal fit', color='k', linestyle='--')

axs[1].plot(TEbins, TEhist, label='TE histogram', color='gold')
axs[1].plot(TEbins, skewGaussian(TEbins, TEpopt[0], TEpopt[1], TEpopt[2]),\
            label='TE skew normal fit', color='k', linestyle='--')

axs[0].set_title(r'Skew normal: $\mu$=%1.3f$\pm$%1.3f, $\sigma$=%1.3f$\pm$%1.3f, $\alpha$=%1.3f$\pm$%1.3f'\
                 %(LEpopt[0],LEperr[0],LEpopt[1],LEperr[1],LEpopt[2],LEperr[2]), fontsize=15)
axs[1].set_title(r'Skew normal: $\mu$=%1.3f$\pm$%1.3f, $\sigma$=%1.3f$\pm$%1.3f, $\alpha$=%1.3f$\pm$%1.3f'\
                 %(TEpopt[0],TEperr[0],TEpopt[1],TEperr[1],TEpopt[2],TEperr[2]), fontsize=15)

axs[0].set_xlabel('pixel value', fontsize=25)
axs[0].set_ylabel('probability density', fontsize=25)
axs[1].set_xlabel('pixel value', fontsize=25)
axs[1].set_ylabel('probability density', fontsize=25)

axs[0].tick_params(axis='both', labelsize=15)
axs[1].tick_params(axis='both', labelsize=15)

axs[0].legend(fontsize=25)
axs[1].legend(fontsize=25)

#axs[0].semilogy()
#axs[1].semilogy()

plt.show()

# %% hidden=true
print('LE Mean: ', LEpopt[0]+LEpopt[1]*LEpopt[2]/(1+LEpopt[2]**2)*np.sqrt(2/np.pi))
print('TE Mean: ', TEpopt[0]+TEpopt[1]*TEpopt[2]/(1+TEpopt[2]**2)*np.sqrt(2/np.pi))


# %% [markdown] hidden=true
# The Gaussian does not fit exactly for some reason. I also tried to fit a skew normal distribution, which does fit better but results in a wrong mean value.
#
# Question: Are there different processes that cause the noise values to not be Gaussian distributed?

# %% hidden=true

# %% hidden=true

# %% hidden=true

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ## Spatial Correlation

# %% hidden=true
def correlate(x,v,mode='full'):
    corr = signal.correlate(x, v, mode=mode)
    corr /= np.max(corr)
    return corr


# %% hidden=true
# correlation (averaged over y coordinate and time)
LEcor = 0
for i in range(LEfinal.shape[2]):
    for n in range(LEfinal.shape[0]):
        c = correlate(LEfinal[n,:,i],LEfinal[n,:,i], mode='full')
        LEcor += c
        
LEcor = LEcor[len(LEcor)//2:]/LEfinal.shape[0]/LEfinal.shape[2]

TEcor = 0
for i in range(TEfinal.shape[2]):
    for n in range(TEfinal.shape[0]):
        c = correlate(TEfinal[n,:,i],TEfinal[n,:,i], mode='full')
        TEcor += c
        
TEcor = TEcor[len(TEcor)//2:]/TEfinal.shape[0]/TEfinal.shape[2]

# %% hidden=true
plt.plot(LEcor, label='LE')
plt.plot(TEcor, label='TE')
plt.legend()
plt.xlabel(r'$k$ in px')
plt.ylabel(r'$\langle I(n)I(n+k)\rangle/\langle I(n)^2\rangle$')
plt.show()

# %% hidden=true
# correlation (averaged over x coordinate and time) - this takes very long !!!
LEcor = 0
for i in range(LEfinal.shape[2]):
    for n in range(LEfinal.shape[1]):
        c = correlate(LEfinal[:,n,i],LEfinal[:,n,i], mode='full')
        LEcor += c
        
LEcor = LEcor[len(LEcor)//2:]/LEfinal.shape[1]/LEfinal.shape[2]

TEcor = 0
for i in range(TEfinal.shape[2]):
    for n in range(TEfinal.shape[1]):
        c = correlate(TEfinal[:,n,i],TEfinal[:,n,i], mode='full')
        TEcor += c
        
TEcor = TEcor[len(TEcor)//2:]/TEfinal.shape[1]/TEfinal.shape[2]

# %% code_folding=[] hidden=true
plt.plot(LEcor, label='LE')
plt.plot(TEcor, label='TE')
plt.legend()
plt.xlabel(r'$k$ in px')
plt.ylabel(r'$\langle I(n)I(n+k)\rangle/\langle I(n)^2\rangle$')
plt.show()

# %% [markdown] hidden=true
# The correlation of Gaussian random variables for comparison:

# %% hidden=true
cor = 0
nav = 500
for i in range(nav):
    x = np.random.normal(size=100)
    cor += correlate(x,x)
plt.plot(cor[len(cor)//2:]/nav)
plt.show()

# %% hidden=true
# 2d correlation
time = 10
cor2d = signal.correlate2d(LEfinal[:,:,time], LEfinal[:,:,time])

# %% code_folding=[] hidden=true
fig = plt.figure(figsize=(15,10))

lagx = np.arange(-cor2d.shape[1]//2,cor2d.shape[1]//2)
lagy = np.arange(-cor2d.shape[0]//2,cor2d.shape[0]//2)
X,Y = np.meshgrid(lagx,lagy)

plt.pcolormesh(X, Y, cor2d/np.max(cor2d), cmap='gray', shading='auto')
cbar = plt.colorbar()
plt.show()

# %% [markdown] hidden=true
# There is no spatial correlation between the pixels.

# %% [markdown] heading_collapsed=true
# ## Time Correlation

# %% [markdown] hidden=true
# Time correlation calculated for each pixel and averaged over all pixels.

# %% hidden=true
LEcor = 0
for x in range(LEfinal.shape[1]):
    for y in range(LEfinal.shape[0]):
        LEcor += correlate(LEfinal[y,x,:], LEfinal[y,x,:])
        
LEcor = LEcor[len(LEcor)//2:]/LEfinal.shape[0]/LEfinal.shape[1]

TEcor = 0
for x in range(TEfinal.shape[1]):
    for y in range(TEfinal.shape[0]):
        TEcor += correlate(TEfinal[y,x,:], TEfinal[y,x,:])
        
TEcor = TEcor[len(TEcor)//2:]/TEfinal.shape[0]/TEfinal.shape[1]

# %% hidden=true
plt.plot(np.arange(len(LEcor))/fps, LEcor, label='LE')
plt.plot(np.arange(len(TEcor))/fps, TEcor, label='TE')
plt.legend()
plt.xlabel(r'$\Delta t$ in s')
plt.ylabel(r'$\langle I(t)I(t+\Delta t)\rangle/\langle I(t)^2\rangle$')
plt.show()

# %% [markdown] hidden=true
# LE: Small correlation in time.
#
# TE: Uncorrelated.

# %% hidden=true
# just for one pixel (no averages)
x = 10; y=50
cor = correlate(LEfinal[y,x,:], LEfinal[y,x,:])
plt.plot(np.arange(len(cor[len(cor)//2:]))/fps, cor[len(cor)//2:])
plt.show()

# %% code_folding=[] hidden=true

# %% code_folding=[] hidden=true

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ## Spectrum

# %% [markdown] hidden=true
# The spectrum is calculated for each pixel separately and then averaged over all pixels.

# %% code_folding=[] hidden=true
# calculate mean spectrum (mean over the spectrum of each pixel)
yf = np.fft.rfftn(LEfinal, axes=[2])

LENbin = len(yf[0,0])
LExf = np.fft.fftfreq(LENbin, 1/fps)

print(yf.shape, LENbin)

LEspec = np.mean(np.mean(np.abs(yf[:,:,0:LENbin//2]), axis=1), axis=0)

yf = np.fft.rfftn(TEfinal, axes=[2])

TENbin = len(yf[0,0])
TExf = np.fft.fftfreq(TENbin, 1/fps)

print(yf.shape, TENbin)

TEspec = np.mean(np.mean(np.abs(yf[:,:,0:TENbin//2]), axis=1), axis=0)

# %% code_folding=[] hidden=true
# plot
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(LExf[0:LENbin//2], LEspec, label='LE')
ax.plot(TExf[0:TENbin//2], TEspec, label='TE')
ax.tick_params(axis='both', labelsize=15)
ax.set_xlabel('$f$ in Hz', fontsize=25)
ax.set_ylabel('spectrum in a.u.', fontsize=25)
ax.legend(fontsize=25)
ax.semilogx()
plt.show()

# %% [markdown] hidden=true
# Not pure white noise ...

# %% code_folding=[0] hidden=true

# %% code_folding=[0] hidden=true

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ## Some Old Stuff

# %% hidden=true
# spatial correlation (in y direction; averaged over x direction)
y0 = [0,25,50,75,100,125]

# LE ======================================================================
fig, ax = plt.subplots(figsize=(12,8))

for y in y0:
    cor = 0
    for i in range(LEdata2.shape[1]):
        cor += np.mean(LEdata2[y,i,:]*LEdata2[y:,i,:], axis=1)

    cor = cor/(LEdata2.shape[0]-y) 
    
    ax.plot(np.arange(LEdata2.shape[0]-y), cor/cor[0], label='$y_0$=%s' %y)
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$\Delta y$ in px', fontsize=25)
ax.set_ylabel(r'$\langle I(y_0)I(y_0+\Delta y)\rangle$', fontsize=25)

ax.text(5,0.5,'LE', fontsize=25)

ax.set_xlim(-1, LEdata2.shape[0]-max(y0))

plt.show()


# TE ======================================================================
fig, ax = plt.subplots(figsize=(12,8))

for y in y0:
    cor = 0
    for i in range(TEdata2.shape[1]):
        cor += np.mean(LEdata2[y,i,:]*LEdata2[y:,i,:], axis=1)

    cor = cor/(LEdata2.shape[0]-y) 
    
    ax.plot(np.arange(LEdata2.shape[0]-y), cor/cor[0], label='$y_0$=%s' %y)
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$\Delta y$ in px', fontsize=25)
ax.set_ylabel(r'$\langle I(y_0)I(y_0+\Delta y)\rangle$', fontsize=25)

ax.text(5,0.5,'TE', fontsize=25)

ax.set_xlim(-1, LEdata2.shape[0]-max(y0))

plt.show()

# %% hidden=true
# time correlation
LEcor = np.mean(np.mean(LEfinal[:,:,0].T*LEfinal[:,:,:].T, axis=1), axis=1)
TEcor = np.mean(np.mean(TEfinal[:,:,0].T*TEfinal[:,:,:].T, axis=1), axis=1)

# %% hidden=true
# plot
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(np.arange(LEfinal.shape[2])/fps, LEcor/LEcor[0], label='LE')
ax.plot(np.arange(TEfinal.shape[2])/fps, TEcor/TEcor[0], label='TE')
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$t$ in s', fontsize=25)
ax.set_ylabel(r'$\langle I(0)I(t)\rangle/\langle I(0)^2\rangle$', fontsize=25)

plt.show()
