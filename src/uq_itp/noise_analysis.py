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
# File: noise_analyis.py
#
# Author:       Lukas Hecht (hecht@fkp.tu-darmstadt.de)
#
# Date:         November 24, 2021

# %% [markdown]
# ## Import Modules

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from nd2reader import ND2Reader
from matplotlib import animation, rc
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import erf

import dataprep

# %% [markdown]
# ## Load Data

# %% code_folding=[]
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

# %% [markdown] heading_collapsed=true
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

# %% [markdown]
# ## Pre-Processing

# %%
# cut to channel
channel = [27,27]

LEdata = dataprep.cuttochannel(LEdata, channel[0], channel[1])
TEdata = dataprep.cuttochannel(TEdata, channel[0], channel[1])

# %%
# substract background
LEback = np.mean(LEdata, axis=2)
TEback = np.mean(TEdata, axis=2)

LEfinal = dataprep.substractbackground(LEdata, LEback)
TEfinal = dataprep.substractbackground(TEdata, TEback)

# %% [markdown] heading_collapsed=true
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

# %% [markdown]
# ## Statistics

# %%
print('------- Mean values -------')
print('LE Mean: ', np.mean(LEfinal))
print('TE Mean: ', np.mean(TEfinal))

# %%
print('------- Standard Deviation -------')
print('LE std: ', np.std(LEfinal))
print('TE std: ', np.std(TEfinal))


# %% [markdown]
# ### Time-Averaged Histogram of Pixel Values

# %% code_folding=[0, 5]
def histogram(data, nbins):
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    bins = (bin_edges[1:]+bin_edges[:-1])/2
    return hist, bins

def gauss(x, a, sig, mu):
    return a/np.sqrt(2*np.pi*sig**2) * np.exp(-(x-mu)**2/(2*sig**2))


# %% code_folding=[]
# histograms
LEhist, LEbins = histogram(LEfinal, nbins=5000)
TEhist, TEbins = histogram(TEfinal, nbins=5000)

# %% code_folding=[]
# Gauss fit
LEpopt, pcov = curve_fit(gauss, LEbins, LEhist, p0=[1.0, 25, 0.0], bounds=(0.0, np.inf))
LEperr = np.sqrt(np.diag(pcov))

TEpopt, pcov = curve_fit(gauss, TEbins, TEhist, p0=[1.0, 25, 0.0], bounds=(0.0, np.inf))
TEperr = np.sqrt(np.diag(pcov))

# %% code_folding=[]
# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,16), sharex=False)

axs[0].plot(LEbins, LEhist, label='LE histogram', color='gold')
axs[0].plot(LEbins, gauss(LEbins, LEpopt[0], LEpopt[1], LEpopt[2]),\
            label='LE Gauss fit', color='k', linestyle='--')

axs[1].plot(TEbins, TEhist, label='TE histogram', color='gold')
axs[1].plot(TEbins, gauss(TEbins, TEpopt[0], TEpopt[1], TEpopt[2]),\
            label='TE Gauss fit', color='k', linestyle='--')

axs[0].set_title('Gaussian: $\mu$=%1.3f$\pm$%1.3f, $\sigma$=%1.3f$\pm$%1.3f, $a$=%1.3f$\pm$%1.3f'\
                 %(LEpopt[2],LEperr[2],LEpopt[1],LEperr[1],LEpopt[0],LEperr[0]), fontsize=15)
axs[1].set_title('Gaussian: $\mu$=%1.3f$\pm$%1.3f, $\sigma$=%1.3f$\pm$%1.3f, $a$=%1.3f$\pm$%1.3f'\
                 %(TEpopt[2],TEperr[2],TEpopt[1],TEperr[1],TEpopt[0],TEperr[0]), fontsize=15)

axs[0].set_xlabel('pixel value', fontsize=25)
axs[0].set_ylabel('probability density', fontsize=25)
axs[1].set_xlabel('pixel value', fontsize=25)
axs[1].set_ylabel('probability density', fontsize=25)

axs[0].tick_params(axis='both', labelsize=15)
axs[1].tick_params(axis='both', labelsize=15)

axs[0].legend(fontsize=25)
axs[1].legend(fontsize=25)


plt.show()

# %% [markdown]
# The Gaussian does not fit exactly for some reason. I also tried to fit a skew normal distribution, which does fit better but results in a wrong mean value.
#
# ToDo: Test if an exponentially modified Gaussian (https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution), which is the joined distribution of a Gaussian and exponential random variable, fits better.
#
# Question: Are there different processes that cause Gaussian and exponentially distributed noise?

# %% [markdown]
# The next sections are not finished yet!!

# %% [markdown] heading_collapsed=true
# ## Spatial Correlation

# %% code_folding=[0] hidden=true
# spatial correlation (in x direction; averaged over y direction)
x0 = [0,50,100,150,200,250,300,350,400]


# LE ================================================================================
fig, ax = plt.subplots(figsize=(12,8))

for x in x0:
    cor = 0
    for i in range(LEdata2.shape[0]):
        cor += np.mean(LEdata2[i,x,:]*LEdata2[i,x:,:], axis=1)

    cor = cor/(LEdata2.shape[1]-x) 
    
    ax.plot(np.arange(LEdata2.shape[1]-x), cor/cor[0], label='$x_0$=%s' %x)
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$\Delta x$ in px', fontsize=25)
ax.set_ylabel(r'$\langle I(x_0)I(x_0+\Delta x)\rangle$', fontsize=25)

ax.text(20,0.5,'LE',fontsize=25)

ax.set_xlim(-1, LEdata2.shape[1]-max(x0))

plt.show()


# TE ================================================================================
fig, ax = plt.subplots(figsize=(12,8))

for x in x0:
    cor = 0
    for i in range(TEdata2.shape[0]):
        cor += np.mean(LEdata2[i,x,:]*LEdata2[i,x:,:], axis=1)

    cor = cor/(LEdata2.shape[1]-x) 
    
    ax.plot(np.arange(LEdata2.shape[1]-x), cor/cor[0], label='$x_0$=%s' %x)
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$\Delta x$ in px', fontsize=25)
ax.set_ylabel(r'$\langle I(x_0)I(x_0+\Delta x)\rangle$', fontsize=25)

ax.text(20,0.5,'TE',fontsize=25)

ax.set_xlim(-1, LEdata2.shape[1]-max(x0))

plt.show()

# %% code_folding=[0] hidden=true
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

# %% [markdown] heading_collapsed=true
# ## Time Correlation

# %% [markdown] hidden=true
# Time correlation calculated for each pixel and averaged over all pixels.

# %% code_folding=[0] hidden=true
# time correlation
LEcor = np.mean(np.mean(LEdata2[:,:,0].T*LEdata2[:,:,:].T, axis=1), axis=1)
TEcor = np.mean(np.mean(TEdata2[:,:,0].T*TEdata2[:,:,:].T, axis=1), axis=1)

# %% code_folding=[0] hidden=true
# plot
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(np.arange(LEdata2.shape[2])/fps, LEcor/LEcor[0], label='LE')
ax.plot(np.arange(TEdata2.shape[2])/fps, TEcor/TEcor[0], label='TE')
    
ax.axhline(0.0, linewidth=1.0, color='k')

ax.tick_params(axis='both', labelsize=15)

ax.legend(fontsize=25)

ax.set_xlabel('$t$ in s', fontsize=25)
ax.set_ylabel(r'$\langle I(0)I(t)\rangle$', fontsize=25)

plt.show()

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ## Spectrum

# %% [markdown] hidden=true
# The spectrum is calculated for each pixel separately and then averaged over all pixels.

# %% code_folding=[0] hidden=true
# calculate mean spectrum (mean over the spectrum of each pixel)
yf = np.fft.rfftn(LEdata2, axes=[2])

LENbin = len(yf[0,0])
LExf = np.fft.fftfreq(LENbin, 1/fps)

print(yf.shape, LENbin)

LEspec = np.mean(np.mean(np.abs(yf[:,:,0:LENbin//2]), axis=1), axis=0)

yf = np.fft.rfftn(TEdata2, axes=[2])

TENbin = len(yf[0,0])
TExf = np.fft.fftfreq(TENbin, 1/fps)

print(yf.shape, TENbin)

TEspec = np.mean(np.mean(np.abs(yf[:,:,0:TENbin//2]), axis=1), axis=0)

# %% code_folding=[0] hidden=true
# plot
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(LExf[0:LENbin//2], LEspec, label='LE')
ax.plot(TExf[0:TENbin//2], TEspec, label='TE')

ax.tick_params(axis='both', labelsize=15)

ax.set_xlabel('$f$ in Hz', fontsize=25)
ax.set_ylabel('spectrum in a.u.', fontsize=25)

ax.legend(fontsize=25)

ax.set_xlim(-0.5, 10)

plt.show()

# %% hidden=true

# %% code_folding=[0] hidden=true

# %% code_folding=[0] hidden=true

# %% hidden=true

# %% hidden=true

# %% hidden=true
