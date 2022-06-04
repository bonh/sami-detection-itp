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

# %load_ext autoreload
# %autoreload 2

# +
import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import re

import bayesian
import helper
import dataprep


# +
def get_conc_name(concentration):
    conc = concentration.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000
        
    return concconcentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]

rope_velocity = [117, 184]
rope_sigma = [5, 17]
ref_snr = 3

N = 6

times = [200]#[150, 200, 250]


# -

def get_conc_name(concentration):
    conc = concentration.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000
        
    return conc


# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
        except FileNotFoundError as e:
            print(e)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
            continue

        try:
            conc = get_conc_name(concentrations[j])
            
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            
            mode = bayesian.get_mode(idata.posterior, ["sigma"])[0]
            hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")

            ax = az.plot_posterior(idata, var_names=["sigma"]
                        , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_sigma);    
            ax.set_title("")
            #ax.set_xlim(6,14)
        except FileNotFoundError as e:
            print(e)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'probably no sample present\n or sampling failed', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.suptitle("spread")
fig.tight_layout()
fig.savefig("spread.png")

# +
snr = np.zeros((len(concentrations)*N, N))
snr += np.nan

fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
                
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 

            ax = az.plot_posterior(idata, var_names=["snr"]
                        , kind="kde", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, ref_val=ref_snr);    
            ax.set_title("")
            #ax.set_xlim(6,14)
                
            hdi = az.hdi(idata, hdi_prob=.95, var_names="snr")
            p
            
        except FileNotFoundError:
            print(inname)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'no sample present\n or sampling failed', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))
      
fig.suptitle("snr")
fig.tight_layout()
fig.savefig("snr.png")
#np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean", delimiter=",", comments='', fmt='%1.1f')

# +
hdis = np.zeros((len(concentrations)*N, 3))
print(hdis.shape)
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
            
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname)
            
            hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
            
            data = idata.observed_data.y
            x = np.linspace(0, len(data), len(data))
            axs[j,i].plot(x, data, alpha=0.8)
            axs[j,i].plot(x, idata.posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red")
            axs[j,i].fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")
            
            
            
            #mode = bayesian.get_mode(idata.posterior, ["snr"])[0]

        except FileNotFoundError:
            print(inname)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'no sample present\n or sampling failed', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))
            
fig.suptitle("multi-frame averaged+estimated peak")
fig.tight_layout()
fig.savefig("peak.png")
# -


