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
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l"]

rope_velocity = [100, 200]
rope_sigma = [6, 10]
ref_snr = 3

N = 5

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
            conc = get_conc_name(concentrations[j])
            
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            
            mode = bayesian.get_mode(idata.posterior, ["sigma"])[0]
            hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")

            ax = az.plot_posterior(idata, var_names=["sigma"]
                        , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8);    
            ax.set_title("")
            #ax.set_xlim(6,14)
        except FileNotFoundError as e:
            print(e)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.suptitle("spread")
fig.savefig("spread.png")

# +
hdis = np.zeros((len(concentrations)*N, 5))
print(hdis.shape)
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
                
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            
            hdi = az.hdi(idata, hdi_prob=.95, var_names="snr")

            ax = az.plot_posterior(idata, var_names=["snr"]
                        , kind="kde", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8);    
            ax.set_title("")
            #ax.set_xlim(6,14)
        except FileNotFoundError:
            print(inname)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))
      
fig.suptitle("snr")
fig.savefig("snr.png")
#np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean", delimiter=",", comments='', fmt='%1.1f')

# +
hdis = np.zeros((len(concentrations)*N, 5))
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
            axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))
            
fig.suptitle("multi-frame averaged+estimated peak")

#np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean", delimiter=",", comments='', fmt='%1.1f')

# +
basepath = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/" 

time = 200

channel = [27, 27]
      
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
        
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            inname = basepath + "AF_{}ng_l/00{}.nd2".format(concentrations[j], i+1)
            
            data_raw = helper.raw2images(inname, channel)
            data = dataprep.averageoverheight(data_raw)
            data = dataprep.standardize(data)

            x = np.linspace(0, len(data), len(data))
            axs[j,i].plot(x, data[:,time])

        except FileNotFoundError:
            print(inname)
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
            continue 
 
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(concentrations[j]))       
            
fig.suptitle("single-frame")

# +
detected = np.zeros((len(concentrations)*N, 6))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            inname = "./AF_{}ng_l/00{}".format(concentrations[j], i+1)
            idata_cross = az.InferenceData.from_netcdf(inname+"/idata_cross.nc")
            idata = az.InferenceData.from_netcdf(inname+"/idata.nc")
            
            value1 = bayesian.check_rope(idata_cross.posterior["velocity"], rope_velocity)
            value2 = bayesian.check_rope(idata.posterior["sigma"], rope_sigma)
            value3 = bayesian.check_refvalue(idata.posterior["snr"], ref_snr)
            
            decision = (value1>.95) and (value2>.95) and (value3>.95)
                                 
            detected[j*N+i,:] = np.array([concentrations[j], i+1, value1, value2, value3, decision])
  
        except FileNotFoundError:
            print(inname)
            continue
            
np.savetxt("detected.csv", detected, header="c, n, velocity, spread, snr, decision", delimiter=",", comments='', fmt='%5g, %1.1g, %1.2f, %1.2f, %1.2f, %1.1g')

# +
hdis[np.where(hdis[:,4]>400),4] = np.nan

means = np.nanmean(hdis[:,4].reshape(4,-1,5), axis=2).reshape(4,)
stds = np.nanstd(hdis[:,4].reshape(4,-1,5), axis=2).reshape(4,)

plt.scatter(hdis[:,0], hdis[:,4], label="mode")
plt.errorbar(concentrations, means, yerr=stds.T, fmt="ro", label="mean+std", alpha=0.8)
plt.xscale('log')
plt.xlabel("concentration")
plt.ylabel("velocity")
#plt.ylim(200, 300);
plt.legend();
# -

detected

np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean, nframes", delimiter=",", comments='', fmt='%5g, %1.1g, %1.1f, %1.1f, %1.1f, %1.1g')

# +
detected = np.zeros((len(concentrations)*N, 5))

fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        for time in times:
            try:
                inname = "./AF_{}ng_l/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname)

                hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)

                data = idata.observed_data.y
                x = np.linspace(0, len(data), len(data))
                axs[j,i].plot(x, data, alpha=0.4)
                axs[j,i].plot(x, idata.posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red",alpha=0.4)
                #axs[j,i].fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")
                
                value2 = bayesian.check_rope(idata.posterior["sigma"], rope_sigma)
                value3 = bayesian.check_refvalue(idata.posterior["snr"], ref_snr)
            
                decision = (value2>.95) and (value3>.95)
                                 
                detected[j*N+i,:] = np.array([concentrations[j], i+1, value2, value3, decision])

            except FileNotFoundError:
                print(inname)
                axs[j,i].axis("off")
                axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
                continue 

            if j == 0:
                axs[j,i].set_title("experiment {}".format(i+1))

            if i == 0:
                axs[j,i].set_ylabel("{} ng/l".format(concentrations[j]))
            
fig.suptitle("single-frame+estimated peak")

np.savetxt("detected_single.csv", detected, header="c, n, velocity, spread, snr, decision", delimiter=",", comments='', fmt='%5g, %1.1g, %1.2f, %1.2f, %1.1g')
# -

inname = "./AF_{}ng_l/00{}/idata_single_t{}.nc".format(10, 2, 200)
idata = az.InferenceData.from_netcdf(inname)
hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
plt.plot(x, idata.observed_data.to_array().T, alpha=0.4)
plt.plot(x, idata.posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red",alpha=0.4)
plt.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")

az.plot_posterior(idata, var_names=["sigma", "snr", "fmax", "sigma_noise"])

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        for time in times:
            try:
                inname = "./AF_{}ng_l/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
        
                idata = az.InferenceData.from_netcdf(inname) 

                mode = bayesian.get_mode(idata.posterior, ["sigma"])[0]
                hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")

                ax = az.plot_posterior(idata, var_names=["sigma"]
                            , kind="kde", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8);    
                ax.set_title("")
            except FileNotFoundError:
                print(inname)
                axs[j,i].axis("off")
                axs[j,i].text(0.5, 0.5, 'no sample present\n or error', horizontalalignment='center', verticalalignment='center')
                continue 

            if j == 0:
                axs[j,i].set_title("experiment {}".format(i+1))

            if i == 0:
                axs[j,i].set_ylabel("{} ng/l".format(concentrations[j]))
            
fig.suptitle("single frame: spread")
# -


