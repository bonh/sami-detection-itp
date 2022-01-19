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
import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import re

import bayesian
import helper
import dataprep

# +
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l"]

rope_velocity = [117, 184]
rope_sigma = [5, 17]
ref_snr = 3

N = 6

times = [150, 200, 250]


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
        for time in times:
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname) 
                
                ax = az.plot_posterior(idata, var_names=["sigma"]
                            , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_sigma);    
                ax.set_title("")
                #ax.set_xlim(6,14)
            except FileNotFoundError as e:
                print(e)
                axs[j,i].axis("off")
                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
                continue

            if j == 0:
                axs[j,i].set_title("experiment {}".format(i+1))

            if i == 0:
                axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.suptitle("spread - single frame")
fig.tight_layout()
fig.savefig("spread_singleframe.png")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        for time in times:
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname) 
                
                ax = az.plot_posterior(idata, var_names=["snr"]
                            , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_sigma);    
                ax.set_title("")
                #ax.set_xlim(6,14)
            except FileNotFoundError as e:
                print(e)
                axs[j,i].axis("off")
                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
                continue

            if j == 0:
                axs[j,i].set_title("experiment {}".format(i+1))

            if i == 0:
                axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.suptitle("snr - single frame")
fig.tight_layout()
fig.savefig("snr_singleframe.png")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        for time in times:
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname) 
                
                hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
            
                data = idata.observed_data.y
                x = np.linspace(0, len(data), len(data))
                axs[j,i].plot(x, data, alpha=0.8)
                axs[j,i].plot(x, idata.posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red")
                axs[j,i].fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")
            
            except FileNotFoundError as e:
                print(e)
                axs[j,i].axis("off")
                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
                continue

            if j == 0:
                axs[j,i].set_title("experiment {}".format(i+1))

            if i == 0:
                axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.suptitle("peak - single frame")
fig.tight_layout()
fig.savefig("peak_singleframe.png")

# +
results = np.zeros((len(concentrations)*N, 6))
results += np.nan

for j in range(0, len(concentrations)):
    conc = get_conc_name(concentrations[j])
    for i in range(0, N):
        results[j*N+i,0:2] = np.array([conc, i+1])
        for t, time in times:
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname)
                w = int(bayesian.check_rope(idata.posterior["sigma"], rope_sigma)>.95)
                snr = int(bayesian.check_refvalue(idata.posterior["snr"], ref_snr)>.95)
                results[j*N+i,3:6] = np.array([time, w, snr])

            except FileNotFoundError as e:
                print(e)
                continue

print(results)

np.savetxt("detection.csv", results, header="c, i, v, w, snr", delimiter=",", comments='', fmt='%5g, %1.1g, %1.1g, %1.1g, %1.1g')
