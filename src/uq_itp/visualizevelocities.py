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
import bayesian

import re

# +
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l"]

rope_velocity = [200, 250]

N = 5

# +
rope = {'velocity': [{'rope': rope_velocity}]}

hdis = np.zeros((len(concentrations)*N, 6))
print(hdis.shape)
fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = concentrations[j].split("_")
            match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
            conc, unit = match.groups()
            
            if unit == "pg":
                conc = int(conc)/1000
            
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            
            ax = az.plot_posterior(
                idata, var_names=["velocity"], 
                kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8);
            #ax = az.plot_posterior(
            #    idata, var_names=["velocity"], group="prior", kind="hist", ax=axs[j,i], textsize=8, point_estimate=None, hdi_prob="hide");
            #ax.set_xlim(100, 200)
            ax.set_title("")
            
            hdi = az.hdi(idata, hdi_prob=.95, var_names="velocity")
            
            with open("./{}/00{}/intervals.dat".format(concentrations[j], i+1)) as f:
                min_, max_ = [int(x) for x in next(f).split()]
            nframes = max_-min_
            
            mode = bayesian.get_mode(idata.posterior, ["velocity"])[0]
            hdis[j*N+i,:] = np.array([conc, i+1, hdi.velocity.values[0], hdi.velocity.values[1], mode, nframes])
        
        except FileNotFoundError as e:
            print(e)        
                
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.savefig("velocities.png")
np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean, nframes", delimiter=",", comments='', fmt='%5g, %1.1g, %1.1f, %1.1f, %1.1f, %1.1g')
# -

np.isnan(data)

hdis[np.where(hdis[:,4]>400),4] = np.nan

means = np.nanmean(hdis[:,4].reshape(3,-1,5), axis=2).reshape(3,)
stds = np.nanstd(hdis[:,4].reshape(3,-1,5), axis=2).reshape(3,)

plt.scatter(hdis[:,0], hdis[:,4], label="mode")
plt.errorbar(concentrations, means, yerr=stds.T, fmt="ro", label="mean+std", alpha=0.8)
plt.xscale('log')
plt.xlabel("concentration")
plt.ylabel("velocity")
#plt.ylim(200, 300);
plt.legend();
