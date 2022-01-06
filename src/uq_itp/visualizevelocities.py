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
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]

N = 6

rope_velocity = (120, 180)

# +
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

            hdi = az.hdi(idata, hdi_prob=.95, var_names="velocity").velocity.values[0]

            if hdi[1] > 200 or hdi[0] < 100:
                ax = az.plot_posterior(
                    idata, var_names=["velocity"], 
                    kind="hist", hdi_prob="hide", point_estimate=None, ax=axs[j,i], textsize=8, rope=rope_velocity);
                axs[j,i].text(0.5, 0.9, 'probably no sample present\n or sampling failed', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=ax.transAxes)
            else: 
                ax = az.plot_posterior(
                    idata, var_names=["velocity"], 
                    kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_velocity);
                #ax = az.plot_posterior(
                #    idata, var_names=["velocity"], group="prior", kind="hist", ax=axs[j,i], textsize=8, point_estimate=None, hdi_prob="hide");
                ax.set_xlim(100, 200)
            ax.set_title("")
            
            with open("./{}/00{}/intervals.dat".format(concentrations[j], i+1)) as f:
                min_, max_ = [int(x) for x in next(f).split()]
            nframes = max_-min_
            
            mode = bayesian.get_mode(idata.posterior, ["velocity"])[0]
            hdis[j*N+i,:] = np.array([conc, i+1, hdi[0], hdi[1], mode, nframes])
        
        except FileNotFoundError as e:
            axs[j,i].axis("off")
            axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
            continue
                
        if j == 0:
            axs[j,i].set_title("experiment {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel("{} ng/l".format(conc))

fig.tight_layout()
fig.savefig("velocities.png")
np.savetxt("velocities.csv", hdis, header="c, n, low, high, mean, nframes", delimiter=",", comments='', fmt='%5g, %1.1g, %1.1f, %1.1f, %1.1f, %1.1g')
# -

#hdis[np.where(hdis[:,4]>400),4] = np.nan

means = np.nanmean(hdis[:,4].reshape(3,-1,N), axis=2).reshape(3,)
stds = np.nanstd(hdis[:,4].reshape(3,-1,N), axis=2).reshape(3,)

plt.scatter(hdis[:,0], hdis[:,4], label="mode")
plt.errorbar(concentrations, means, yerr=stds.T, fmt="ro", label="mean+std", alpha=0.8)
plt.xscale('log')
plt.xlabel("concentration")
plt.ylabel("velocity")
#plt.ylim(200, 300);
plt.legend();
plt.tight_layout()
plt.savefig("velocities_summary.png")
