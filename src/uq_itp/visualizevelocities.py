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

rope_velocity = (117, 184)


# -

def get_conc(concentrations):
    array = []
    for c in concentrations:
        conc = c.split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
            
        if unit == "pg":
            conc = int(conc)/1000
        array.append(float(conc))
    return array


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

d = np.loadtxt("detection.csv", delimiter=",", skiprows=1)
d = np.sum(d[:,2:5], axis=1)
d[d != 3] = np.nan
d[d == 3] = 1
d

velocities = np.loadtxt("velocities.csv", delimiter=",", skiprows=1)

mode = velocities[:,4]*d
mode = mode.reshape(6, -1)
mode = mode[:-1,:]
mean = np.nanmean(mode, axis=1)
std = np.nanstd(mode, axis=1)
print(mean.shape, std.shape)

hdi_delta = (velocities[:,3]-velocities[:,2])/velocities[:,4]*100*d
hdi_delta = hdi_delta.reshape(6, -1)
hdi_delta = hdi_delta[:-1,:]
mean_hdi_delta = np.nanmean(hdi_delta, axis=1)
std_hdi_delta = np.nanstd(hdi_delta, axis=1)

n = np.sum(~np.isnan(mode), axis=1)
n

# +
r = np.array([10, 1, 0.1, 0.01, 0.001]).flatten()

fig, ax = plt.subplots()

ax.scatter(np.repeat(r, 6), mode, label="modes", color="red", alpha=0.3)

ax.errorbar(r, mean, yerr=std, fmt="ro", label="mean+std (mode)", alpha=1)

for i, txt in enumerate(n):
    ax.annotate(txt, (r[i], mean[i]), xytext=(6, 5), textcoords='offset points')

ax.annotate("includes only measurements with\nsuccessful detection (see number near point)", (1e-3,185))

ax.plot(r, 0*r+rope_velocity[1], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (r[-1],rope_velocity[1]-4))
ax.plot(r, 0*r+rope_velocity[0], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (r[-1],rope_velocity[0]+1))
    
ax.set_xscale("log")
#ax.yscale("log")

#ax.set_ylim(rope_velocity[0]-20, rope_velocity[1]+20);

ax.set_xlabel("concentration (ng/L)")
ax.set_ylabel("velocity (microm/s)");

ax2 = ax.twinx()
ax2.scatter(np.repeat(r, 6), hdi_delta, label="$\Delta$ hdi", color="green", alpha=0.3)
ax2.errorbar(r, mean_hdi_delta, yerr=std_hdi_delta, fmt="go", label="mean+std ($\Delta$ hdi)", alpha=1)
ax2.set_ylabel("velocity uncertainty (% of mode)");
#ax2.set_yscale("log")

fig.legend(bbox_to_anchor=(.68, 0.95) )

fig.tight_layout()
fig.savefig("velocities_summary.png")
# -


