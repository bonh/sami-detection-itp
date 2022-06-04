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
import matplotlib as mpl
import os
import numpy as np
import bayesian

import re

# +
mpl.style.use(['science'])

mpl.rcParams['figure.dpi'] = 150
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
figsize = np.array([700*px,500*px])
mpl.rcParams["figure.figsize"] = figsize

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathtools}'

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['axes.titlesize'] = 10

mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["xtick.top"] = False
mpl.rcParams["ytick.right"] = False

# +
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]

N = 6

rope_sigma = (4, 17)


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
# -

d = np.loadtxt("detection.csv", delimiter=",", skiprows=1)
d = np.sum(d[:,2:5], axis=1)
d[d != 3] = np.nan
d[d == 3] = 1
d

spread = np.loadtxt("sigma.csv", delimiter=",", skiprows=1)

mode = spread[:,2]*d
mode = mode.reshape(6, -1)
mode = mode[:-1,:]
mean = np.nanmean(mode, axis=1)
std = np.nanstd(mode, axis=1)
print(mean.shape, std.shape)

hdi_delta = (spread[:,4]-spread[:,3])/spread[:,2]*100*d
hdi_delta = hdi_delta.reshape(6, -1)
hdi_delta = hdi_delta[:-1,:]
mean_hdi_delta = np.nanmean(hdi_delta, axis=1)
std_hdi_delta = np.nanstd(hdi_delta, axis=1)

mean_hdi_delta, std_hdi_delta

n = np.sum(~np.isnan(mode), axis=1)
n

# +
r = np.array([10, 1, 0.1, 0.01, 0.001]).flatten()

fig, ax = plt.subplots()

ax.scatter(np.repeat(r, 6), mode, label="modes", c="red", marker="x",alpha=0.3)

ax.errorbar(r, mean, yerr=std, fmt="ro", label="mean+std (mode)", alpha=1)

for i, txt in enumerate(n):
    ax.annotate(txt, (r[i], mean[i]), xytext=(6, 5), textcoords='offset points')

ax.annotate("includes only measurements with\nsuccessful detection (see number near point)", (1e-3,185))

ax.plot(r, 0*r+rope_sigma[1], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (r[-1],rope_sigma[1]-0.6))
ax.plot(r, 0*r+rope_sigma[0], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (r[-1],rope_sigma[0]+0.1))
    
ax.set_xscale("log")
ax.set_ylim(rope_sigma[0]-rope_sigma[1]*0.2,rope_sigma[1]*1.2)
#ax.yscale("log")

ax.set_xlabel("concentration (ng/L)")
ax.set_ylabel("spread (px)");

ax2 = ax.twinx()
ax2.scatter(np.repeat(r, 6), hdi_delta, label="$\Delta$ hdi", color="green", alpha=0.3, marker="x")
ax2.errorbar(r, mean_hdi_delta, yerr=std_hdi_delta, fmt="go", label="mean+std ($\Delta$ hdi)", alpha=1)
ax2.set_ylabel("spread uncertainty (\% of mode)");
#ax2.set_ylim(0,35)
#ax2.set_yscale("log")

fig.legend(bbox_to_anchor=(.9, 0.82), fontsize=8, frameon=True)

#fig.tight_layout()
#fig.savefig("velocities_summary.png")
# -




