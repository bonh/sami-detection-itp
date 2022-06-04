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
import re
from matplotlib.gridspec import GridSpec

import bayesian
import helper
import dataprep

# +
mpl.style.use(['science'])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([3.42,2.20])
mpl.rcParams["figure.figsize"] = figsize

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['figure.dpi'] = 300

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["xtick.top"] = False
mpl.rcParams["ytick.right"] = False

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['figure.titlesize'] = 9

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

mpl.use("pgf")

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ,r"\usepackage{sansmathfonts}"
        ,r"\usepackage[scaled=0.95]{helvet}"
        ,r"\renewcommand{\rmdefault}{\sfdefault}"
        ])
    }

plt.rcParams.update(pgf_with_latex)

# +
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l"]

rope_sigma = [4.3, 10.7]
ref_snr = 3

N = 7

times = [150, 200, 250]

c_fit = "#EE6677"
c_data = "dimgray"#"#BBBBBB"#black"
c_hist = c_data


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
conc = get_conc(concentrations)

fig = plt.figure(figsize=(4.5,3))
gs = GridSpec(3, 4, figure=fig)
    
ax = fig.add_subplot(gs[0, 0:2])

ax.annotate("{} ng/l".format(conc[0]), xy=(0.1,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")

idata_single = az.InferenceData.from_netcdf(concentrations[0]+"/001/idata_single_t200.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

x = np.arange(0,512)

ax.plot(x, idata_single.observed_data.to_array().T, c=c_data)
ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.set_ylabel("$I$ (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
#ax.legend(loc="upper right")
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)

ax = fig.add_subplot(gs[0, 2])
tmp = idata_single.posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax = fig.add_subplot(gs[0, 3])
tmp = idata_single.posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticks([0,2,4,6,8])

##
ax = fig.add_subplot(gs[1, 0:2])

ax.annotate("{} ng/l".format(conc[1]), xy=(0.1,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")

idata_single = az.InferenceData.from_netcdf(concentrations[1]+"/001/idata_single_t200.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

x = np.arange(0,512)

ax.plot(x, idata_single.observed_data.to_array().T, c=c_data)
ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.set_ylabel("$I$ (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)

ax = fig.add_subplot(gs[1, 2])
tmp = idata_single.posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax = fig.add_subplot(gs[1, 3])
tmp = idata_single.posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticks([0,2,4,6,8])

##
ax = fig.add_subplot(gs[2, 0:2])

ax.annotate("{} ng/l".format(conc[2]), xy=(0.1,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")

idata_single = az.InferenceData.from_netcdf(concentrations[2]+"/002/idata_single_t200.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

x = np.arange(0,512)

ax.plot(x, idata_single.observed_data.to_array().T, c=c_data)
ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.set_ylabel("$I$ (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
ax.tick_params(labelleft = False)
ax.set_xlabel("$x$ (\si{\pixel})")
ax.set_xticks([0, 256, 512])

ax = fig.add_subplot(gs[2, 2])
tmp = idata_single.posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r"$w$ (\si{\pixel})", fontsize=8);

ax = fig.add_subplot(gs[2, 3])
tmp = idata_single.posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r"$\mathit{SNR}$ (-)", fontsize=8);
#ax.set_xticks([0,2,4,6,8])



fig.tight_layout()
fig.savefig("singleframe.pdf")


# -

def get_conc_name(concentration):
    conc = concentration.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000
        
    return conc


# +
#fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
#for j in range(0, len(concentrations)):
#    conc = get_conc_name(concentrations[j])
#    for i in range(0, N):
#        for time in times:
#            try:
#                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
#                idata = az.InferenceData.from_netcdf(inname) 
#                
#                ax = az.plot_posterior(idata, var_names=["sigma"]
#                            , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_sigma);    
#                ax.set_title("")
#                #ax.set_xlim(6,14)
#            except FileNotFoundError as e:
#                print(e)
#                axs[j,i].axis("off")
#                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
#                continue
#
#            if j == 0:
#                axs[j,i].set_title("experiment {}".format(i+1))
#
#            if i == 0:
#                axs[j,i].set_ylabel("{} ng/l".format(conc))
#
#fig.suptitle("spread - single frame")
#fig.tight_layout()
#fig.savefig("spread_singleframe.png")
#
# # +
#fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
#for j in range(0, len(concentrations)):
#    for i in range(0, N):
#        for time in times:
#            try:
#                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
#                idata = az.InferenceData.from_netcdf(inname) 
#                
#                ax = az.plot_posterior(idata, var_names=["snr"]
#                            , kind="hist", point_estimate='mode', hdi_prob=.95, ax=axs[j,i], textsize=8, rope=rope_sigma);    
#                ax.set_title("")
#                #ax.set_xlim(6,14)
#            except FileNotFoundError as e:
#                print(e)
#                axs[j,i].axis("off")
#                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
#                continue
#
#            if j == 0:
#                axs[j,i].set_title("experiment {}".format(i+1))
#
#            if i == 0:
#                axs[j,i].set_ylabel("{} ng/l".format(conc))
#
#fig.suptitle("snr - single frame")
#fig.tight_layout()
#fig.savefig("snr_singleframe.png")
#
# # +
#fig, axs = plt.subplots(len(concentrations), N, figsize=(20,8))
#for j in range(0, len(concentrations)):
#    conc = get_conc_name(concentrations[j])
#    for i in range(0, N):
#        for time in times:
#            try:
#                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
#                idata = az.InferenceData.from_netcdf(inname) 
#                
#                hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
#            
#                data = idata.observed_data.y
#                x = np.linspace(0, len(data), len(data))
#                axs[j,i].plot(x, data, alpha=0.8)
#                axs[j,i].plot(x, idata.posterior_predictive.mean(("chain", "draw")).y, label="fit", color="red")
#                axs[j,i].fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.3, label=".95 HDI", color="red")
#            
#            except FileNotFoundError as e:
#                print(e)
#                axs[j,i].axis("off")
#                axs[j,i].text(0.5, 0.5, 'experiment not found', horizontalalignment='center', verticalalignment='center', fontsize=8)
#                continue
#
#            if j == 0:
#                axs[j,i].set_title("experiment {}".format(i+1))
#
#            if i == 0:
#                axs[j,i].set_ylabel("{} ng/l".format(conc))
#
#fig.suptitle("peak - single frame")
#fig.tight_layout()
#fig.savefig("peak_singleframe.png")

# +
C = len(concentrations)
T = len(times)
results = np.zeros((C, T, N, 5))
results += np.nan
print(results.shape)

for j, c in enumerate(concentrations):
    conc = get_conc_name(c)

    for t, time in enumerate(times):
        for i in range(0, N):
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname)
                w = int(bayesian.check_rope(idata.posterior["sigma"], rope_sigma)>.95)
                snr = int(bayesian.check_refvalue(idata.posterior["snr"], ref_snr)>.95)
                
                hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")
                whdi = int(bayesian.check_rope_hdi(hdi.sigma, rope_sigma))
                hdi = az.hdi(idata, hdi_prob=.95, var_names="snr")
                snrhdi = int(bayesian.check_refvalue_hdilow(hdi.snr[0], ref_snr))
                print(w, whdi, snr, snrhdi) 

                results[j,t,i,:] = np.array([conc, time, i+1, w, snr])

            except FileNotFoundError as e:
                print(e)
                continue

results = results.reshape(N*T*C,-1)
print(results)
#
np.savetxt("detection_singleframe.csv", results, header="c, t, i, w, snr", delimiter=",", comments='', fmt='%5g, %5g, %1.1g, %1.1g, %1.1g')
# -






