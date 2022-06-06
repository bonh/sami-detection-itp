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

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import arviz as az
import helper
import dataprep

# +
mpl.style.use(['science'])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([3.42,2.50])
mpl.rcParams["figure.figsize"] = figsize

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['axes.titlesize'] = 11 
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['lines.markersize'] = 5

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
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ,r"\usepackage{sansmathfonts}"
        ,r"\usepackage[scaled=0.95]{helvet}"
        ,r"\renewcommand{\rmdefault}{\sfdefault}"
        ])
    }

plt.rcParams.update(pgf_with_latex)

# +
inname_base = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]
Ns = [[1,2,3,4,5],
      [1,2,3,5,6],
      [2,3,4,5,7],
      [1,2,3,4,5],
      [1,2,3,4,5],
      [1,2,3,4,5]]

N = 5

rope_sigma = (5,17)
rope_velocity = (117,184)
snr_ref = 3

rope = {'sigma': [{'rope': rope_sigma}]
        , 'velocity': [{'rope': rope_velocity}]}

# + tags=[]
fig, axs = plt.subplots(len(concentrations), N, figsize=(9, 7), sharex=True, sharey=True)
for j in range(0, len(concentrations)):
    for i in range(0, 5):
        conc = concentrations[j].split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
        if unit == "pg":
            conc = int(conc)/1000
        
        if j == 0:
            axs[j,i].set_title("Exp. {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel(r"{} \si{{\nano\gram\per\liter}}".format(conc))
            
        axs[j,i].set_yticks([0, 230,460])
        axs[j,i].set_xticks([0, 256, 512])
        
        try:
            data_raw = helper.raw2images(inname_base + concentrations[j] + "/00{}.nd2".format(Ns[j][i]), (27, 27))
            data = dataprep.averageoverheight(data_raw)
            data = dataprep.standardize(data, axis=0)
            data_fft, mask, ff = dataprep.fourierfilter(data, 100, 40/4, -45, True, True)
            data_fft = dataprep.standardize(data_fft, axis=0)
            
            axs[j][i].imshow(data_fft.T, aspect="auto") 
        except FileNotFoundError as e:
            print(e)
            #axs[j,i].axis("off")
            #axs[j,i].spines["left"].set_color("white")
            #axs[j,i].spines["bottom"].set_color("white")
            #axs[j,i].spines["right"].set_color("white")
            #axs[j,i].spines["top"].set_color("white")
            #axs[j,i].tick_params(axis='x', colors='white', which="both")
            #axs[j,i].tick_params(axis='y', colors='white', which="both")
            axs[j,i].text(0.5, 0.5, 'No exp.\ndone', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=axs[j,i].transAxes)
            continue

plt.tight_layout()
#fig.savefig("stacked.pgf")
fig.savefig("stacked.pdf")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(9,7), sharex=True, sharey=True)
for j in range(0, len(concentrations)):
    for i in range(0, N):
        
        conc = concentrations[j].split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
        if unit == "pg":
            conc = int(conc)/1000
        
        if j == 0:
            axs[j,i].set_title("Exp. {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel(r"{} \si{{\nano\gram\per\liter}}".format(conc))
            
        axs[j,i].tick_params(labelleft = False)
        
        axs[j,i].set_xticks([0, 256, 512]) 
            
        try:
            inname = "./{}/00{}/idata.nc".format(concentrations[j], Ns[j][i])
            idata = az.InferenceData.from_netcdf(inname) 
            
            x = np.arange(0, idata.observed_data.sizes["y_dim_0"])
            axs[j,i].plot(x, idata.observed_data.to_array().T, color="black")
            #axs[j,i].plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
            
            #hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
            #axs[j,i].fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color="r");
        
        except FileNotFoundError as e:
            print(e)
            #axs[j,i].axis("off")
            #axs[j,i].spines["left"].set_color("white")
            #axs[j,i].spines["bottom"].set_color("white")
            #axs[j,i].spines["right"].set_color("white")
            #axs[j,i].spines["top"].set_color("white")
            #axs[j,i].tick_params(axis='x', colors='white', which="both")
            #axs[j,i].tick_params(axis='y', colors='white', which="both")
            axs[j,i].text(0.5, 0.5, 'No exp.\ndone', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=axs[j,i].transAxes)
            continue

fig.tight_layout()
fig.savefig("peaks.pdf")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(9,7))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        
        conc = concentrations[j].split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
        if unit == "pg":
            conc = int(conc)/1000
        
        if j == 0:
            axs[j,i].set_title("Exp. {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel(r"{} \si{{\nano\gram\per\liter}}".format(conc))
            
        axs[j,i].tick_params(axis='both', which='both', labelleft=False, left=False, right=False, top=False)
        axs[j,i].spines['left'].set_visible(False)
        axs[j,i].spines['top'].set_visible(False)
        axs[j,i].spines['right'].set_visible(False)
        
        try:
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], Ns[j][i])
            idata = az.InferenceData.from_netcdf(inname) 
            
            tmp = idata.posterior.velocity.values.flatten()
            _, bla, _ = axs[j,i].hist(tmp, bins="auto", density=True, alpha=0.7, color="black")
            
            min_, max_ = int(np.floor(np.min(bla))), int(np.ceil(np.max(bla)))
           
            axs[j,i].set_xticks([min_, max_])
        
        except FileNotFoundError as e:
            #axs[j,i].axis("off")
            #axs[j,i].spines["left"].set_color("white")
            #axs[j,i].spines["bottom"].set_color("white")
            #axs[j,i].spines["right"].set_color("white")
            #axs[j,i].spines["top"].set_color("white")
            axs[j,i].tick_params(axis='x', colors='white', which="both")
            #axs[j,i].tick_params(axis='y', colors='white', which="both")
            axs[j,i].text(0.5, 0.5, 'No exp.\ndone', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=axs[j,i].transAxes)
            continue

fig.suptitle("Velocity (\si{\micro\meter\per\second})", fontsize=11)
fig.tight_layout()
fig.savefig("posterior_velocity.pdf")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(9,7))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        
        conc = concentrations[j].split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
        if unit == "pg":
            conc = int(conc)/1000
        
        if j == 0:
            axs[j,i].set_title("Exp. {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel(r"{} \si{{\nano\gram\per\liter}}".format(conc))
            
        axs[j,i].tick_params(axis='both', which='both', labelleft=False, left=False, right=False, top=False)
        axs[j,i].spines['left'].set_visible(False)
        axs[j,i].spines['top'].set_visible(False)
        axs[j,i].spines['right'].set_visible(False)
        
        try:
            inname = "./{}/00{}/idata.nc".format(concentrations[j], Ns[j][i])
            idata = az.InferenceData.from_netcdf(inname) 
            
            tmp = idata.posterior.sigma.values.flatten()
            _, bla, _ = axs[j,i].hist(tmp, bins="auto", density=True, alpha=0.7, color="black")
            
            min_, max_ = int(np.floor(np.min(bla))), int(np.ceil(np.max(bla)))
            print(min_, max_)
           
            axs[j,i].set_xticks([min_, max_])
        
        except FileNotFoundError as e:
            #axs[j,i].axis("off")
            #axs[j,i].spines["left"].set_color("white")
            #axs[j,i].spines["bottom"].set_color("white")
            #axs[j,i].spines["right"].set_color("white")
            #axs[j,i].spines["top"].set_color("white")
            axs[j,i].tick_params(axis='x', colors='white', which="both")
            #axs[j,i].tick_params(axis='y', colors='white', which="both")
            axs[j,i].text(0.5, 0.5, 'No exp.\ndone', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=axs[j,i].transAxes)
            continue

fig.suptitle("Sample spread (\si{px})", fontsize=11)
fig.tight_layout()
fig.savefig("posterior_spread.pdf")

# +
fig, axs = plt.subplots(len(concentrations), N, figsize=(9,7))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        
        conc = concentrations[j].split("_")
        match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
        conc, unit = match.groups()
        if unit == "pg":
            conc = int(conc)/1000
        
        if j == 0:
            axs[j,i].set_title("Exp. {}".format(i+1))
            
        if i == 0:
            axs[j,i].set_ylabel(r"{} \si{{\nano\gram\per\liter}}".format(conc))
            
        axs[j,i].tick_params(axis='both', which='both', labelleft=False, left=False, right=False, top=False)
        axs[j,i].spines['left'].set_visible(False)
        axs[j,i].spines['top'].set_visible(False)
        axs[j,i].spines['right'].set_visible(False)
        
        try:
            inname = "./{}/00{}/idata.nc".format(concentrations[j], Ns[j][i])
            idata = az.InferenceData.from_netcdf(inname) 
            
            tmp = idata.posterior.snr.values.flatten()
            _, bla, _ = axs[j,i].hist(tmp, bins="auto", density=True, alpha=0.7, color="black")
            
            min_, max_ = int(np.floor(np.min(bla))), int(np.ceil(np.max(bla)))
            print(min_, max_)
           
            axs[j,i].set_xticks([min_, max_])
        
        except FileNotFoundError as e:
            #axs[j,i].axis("off")
            #axs[j,i].spines["left"].set_color("white")
            #axs[j,i].spines["bottom"].set_color("white")
            #axs[j,i].spines["right"].set_color("white")
            #axs[j,i].spines["top"].set_color("white")
            axs[j,i].tick_params(axis='x', colors='white', which="both")
            #axs[j,i].tick_params(axis='y', colors='white', which="both")
            axs[j,i].text(0.5, 0.5, 'No exp.\ndone', horizontalalignment='center', verticalalignment='center', fontsize=8, transform=axs[j,i].transAxes)
            continue

fig.suptitle("Signal-to-noise ratio", fontsize=11)
fig.tight_layout()
fig.savefig("posterior_snr.pdf")
# -


