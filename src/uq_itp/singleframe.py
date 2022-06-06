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

# +
import numpy as np
import matplotlib.pyplot as plt
import dataprep
import helper
import utilities
import bayesian

import arviz as az
from sklearn import preprocessing
import pymc3 as pm

from IPython.display import HTML

import matplotlib as mpl

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import re

# +
mpl.style.use(['science'])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([3.42,2.20])
mpl.rcParams["figure.figsize"] = figsize

mpl.rcParams["image.origin"] = "lower"

mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 8
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
inname_base = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l"]

N = 6
time = 200


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


for c in concentrations:
    
    conc = c.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000
    
    for i in range(1,N+1):
        while True:
            try:
                data = helper.raw2widthaverage(inname_base + c + "/00{}.nd2".format(i), (40,40), background=True)
                x = np.arange(0, data.shape[0])

                data = (data[:,time]-np.mean(data[:,time]))/np.std(data[:,time])

                with bayesian.signalmodel(data, x) as model:
                    trace = pm.sample(1000, tune=1000, return_inferencedata=False, cores=4, target_accept=0.9)

                    ppc = pm.fast_sample_posterior_predictive(trace, model=model)
                    idata_single = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 

                    hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)
            except Exception as e:
                print(e)
                if j<3:
                    print("retry")
                    j+=1
                    continue
                else:
                    break

            from pathlib import Path
            Path("./singleframe").mkdir(parents=True, exist_ok=True)

            idata_single.to_netcdf("./singleframe/idata_single_{}_{}_{}.nc".format(conc, time, i))
            
            break

# +
conc = get_conc(concentrations)

fig = plt.figure(figsize=2*figsize)
gs = GridSpec(3, 3, figure=fig)
    
ax = fig.add_subplot(gs[0, 0:2])
ax.set_ylim(-2.5, 5.5)
ax.set_yticks([-2, 0, 2, 4])
ax.set_xticks([])
ax.set_xlim(0,512)

ax.set_title("{} ng/l".format(conc[0]), loc="left")

data = helper.raw2widthaverage(inname_base + concentrations[0] + "/001.nd2", (40,40), background=True)
x = np.arange(0, data.shape[0])

data = (data[:,time]-np.mean(data[:,time]))/np.std(data[:,time])
ax.plot(x, data, "b", alpha=0.5, label="data")

idata_single = az.InferenceData.from_netcdf("./singleframe/idata_single_10_200_1.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_ylabel("intensity (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color="r");
ax.legend()

ax = fig.add_subplot(gs[0, 2])
az.plot_posterior(idata_single, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", ax=ax, textsize=10)
ax.set_title("")
ax.set_xlim(0,7)
ax.set_xticks([0, 3.5, 7])

###
ax = fig.add_subplot(gs[1, 0:2])

ax.set_title("{} ng/l".format(conc[1]), loc="left")
ax.set_ylim(-2.5, 5.5)
ax.set_yticks([-2, 0, 2, 4])
ax.set_xticks([])
ax.set_xlim(0,512)

data = helper.raw2widthaverage(inname_base + concentrations[1] + "/001.nd2", (40,40), background=True)
x = np.arange(0, data.shape[0])

data = (data[:,time]-np.mean(data[:,time]))/np.std(data[:,time])
ax.plot(x, data, "b", alpha=0.5, label="data")

idata_single = az.InferenceData.from_netcdf("./singleframe/idata_single_1_200_1.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_ylabel("intensity (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color="r");

ax = fig.add_subplot(gs[1, 2])
az.plot_posterior(idata_single, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", ax=ax, textsize=10)
ax.set_title("")
ax.set_xlim(0,7)
ax.set_xticks([0, 3.5, 7])

###
ax = fig.add_subplot(gs[2, 0:2])

ax.set_title("{} ng/l".format(conc[2]), loc="left")
ax.set_ylim(-2.5, 5.5)
ax.set_yticks([-2, 0, 2, 4])
ax.set_xticks(np.linspace(0,512,5))
ax.set_xlim(0,512)

data = helper.raw2widthaverage(inname_base + concentrations[2] + "/002.nd2", (40,40), background=True)
x = np.arange(0, data.shape[0])

data = (data[:,time]-np.mean(data[:,time]))/np.std(data[:,time])
ax.plot(x, data, "b", alpha=0.5)

idata_single = az.InferenceData.from_netcdf("./singleframe/idata_single_0.1_200_2.nc")
hdi_single = az.hdi(idata_single.posterior_predictive, hdi_prob=.95)

ax.plot(x, idata_single.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color="r")
ax.set_xlabel("length (px)")
ax.set_ylabel("intensity (-)")
ax.fill_between(x, hdi_single["y"][:,0], hdi_single["y"][:,1], alpha=0.2, label=".95 HDI", color="r");

ax = fig.add_subplot(gs[2, 2])
az.plot_posterior(idata_single, "snr", hdi_prob=.95, point_estimate="mode", kind="hist", ax=ax, textsize=10)
ax.set_title("")
ax.set_xlabel(r"snr");
ax.set_xlim(0,7)
ax.set_xticks([0, 3.5, 7])

fig.tight_layout()
