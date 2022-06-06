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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np
import pymc3 as pm
import arviz as az
from scipy import stats, signal

import helper
import dataprep
import bayesian

# +
mpl.style.use(['science', "bright"])

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
basepath = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"
concentration = "AF647_10ng_l/"

number = "005.nd2"
inname = basepath + concentration + number

channel_lower = 27
channel_upper = 27

startframe = 125
endframe = startframe+100

# +
data_raw = helper.raw2images(inname, (channel_lower, channel_upper))
data = dataprep.averageoverheight(data_raw)
data = dataprep.standardize(data, axis=0)

plt.imshow(data.T, origin="lower")

# +
x = np.arange(0, data[:,0].shape[0])
    
idatas = []
times = range(startframe, endframe, 5)
print(times)
for time in times:
    with bayesian.signalmodel(data[:,time], x, artificial=True) as model:
        trace = pm.sample(1000, tune=1000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata_single = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
        
        idatas.append(idata_single)

# +
modes = np.zeros(len(times))
hdis = np.zeros((len(times), 1, 2))
for i, idata in enumerate(idatas):
    hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")
    bla = hdi.sigma.to_numpy().reshape(1,2)
    hdis[i,:,:] = bla
    modes[i] = bayesian.get_mode(idata.posterior, ["sigma"])[0]
   
hdis = hdis.reshape(len(times),2).T
hdis[0,:] = modes - hdis[0,:]
hdis[1,:] = hdis[1,:] - modes

# +
c_data = "dimgray"#"#BBBBBB"#black"
c_hist = c_data

fig = plt.figure(figsize=(4.5, 3.5))

ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
ax2 = plt.subplot2grid((2,2), (1,0))
ax3 = plt.subplot2grid((2,2), (1,1))

#plt.subplots_adjust(hspace=0.4)
#plt.subplots_adjust(wspace=0.25)

fit1 = idatas[8].posterior_predictive.mean(("chain", "draw"))["y"].to_numpy()
fit2 = idatas[18].posterior_predictive.mean(("chain", "draw"))["y"].to_numpy()
data1 = idatas[8].observed_data.to_array().T
data2 = idatas[18].observed_data.to_array().T

ax1.set_title(r"A: Intensity at 10 \si{\nano\gram\per\liter} from two different images", loc="left", weight="bold")
ax1.plot(x,fit1, c="#EE6677")
ax1.plot(x,fit2)
ax1.plot(x,data1,alpha=0.6)
ax1.plot(x,data2,alpha=0.6)
ax1.vlines(np.argmax(fit1), -2.5, 7, ls="dashed", color="black")
ax1.vlines(np.argmax(fit2), -2.5, 7, ls="dashed", color="black")
           
ax1.set_ylim(-2.5, 7)
ax1.set_xlabel('$x$ (\si{\pixel})')
ax1.set_ylabel('$I$ (-)')
ax1.set_yticklabels("")

ax1.annotate("$n$", (200,2.5), (150,3), arrowprops=dict(arrowstyle="->"))
ax1.annotate("$n+\Delta n$", (325,1.5), (375,2), arrowprops=dict(arrowstyle="->"))
ax1.annotate(r"", (np.argmax(fit1)+1,5), (np.argmax(fit2)+1,5), arrowprops=dict(arrowstyle="<->"))
ax1.annotate(r"$\Delta x_\text{max}$", (np.argmax(fit1)+1,5.3), (18,0), textcoords="offset points")
ax1.annotate(r"$v_\mathrm{ITP} = \frac{\Delta x_\text{max}}{\Delta n}$", (400,4.5))

####

ax2.set_title("B: Sample spread along channel and histogram", loc="left", weight="bold")

v_itp = (np.argmax(fit2)-np.argmax(fit1))/(times[18]-times[8])
positions = np.array(times)*v_itp
positions -= np.min(positions)
positions += startframe/v_itp
times = positions

ax2.errorbar(times, modes, yerr=hdis, marker="o", ls="")

model = np.polyfit(times, modes, 1)
predict = np.poly1d(model)
x_lin_reg = np.arange(times[0], times[-1])
y_lin_reg = predict(x_lin_reg)
ax2.plot(x_lin_reg, y_lin_reg)
print(model[0])
ax2.annotate("Slope: $\SI{0.005}{\pixel\per\pixel}$", (280, 5.9), (155, 8), arrowprops=dict(arrowstyle="->"))
#
ax2.hlines(np.mean(modes), x_lin_reg[0], x_lin_reg[-1], color="black", ls="dashed")
print(np.mean(modes))
ax2.annotate("Mean: $\SI{5.5}{\pixel}$", (270, 5.5), (205, 3), arrowprops=dict(arrowstyle="->"))
#
ax2.set_ylabel("Mode, 95% HDI of $w$ (\si{\pixel})");
ax2.set_ylim(2,9)
ax2.set_yticks([3, 5, 7])
ax2.set_xlim(np.min(times)-15, np.max(times)+15)
ax2.set_xlabel('$x$ (\si{\pixel})')
ax2.set_xticks([128, 256])

####
_, binedges, _ = ax3.hist(modes, bins=8, density=True, alpha=0.7, color=c_data)
bins = (binedges[1:]+binedges[:-1])/2
p1, p2 = stats.norm.fit(modes)
mean, var = stats.norm.stats(p1, p2)
binsplot = np.linspace(1, 10, 200)
norm = stats.norm.pdf(binsplot, p1, p2)

#ax3.set_title("C: Frequency", loc="left", weight="bold")
ax3.plot(binsplot, norm, color="#EE6677", label=r"$\mathcal{{N}}(\mu\approx{:.1f},\sigma\approx{:.1f})$".format(p1, p2))
ax3.set_xlabel('$w$ (\si{\pixel})')
#ax3.set_ylabel('Probability')
ax3.tick_params(labelleft = False, left=False)
ax3.tick_params(axis='y', which='both', labelleft=False, left=False)
ax3.spines['left'].set_visible(False)
ax3.legend()
ax3.set_xlim(5.6-2, 5.6+2)

fig.tight_layout()
fig.savefig("EstimationROPE.pdf")
# -

np.min(positions), np.max(times)


