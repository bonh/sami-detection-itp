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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import numpy as np
import pymc3 as pm
import arviz as az

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
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['figure.titlesize'] = 9

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

mpl.rcParams["image.origin"] = "lower"

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
# -

labeller = az.labels.MapLabeller(var_name_map={"sigma":"spread", "centroid":"position 1", "deltac":"shift", "sigmanoise":r"$\sigma_n$"})

# +
basepath = "/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/"

concentration_low = "AF647_10pg_l/"


number_low = "002.nd2"
inname_low = basepath + concentration_low + number_low

channel_lower = 27
channel_upper = 27

startframe = 100
endframe = 300#startframe+200

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)

sigma_mean = 10
rope_sigma = (5,17)
rope_velocity = (117,184)
snr_ref = 3

time = 200

rope = {'sigma': [{'rope': rope_sigma}]
        , 'velocity': [{'rope': rope_velocity}]}


# -

def process_singleframe(inname_raw):
    data_raw = helper.raw2images(inname_raw, (channel_lower, channel_upper))

    data = dataprep.averageoverheight(data_raw)
    data = dataprep.standardize(data, axis=0)
    data = data[:,time]
    
    x = np.arange(0, data.shape[0])
    with bayesian.signalmodel(data, x, artificial=True) as model:
        trace = pm.sample(8000, tune=8000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 

        hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
    
    return hdi, idata


def process(inname_raw):
    data_raw = helper.raw2images(inname_raw, (channel_lower, channel_upper))

    data = dataprep.averageoverheight(data_raw)
    data = dataprep.standardize(data, axis=0)

    data_fft, mask, ff = dataprep.fourierfilter(data, 100, 40/4, -45, True, True)
    data_fft = dataprep.standardize(data_fft, axis=0)

    N = 4
    deltalagstep = 5
    lagstepstart = 30
    x_lag, corr_mean_combined = \
        dataprep.correlation(data_fft, startframe, endframe, lagstepstart=lagstepstart, deltalagstep=deltalagstep, N=N)

    with bayesian.signalmodel_correlation(corr_mean_combined.T, -np.array([x_lag,]*N).T, px, deltalagstep, lagstepstart, fps, artificial=True) as model:
        trace = pm.sample(8000, tune=8000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idatacross = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
        
        hdicross = az.hdi(idatacross.posterior_predictive, hdi_prob=.95)
    
    v = bayesian.get_mode(idatacross.posterior, ["velocity"])[0]*1e-6
    data_shifted = dataprep.shift_data(data, v, fps, px)
    
    data_fft_shifted, mask_shifted, ff_shifted = dataprep.fourierfilter(data_shifted, 30, 30, 0, True, False)
    data_fft_shifted = dataprep.standardize(data_fft_shifted)
    
    data_mean = np.mean(data_fft_shifted[:,startframe:endframe], axis=1)
    data_mean = dataprep.standardize(data_mean)
    
    x = np.arange(0, data_mean.shape[0])
    with bayesian.signalmodel(data_mean, x, artificial=True) as model:
        trace = pm.sample(8000, tune=8000, return_inferencedata=False, cores=4, target_accept=0.9)

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 

        hdi = az.hdi(idata.posterior_predictive, hdi_prob=.95)
    
    return data_raw, data, x_lag, corr_mean_combined, idatacross, hdicross, data_shifted, data_mean, hdi, idata, x


# +
concentration = "AF647_10ng_l/"
number = "002"
inname_raw = basepath + concentration + number + ".nd2"

high = process(inname_raw)

high_single = process_singleframe(inname_raw)

# +
concentration = "AF647_10pg_l/"
number = "004"
inname_raw = basepath + concentration + number + ".nd2"

low = process(inname_raw)

# +
columns = 6
rows = 7

c_fit = "#EE6677"
c_data = "dimgray"#"#BBBBBB"#black"
c_hist = c_data
# -

#w, h = figsize[0]*columns*0.4, figsize[1]*rows*0.4
w, h = 7, 7

# +
fig = plt.figure(figsize=(w, h), constrained_layout=True)
gs = GridSpec(rows, 2, figure=fig, height_ratios=[1,1,1,1,1,1,2])

ax = fig.add_subplot(gs[5, 0])
tmp = ax.annotate("F: Averaged and fitted\n$I_{y, n}(x')$", xy=(0.01,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")
fig.savefig("imageprocessing.pdf")
bla = tmp.get_window_extent()
title_coord = (bla.bounds[0], bla.bounds[1])
print(title_coord)
# -

fig = plt.figure(figsize=(w, h), constrained_layout=True)
gs = GridSpec(rows, 2, figure=fig, height_ratios=[1,1,1,1,1,1,2])
gs

# +
data_raw, data, x_lag, corr_mean_combined, idatacross, hdicross, data_shifted, data_mean, hdi, idata, x = high

hdisingle, idatasingle = high_single

height = data_raw.shape[0]

axmain = fig.add_subplot(gs[0, 0])
axmain.set_title(r"Concentration \SI{10}{\nano\gram\per\liter}")
axmain.annotate("A: Cut to channel and background subtracted\
                \n$I(x, y, n=100)$", xy=(0.01,0.95), xycoords='axes fraction', color="w", horizontalalignment="left", verticalalignment="top", weight="bold")
axmain.imshow(data_raw[:,:,time], aspect="auto", cmap="viridis");
axmain.set_yticks(np.linspace(0, height, 3))
axmain.set_ylabel("$y$ (px)");
axmain.yaxis.set_label_coords(-0.03, 0.5)
axmain.tick_params(labelbottom = False)
axmain.tick_params(labelleft = False)

#
ax = fig.add_subplot(gs[1, 0], sharex=axmain)
ax.annotate("B: Height averaged\n$I_{N_y}(x, n=100)$", xy=(0.01,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top", weight="bold")
ax.plot(x, data[:,time], color=c_data, alpha=1.0)
ax.plot(x, idatasingle.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.fill_between(x, hdisingle["y"][:,0], hdisingle["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
ax.set_ylabel("$I_{N_y}$ (-)")
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)

ax.annotate("single image fit\nresulting $\mathit{SNR}\\approx 6$", xy=(0.6,0.95), xycoords='axes fraction', 
            color="black", horizontalalignment="left", verticalalignment="top")

#
ax = fig.add_subplot(gs[2, 0], sharex=axmain)
ax.annotate("C: Stacked\n$I_{N_y}(x, n)$", xy=(0.01,0.95), xycoords='axes fraction', color="white", horizontalalignment="left", verticalalignment="top", weight="bold")
ax.imshow(data.T, aspect="auto", cmap="viridis")
ax.set_ylabel("$n$");
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.set_xlabel("$x$ (px)")
ax.set_xticks([0, 256, 511])
ax.set_xticklabels([0, 256, 512])
ax.set_yticks([0, 460])
#ax.xaxis.set_label_coords(0.5, -0.1)

#
ax = fig.add_subplot(gs[3, 0])
ax.plot(x_lag, corr_mean_combined.T[:,0], color=c_data, alpha=1.0, label="data");
ax.plot(x_lag, idatacross.posterior_predictive.mean(("chain", "draw"))["y"][:,0], label="fit", color=c_fit)
ax.annotate("D: Correlated, averaged, and fitted\n$X_{N_n}^{L=30}(\Delta x)$", xy=(0.01,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top", weight="bold")
#for i in [-1]:
#    ax.plot(x_lag, corr_mean_combined.T[:,i], "b", alpha=0.3);
#    ax.plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"][:,i], label="fit", color="r")
ax.set_xlabel("$\Delta x$ (\si{\pixel})")
ax.set_ylabel("$X_{N_n}^L$ (-)")
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.tick_params(labelleft = False)
#ax.xaxis.set_label_coords(0.5, -0.1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.set_xticks([-1, -128, -256])
ax.set_xticklabels([0,-128,-256])

for i in [0]:#range(0,N):
    ax.fill_between(x_lag, hdicross["y"][:,i,0], hdicross["y"][:,i,1], alpha=0.2, label=".95 HDI", color="r");
    
handles, labels = ax.get_legend_handles_labels()
#ax.legend([handles[0], handles[N], handles[-1]], [labels[0], labels[N], labels[-1]]);

#
ax = fig.add_subplot(gs[4, 0], sharex=axmain)
ax.annotate("E: Shifted\n$I_{N_y}(x',n)$", xy=(0.01,0.95), xycoords='axes fraction', color="white", horizontalalignment="left", verticalalignment="top", weight="bold")
ax.imshow(data_shifted.T, aspect="auto", cmap="viridis")
ax.set_ylabel("$n$");
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.tick_params(labelbottom = False)
ax.set_yticks([0, 460])

#
ax = fig.add_subplot(gs[5, 0], sharex=axmain)
tmp = ax.annotate("F: Averaged and fitted\n$I_{N_y, N_n}(x')$", xy=(0.01,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top", weight="bold")
ax.plot(x, data_mean, color=c_data, label="data");
ax.plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.set_xlabel("$x$ (\si{\pixel})")
ax.set_ylabel("$I_{N_y, N_n}$ (-)")
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
ax.tick_params(labelleft = False)
#ax.xaxis.set_label_coords(0.5, -0.1)
#ax.legend()
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)

#fig.canvas.draw() 
#box = mpl.text.Text.get_window_extent(tmp)
#tcbox = fig.dpi_scale_trans.inverted().transform(box)
#print(tcbox)

#
gs0 = gs[-1,0].subgridspec(1, 5, width_ratios=[1,3.5,3.5,3.5,0.5])

ax1 = fig.add_subplot(gs0[0,1])
ax1.annotate("G: Marginal posterior distributions and detection", xy=(-0.05,1.10), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top", annotation_clip=False, weight="bold")
fig.texts.append(ax1.texts.pop())
#ax1.set_title("G: Marginal posterior distributions and detection", x=1.5)
xmin, xmax = 162.5, 163.5
tmp = idatacross.posterior.velocity.values.flatten()
ax1.hist(tmp, bins="auto", density=True, alpha=1.0, color=c_hist, rwidth=1)
ax1.tick_params(axis='y', which='both', labelleft=False, left=False)
ax1.spines['left'].set_visible(False)

ax1.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
ax1.annotate('ROPE', xy=(0, 0.12), xycoords="axes fraction", xytext=(10, 2), textcoords='offset points', fontsize=6)

ax1.set_xlabel(r"$v_\mathrm{ITP}$ (\si{\micro\meter\per\second})", fontsize=8);
ax1.set_xticks(np.linspace(157, 163, 2))
#ax1.set_xlim(xmin,xmax);
#ax1.set_yticks(np.linspace(0, height, 3))

#
ax2 = fig.add_subplot(gs0[0,2])
xmin, xmax = 4,9
tmp = idata.posterior.sigma.values.flatten()
ax2.hist(tmp, bins="auto", density=True,alpha=1.0, color=c_hist, rwidth=1)
ax2.tick_params(axis='y', which='both', labelleft=False, left=False)
ax2.spines['left'].set_visible(False)

ax2.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
ax2.annotate('ROPE', xy=(0, 0.12), xycoords="axes fraction", xytext=(10, 2), textcoords='offset points', fontsize=6)

ax2.set_xlabel(r"$w$ (\si{\pixel})", fontsize=8);
ax2.set_xticks(np.linspace(4.8, 8.7, 2))



#
ax = fig.add_subplot(gs0[0,3])
xmin, xmax = 0,100
tmp = idata.posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True,alpha=1.0, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)

#ax.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
#ax.annotate('', xy=(0, 0.12), xycoords="axes fraction", xytext=(10, 2), textcoords='offset points', fontsize=6)

ax.set_xlabel(r"$\mathit{SNR}$ (-)", fontsize=8);
ax.set_xticks(np.linspace(80, 120, 2))

# +
data_raw, data, x_lag, corr_mean_combined, idatacross, hdicross, data_shifted, data_mean, hdi, idata, x = low

axmain = fig.add_subplot(gs[0, 1])
axmain.set_title(r"Concentration \SI{0.01}{\nano\gram\per\liter}")
axmain.imshow(data_raw[:,:,time], aspect="auto", cmap="viridis");
axmain.tick_params(labelbottom = False)
axmain.tick_params(labelleft = False)

#
ax = fig.add_subplot(gs[1, 1], sharex=axmain)
ax.plot(data[:,time], color=c_data)
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)

#
ax = fig.add_subplot(gs[2, 1], sharex=axmain)
ax.imshow(data.T, aspect="auto", cmap="viridis")
ax.set_xlabel("$x$ (\si{\pixel})")
ax.tick_params(labelleft = False)
ax.set_xticks([0, 256, 511])
ax.set_xticklabels([0, 256, 512])
#ax.xaxis.set_label_coords(0.5, -0.1)
ax.set_yticks([0, 460])

#
ax = fig.add_subplot(gs[3, 1])
ax.plot(x_lag, corr_mean_combined.T[:,0], color=c_data, label="data");
ax.plot(x_lag, idatacross.posterior_predictive.mean(("chain", "draw"))["y"][:,0], label="fit", color=c_fit)
#for i in [-1]:
#    ax.plot(x_lag, corr_mean_combined.T[:,i], "b", alpha=0.3);
#    ax.plot(x_lag, idata.posterior_predictive.mean(("chain", "draw"))["y"][:,i], label="fit", color="r")
ax.set_xlabel("$\Delta x$ (\si{\pixel})")
ax.tick_params(labelleft = False)
#ax.xaxis.set_label_coords(0.5, -0.1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.set_xticks([-1,-128,-256])
ax.set_xticklabels([0,-128,-256])

for i in [0]:#range(0,N):
    ax.fill_between(x_lag, hdicross["y"][:,i,0], hdicross["y"][:,i,1], alpha=0.2, label=".95 HDI", color=c_fit);
    
handles, labels = ax.get_legend_handles_labels()
#ax.legend([handles[0], handles[N], handles[-1]], [labels[0], labels[N], labels[-1]]);

#
ax = fig.add_subplot(gs[4, 1], sharex=axmain)
ax.imshow(data_shifted.T, aspect="auto", cmap="viridis")
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)

#
ax = fig.add_subplot(gs[5, 1], sharex=axmain)
ax.plot(x, data_mean, c_data, label="data");
ax.plot(x, idata.posterior_predictive.mean(("chain", "draw"))["y"], label="fit", color=c_fit)
ax.set_xlabel("$x$ (\si{\pixel})")
ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit);
ax.tick_params(labelleft = False)
#ax.xaxis.set_label_coords(0.5, -0.1)
#ax.legend()
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.set_xticks([0, 256, 511])
ax.set_xticklabels([0, 256, 512])

#
gs0 = gs[-1,1].subgridspec(1, 5, width_ratios=[1,3.5,3.5,3.5,0.5])

ax1 = fig.add_subplot(gs0[0,1])
#ax1.annotate("G: Marginal posteriors and detection", 
#                xy=(0.01,1), xycoords='axes fraction', color="black", 
#           horizontalalignment="left", verticalalignment="top")
#ax1.set_title("G: Marginal posteriors and detection", x=0.8)
xmin, xmax = 157, 162
tmp = idatacross.posterior.velocity.values.flatten()
ax1.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax1.tick_params(axis='y', which='both', labelleft=False, left=False)
ax1.spines['left'].set_visible(False)

ax1.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
ax1.annotate('ROPE', xy=(0, 0.12), xycoords="axes fraction", xytext=(17.4, 2), textcoords='offset points', fontsize=6, color="white")

ax1.set_xlabel(r"$v_\mathrm{ITP}$ (\si{\micro\meter\per\second})", fontsize=8);
ax1.set_xticks(np.linspace(157, 163, 2))

#
ax2 = fig.add_subplot(gs0[0,2])
xmin, xmax = 4,9
tmp = idata.posterior.sigma.values.flatten()
ax2.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax2.tick_params(axis='y', which='both', labelleft=False, left=False)
ax2.spines['left'].set_visible(False)

ax2.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
ax2.annotate('ROPE', xy=(0, 0.12), xycoords="axes fraction", xytext=(25, 2), textcoords='offset points', fontsize=6)

ax2.set_xlabel(r"$w$ (\si{\pixel})", fontsize=8);
ax2.set_xticks(np.linspace(4.8, 8.7, 2))

#
ax = fig.add_subplot(gs0[0,3])
xmin, xmax = 0,100
tmp = idata.posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)

#ax.annotate("", xy=(0,0.1), xycoords="axes fraction", xytext=(1,0.1), arrowprops=dict(arrowstyle='<->'), zorder=0)
#ax.annotate('', xy=(0, 0.12), xycoords="axes fraction", xytext=(10, 2), textcoords='offset points', fontsize=6)

ax.set_xlabel(r"$\mathit{SNR}$ (-)", fontsize=8);
ax.set_xticks(np.linspace(6, 10, 2))

# +
#fig.align_ylabels()

fig.set_constrained_layout_pads(wspace=1, w_pad=0)
# -

fig.savefig("imageprocessing.pdf")




