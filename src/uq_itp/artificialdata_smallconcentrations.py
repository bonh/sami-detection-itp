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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm
import arviz as az
import pymc3 as pm

import bayesian
import dataprep
import idata_crosscorrelation
import idata_sample

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
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ,r"\usepackage{sansmathfonts}"
        ,r"\usepackage[scaled=0.95]{helvet}"
        ,r"\renewcommand{\rmdefault}{\sfdefault}"
        ])
    }

plt.rcParams.update(pgf_with_latex)


# -

def create_data(length, nframes, a, c, w, alpha, x, height=None): 
    data = np.zeros((length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    if height:
        for h in range(0, height):
            data[h, :, :] = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
    else:
        data = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
            
    return data


def create_images(length, height, nframes, a, c, w, alpha, x): 
    data = np.zeros((height, length, nframes))
    
    xx, cc = np.meshgrid(x, c, sparse=True)
           
    for h in range(0, height):
        data[h, :, :] = bayesian.model_sample(a, cc, w, alpha, xx).eval().T
            
    return data


# +
snr = 0.01
a = 1
w = 6 
alpha = 0
x = np.arange(0, 512)
   
nframes = 200000

c = np.ones((nframes))*100

data_raw = create_data(512, nframes, a, c, w, alpha, x)
# -

sigma = bayesian.fmax(a, c, w, alpha)/snr
data_noisy = data_raw + np.random.normal(0,sigma.eval(),data_raw.shape)

# +
x = np.arange(0, data_noisy.shape[0])

idatas = []
for nframes in np.array([5e4, 1e5, 2e5],dtype=int):
    data_mean = np.mean(data_noisy[:,0:nframes], axis=1)
    with bayesian.signalmodel(data_mean, x, artificial=True) as model:
        trace = pm.sample(8000, tune=8000, return_inferencedata=False, cores=4, target_accept=0.9)
        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idatas.append(az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model))

# +
c_fit = "#EE6677"
c_data = "dimgray"#"#BBBBBB"#black"
c_hist = c_data

##
fig = plt.figure(constrained_layout=True, figsize=(4.5,4))
spec2 = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)

##
ax = fig.add_subplot(spec2[0, 0:2])
ax.set_title("A: Noisy data with $\mathit{SNR}=0.01$", loc="left", weight="bold")
ax.plot(x, data_noisy[:,100], c=c_data)
#ax.plot(x, data_raw[:,0], c=c_fit)
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)

##
ax = fig.add_subplot(spec2[1, 0:2])
ax.set_title("B: Averaged data with fit", loc="left", weight="bold")
ax.annotate("$50000$ images included", xy=(0.3,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
ax.plot(x, idatas[0].observed_data.to_array().T, color=c_data)
ax.plot(x, idatas[0].posterior_predictive.mean(("chain", "draw")).y, color=c_fit)
hdi = az.hdi(idatas[0].posterior_predictive, hdi_prob=.95)
ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit)
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
#ax.set_title("{:.0e} images included in average".format(results[i,0]), loc="left")

##
ax = fig.add_subplot(spec2[2, 0:2])
ax.annotate("$100000$ images included", xy=(0.3,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
ax.plot(x, idatas[1].observed_data.to_array().T, color=c_data)
ax.plot(x, idatas[1].posterior_predictive.mean(("chain", "draw")).y, color=c_fit)
hdi = az.hdi(idatas[1].posterior_predictive, hdi_prob=.95)
ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit)
ax.tick_params(labelbottom = False)
ax.tick_params(labelleft = False)
#ax.set_title("{:.0e} images included in average".format(results[i,0]), loc="left")

##
ax = fig.add_subplot(spec2[3, 0:2])
ax.annotate("$200000$ images included", xy=(0.3,0.95), xycoords='axes fraction', color="black", horizontalalignment="left", verticalalignment="top")
ax.plot(x, idatas[2].observed_data.to_array().T, color=c_data)
ax.plot(x, idatas[2].posterior_predictive.mean(("chain", "draw")).y, color=c_fit)
hdi = az.hdi(idatas[2].posterior_predictive, hdi_prob=.95)
ax.fill_between(x, hdi["y"][:,0], hdi["y"][:,1], alpha=0.2, label=".95 HDI", color=c_fit)
ax.tick_params(labelleft = False)
ax.set_xlabel("$x$ (\si{\pixel})")
ax.set_xticks([0, 256, 512])
#ax.set_title("{:.0e} images included in average".format(results[i,0]), loc="left")

###
ax = fig.add_subplot(spec2[1, 2])
tmp = idatas[0].posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(2,12)

ax = fig.add_subplot(spec2[1, 3])
tmp = idatas[0].posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(1.0,6)

###
ax = fig.add_subplot(spec2[2, 2])
tmp = idatas[1].posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(2,12)

ax = fig.add_subplot(spec2[2, 3])
tmp = idatas[1].posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(1.0,6)

###
ax = fig.add_subplot(spec2[3, 2])
tmp = idatas[2].posterior.sigma.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r"$w$ (\si{\pixel})", fontsize=8);
ax.set_xlim(2,12)

ax = fig.add_subplot(spec2[3, 3])
tmp = idatas[2].posterior.snr.values.flatten()
ax.hist(tmp, bins="auto", density=True, color=c_hist, rwidth=1)
ax.tick_params(axis='y', which='both', labelleft=False, left=False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r"$\mathit{SNR}$ (-)", fontsize=8);
ax.set_xlim(1.0,6)


###
fig.savefig("small_concentration.pdf")
# -



