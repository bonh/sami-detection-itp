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
import arviz as az
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import re

import bayesian
import helper
import dataprep

# +
mpl.style.use(['science', "bright"])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([3.42,2.50])
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

rope_velocity = [130, 190]
rope_sigma = [4.3, 10.7]
ref_snr = 3

# +
data = np.loadtxt("detection.csv", delimiter=",", skiprows=1)
data_s = np.loadtxt("detection_singleframe.csv", delimiter=",", skiprows=1)

data = np.nan_to_num(data)
data_s = np.nan_to_num(data_s)

d = np.sum(data[:,2:5], axis=1)
d_s = np.sum(data_s[:,3:5], axis=1)

d_s = np.append(d_s, np.tile(0, 3*7*3))
assert len(d)*3 == len(d_s)

d[d != 3] = 0
d[d == 3] = 1

d_s[d_s == 1] = 0
d_s[d_s == 2] = 1

d_s = d_s.reshape(-1,3,7)
d_s_full = np.array(d_s)
d_s = np.sum(d_s, axis=1)
d_s[d_s > 0] = 1
d_s = d_s.flatten()

assert len(d) == len(d_s)
# -

columns = ["c\\n", 1, 2, 3, 4, 5, 6, 7]
rows = [10, 1, 0.1, 0.01, 0.001, 0]

# +
data_t = np.vstack([d_s, d]).astype(str)

data_t[data_t == "1.0"] = "\x1b[32m\u25A0\x1b[0m"
data_t[data_t == "0.0"] = "\x1b[31m\u25A0\x1b[0m"
data_t = np.apply_along_axis(lambda data_t: data_t[0] + '/' + data_t[1], 0, data_t)
data_t = data_t.reshape(len(rows), -1)
data_t = np.vstack([rows, data_t.T]).T
# -

data_t[2:6,6] = ""
data_t[2,1] = ""
data_t[3:6,7] = ""
data_t[1,4] = ""
data_t[0:2,7] = ""
data_t[0,6] = ""

# +
from tabulate import tabulate

print(tabulate(data_t, headers=columns))


# -

def get_snr():
    snr = np.loadtxt("snr.csv", delimiter=",", skiprows=1)
    snr_s = np.loadtxt("snr_single.csv", delimiter=",", skiprows=1)

    mode = snr[:,2]*d
    mode[mode == 0] = np.nan
    mode = mode.reshape(6, -1)
    mode = mode[:-1,:]
    mean = np.nanmean(mode, axis=1)
    std = np.nanstd(mode, axis=1)

    hdi_delta = (snr[:,4]-snr[:,3])/snr[:,2]*100*d
    hdi_delta = hdi_delta.reshape(6, -1)
    hdi_delta = hdi_delta[:-1,:]
    mean_hdi_delta = np.nanmean(hdi_delta, axis=1)
    std_hdi_delta = np.nanstd(hdi_delta, axis=1)

    n = np.sum(~np.isnan(mode), axis=1)
    print(n)

    mode_s = snr_s[:,3]*d_s_full.flatten()
    mode_s[mode_s == 0] = np.nan
    mode_s = mode_s.reshape(-1,3,7)
    mode_s = np.nanmax(mode_s, axis=1)
    mean_s = np.nanmean(mode_s, axis=1)
    std_s = np.nanstd(mode_s, axis=1)

    hdi_delta_s = (snr_s[:,5]-snr_s[:,4])/snr_s[:,3]*100*d_s_full.flatten()
    hdi_delta_s = hdi_delta_s.reshape(6, -1)
    hdi_delta_s = hdi_delta_s[:-1,:]
    mean_hdi_delta_s = np.nanmean(hdi_delta_s, axis=1)
    std_hdi_delta_s = np.nanstd(hdi_delta_s, axis=1)

    mode_s = mode_s[:-1]
    mean_s = mean_s[:-1]
    std_s = std_s[:-1]

    n_s = np.sum(~np.isnan(mode_s), axis=1)
    print(n_s)
    
    return mode, mean, std, mode_s, mean_s, std_s, n, n_s, hdi_delta, mean_hdi_delta, std_hdi_delta


def get_velocities():
    velocities = np.loadtxt("velocities.csv", delimiter=",", skiprows=1)

    mode = velocities[:,2]*d
    mode[mode == 0] = np.nan
    mode = mode.reshape(6, -1)
    mode = mode[:-1,:]
    mean = np.nanmean(mode, axis=1)
    std = np.nanstd(mode, axis=1)
    print(mean.shape, std.shape)

    hdi_delta = (velocities[:,4]-velocities[:,3])/velocities[:,2]*100*d
    hdi_delta = hdi_delta.reshape(6, -1)
    hdi_delta = hdi_delta[:-1,:]
    mean_hdi_delta = np.nanmean(hdi_delta, axis=1)
    std_hdi_delta = np.nanstd(hdi_delta, axis=1)
    
    return mode, mean, std, hdi_delta, mean_hdi_delta, std_hdi_delta


def get_spread():
    spread = np.loadtxt("sigma.csv", delimiter=",", skiprows=1)

    mode = spread[:,2]*d
    mode[mode == 0] = np.nan
    mode = mode.reshape(6, -1)
    mode = mode[:-1,:]
    mean = np.nanmean(mode, axis=1)
    std = np.nanstd(mode, axis=1)

    hdi_delta = (spread[:,4]-spread[:,3])/spread[:,2]*100*d
    hdi_delta = hdi_delta.reshape(6, -1)
    hdi_delta = hdi_delta[:-1,:]
    mean_hdi_delta = np.nanmean(hdi_delta, axis=1)
    std_hdi_delta = np.nanstd(hdi_delta, axis=1)

    spread_s = np.loadtxt("sigma_single.csv", delimiter=",", skiprows=1)
    mode_s = spread_s[:,3]*d_s_full.flatten()
    mode_s[mode_s == 0] = np.nan
    mode_s = mode_s.reshape(-1,3,7)
    mode_s = np.nanmax(mode_s, axis=1)
    mean_s = np.nanmean(mode_s, axis=1)
    std_s = np.nanstd(mode_s, axis=1)

    mode_s = mode_s[:-1]
    mean_s = mean_s[:-1]
    std_s = std_s[:-1]
    
    n = np.sum(~np.isnan(mode), axis=1)
    print(n)
    n_s = np.sum(~np.isnan(mode_s), axis=1)
    print(n_s)
    
    return mode, mean, std, mode_s, mean_s, std_s, n, n_s, hdi_delta, mean_hdi_delta, std_hdi_delta


mode, mean, std, mode_s, mean_s, std_s, n, n_s, hdi_delta, mean_hdi_delta, std_hdi_delta = get_spread()

v_mean_s = [161, 160]
v_std_s = [7, 10]

# +
r = np.array([10, 1, 0.1, 0.01, 0.001]).flatten()
r = r[:-1]

fig, axs = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]*2.5), sharex=True)

#
mode, mean, std, hdi_delta, mean_hdi_delta, std_hdi_delta = get_velocities()
mean = mean[:-1]
std = std[:-1]
mean_hdi_delta = mean_hdi_delta[:-1]
std_hdi_delta = std_hdi_delta[:-1]

ax = axs[0]
ax.set_title("A: Isotachophoretic velocity", loc="left", weight="bold")
#ax.xaxis.set_tick_params(labeltop=True)

#ax.scatter(np.repeat(r, 7), mode, label="modes", color="red", alpha=0.3)

t = ax.errorbar(r, mean, yerr=std, label="mean+std (mode)", alpha=1, marker="o", ls="none")

#for i in range(0,len(mean)-1):
#    ax.annotate(n[i], (r[i], mean[i]), xytext=(6, 5), textcoords='offset points')
#ax.annotate(n[-2], (r[-1], mean[-1]), xytext=(-10, 5), textcoords='offset points')

#ax.annotate("includes only measurements with\nsuccessful detection (see number near point)", (1e-3,185))
bla = r[-2]+(r[-2]-r[-1])/2
ax.plot(r, 0*r+rope_velocity[1], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (bla,rope_velocity[1]-6), fontsize=8)
ax.plot(r, 0*r+rope_velocity[0], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (bla,rope_velocity[0]+1), fontsize=8)

ax.annotate("", (0.8,0.8), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=t[0].get_color()))
    
ax.set_xscale("log")
#ax.yscale("log")

#ax.set_ylim(rope_velocity[0]-20, rope_velocity[1]+20);


ax.set_ylabel(r"$v_\mathrm{ITP}$ (\si{\micro\meter\per\second})")

ax2 = ax.twinx()
ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler

#ax2.scatter(np.repeat(r, 7), hdi_delta, label="$\Delta$ hdi", color="green", alpha=0.3)

t = ax2.errorbar(r*1.1, mean_hdi_delta, yerr=std_hdi_delta, label="mean+std ($\Delta$ hdi)", alpha=1, marker="x", ls="none")
ax.annotate("", (0.2,0.8), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=t[0].get_color()))
ax2.set_ylabel(".95 HDI (\% of mode)");
#ax2.set_yscale("log")
ax2.set_ylim(-1,5)

t = ax.errorbar(r[0:2]*0.9, v_mean_s, yerr=v_std_s, mfc="white", marker="o", ls="none")
ax.annotate("", (0.8,0.3), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=t[0].get_color()))

#fig.legend(bbox_to_anchor=(.68, 0.95) )

#
mode, mean, std, mode_s, mean_s, std_s, n, n_s, hdi_delta, mean_hdi_delta, std_hdi_delta = get_spread()
mean = mean[:-1]
std = std[:-1]
mean_s = mean_s[:-1]
std_s = std_s[:-1]
mean_hdi_delta = mean_hdi_delta[:-1]
std_hdi_delta = std_hdi_delta[:-1]

ax = axs[1]
ax.set_title("B: Sample spread", loc="left", weight="bold")

#l1 = ax.scatter(np.repeat(r, 7), mode, label="modes", color="red", alpha=0.3)


l2 = ax.errorbar(r, mean, yerr=std, label="mean+std (mode)", alpha=1, marker="o", ls="none")
ax.annotate("", (0.8,0.8), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l2[0].get_color()))


#for i in range(0,len(mean)-1):
#    ax.annotate(n[i], (r[i], mean[i]), xytext=(6, 5), textcoords='offset points')
#ax.annotate(n[-2], (r[-1], mean[-1]), xytext=(-10, -10), textcoords='offset points')
#    
#ax.annotate(n_s[0], (r[0], mean_s[0]), xytext=(1, 5), textcoords='offset points')
#ax.annotate(n_s[1], (r[1], mean_s[1]), xytext=(3, 6), textcoords='offset points')

#ax.annotate("includes only measurements with\nsuccessful detection (see number near point)", (1e-3,185))

ax.plot(r, 0*r+rope_sigma[1], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (bla,rope_sigma[1]-0.6), fontsize=8)
ax.plot(r, 0*r+rope_sigma[0], ls="dashed", color="black", alpha=0.5)
ax.annotate("ROPE", (bla,rope_sigma[0]+0.1), fontsize=8)
    
ax.set_xscale("log")
#ax.yscale("log")

#ax.set_ylim(rope_velocity[0]-20, rope_velocity[1]+20);


ax.set_ylabel("$w$ (\si{\pixel})");

ax2 = ax.twinx()
ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler

#ax2.scatter(np.repeat(r, 7), hdi_delta, label="$\Delta$ hdi", color="green", alpha=0.3)
l4 = ax2.errorbar(r*1.1, mean_hdi_delta, yerr=std_hdi_delta, label="mean+std ($\Delta$ hdi)", alpha=1, ls="none", marker="x")
ax.annotate("", (0.2,0.8), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l4[0].get_color()))
ax2.set_ylabel(".95 HDI (\% of mode)");
#ax2.set_yscale("log")
ax2.set_ylim(0,30)


l3 = ax.errorbar(r*0.9, mean_s, yerr=std_s, mfc="white", marker="o", ls="none")
ax.annotate("", (0.8,0.3), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l3[0].get_color()))

#fig.legend(bbox_to_anchor=(.68, 0.95) )
#
#

#
mode, mean, std, mode_s, mean_s, std_s, n, n_s, hdi_delta, mean_hdi_delta, std_hdi_delta = get_snr()
mean = mean[:-1]
std = std[:-1]
mean_s = mean_s[:-1]
std_s = std_s[:-1]
mean_hdi_delta = mean_hdi_delta[:-1]
std_hdi_delta = std_hdi_delta[:-1]

ax = axs[2]
ax.set_title("C: Signal-to-noise ratio", loc="left", weight="bold")

#l1 = ax.scatter(np.repeat(r, 7).reshape(5,-1), mode, label="modes", color="red", alpha=0.3)
l2 = ax.errorbar(r, mean, yerr=std, label="Mode", alpha=1, marker="o", ls="none")
ax.annotate("", (0.8,0.8), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l2[0].get_color()))

#for i in range(0,len(mean)-1):
#    ax.annotate(n[i], (r[i], mean[i]), xytext=(6, -14), textcoords='offset points')
#ax.annotate(n[-2], (r[-1], mean[-1]), xytext=(-10, -8), textcoords='offset points')
#    
#ax.annotate(n_s[0], (r[0], mean_s[0]), xytext=(1, 5), textcoords='offset points')
#ax.annotate(n_s[1], (r[1], mean_s[1]), xytext=(3, 6), textcoords='offset points')

#ax.annotate("includes only measurements with\nsuccessful detection (see number near point)", (1e-3,200))

ax.plot(r, 3+mean*0, ls="dashed", color="black")
ax.annotate("$\mathit{SNR}=3$", (bla,3.2), fontsize=8)
    
ax.set_xscale("log")
ax.set_yscale("log")

#ax.set_ylim(rope_velocity[0]-20, rope_velocity[1]+20);

ax.set_ylabel("$\mathit{SNR}$ (-)");
ax.set_xlabel(r"$c$ (\si{\nano\gram\per\liter})")

ax2 = ax.twinx()
ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler

#ax2.scatter(np.repeat(r, 7).reshape(5,-1), hdi_delta, label="$\Delta$ hdi", color="green", alpha=0.3)
l4 = ax2.errorbar(r*1.1, mean_hdi_delta, yerr=std_hdi_delta, label=".95 HDI", alpha=1, ls="none", marker="x")
ax.annotate("", (0.2,0.8), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l4[0].get_color()))
ax2.set_ylabel(".95 HDI (\% of mode)");
#ax2.set_yscale("log")
ax2.set_ylim(5,30)

#fig.legend(bbox_to_anchor=(.5, 0.95) )

#ax.scatter(np.repeat(r, 7).reshape(5,-1), mode_s, label="modes", color="yellow", alpha=0.3)
l3 = ax.errorbar(r, mean_s, yerr=std_s, mfc="white", label="Single frame", marker="o", ls="none")
ax.annotate("", (0.8,0.3), xycoords='axes fraction', xytext=(20, 0), textcoords='offset points',arrowprops=dict(arrowstyle="-|>", color=l3[0].get_color()))
#ax.annotate("", xy=(r[0],mean_s[0]), xytext=(r[1],mean_s[1]), arrowprops=dict(arrowstyle='<->'))
#ax.annotate("single\nframe", xy=(r[0],mean_s[0]), xytext=(0.70,0.23), textcoords="axes fraction", bbox=dict(fc="w"))

#ax.legend(handles = [l1,l2,l3,l4],loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=True)
lgnd = ax.legend(handles = [l2[0], l4[0]], labels=["Mean mode", "Mean .95 HDI"], loc='upper center', bbox_to_anchor=(0.30, -0.25), ncol=2, fontsize=7
                 , frameon=True, markerscale=1.0, title="SAMI with 200 frames", title_fontsize=7, labelspacing=0.25, handletextpad=0.05, columnspacing=0.2)
lgnd._legend_box.align = "left"

lgnd2 = ax.legend(handles = [l3[0]], labels=["Mean mode"], loc='upper center', bbox_to_anchor=(0.85, -0.25), ncol=1, fontsize=7
                 , frameon=True, markerscale=1.0, title="Single frame", title_fontsize=7, labelspacing=0.25, handletextpad=0.05, columnspacing=0.2)
lgnd2._legend_box.align = "left"

ax.add_artist(lgnd)

ax.set_xticks(np.array([0.01,0.1,1,10]))
ax.set_xticklabels(["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"])
#ax.invert_xaxis()

#
fig.align_ylabels()
fig.tight_layout(h_pad=0.1)
fig.savefig("results.pdf")
# -








