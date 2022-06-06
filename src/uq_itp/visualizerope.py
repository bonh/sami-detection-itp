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
import matplotlib as mpl
import scipy.stats as ss

import dataprep

# +
mpl.style.use(['science', "bright"])

mpl.rcParams['figure.dpi'] = 300
figsize = np.array([4.5,2.50])
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

# +
fps = 46 # frames per second (frames/s)
px = 1.6e-6# size of pixel (m/px)  

pixel = 512

rope_sigma_px = np.array([4, 11]) # in px

#rope_v_px = np.array([1.6, 2.5]) # in px/frame
rope_v = np.array([130,190])

snr = 3 # a/sigma
# -

rope_sigma = np.array(rope_sigma_px*px*1e6).astype(int)
rope_sigma

#rope_v = np.array(rope_v_px*px*fps*1e6).astype(int)
#rope_v
rope_v_px = np.array([1.6,2.5])#np.array(rope_v/px/fps/1e6)
print(rope_v_px)


def sample(amp, cent, sig, x, exp=1):        
    return amp*np.exp(-((cent - x)**2/2/sig**2)**exp)


x = np.arange(1, pixel+1)

# +
fig, axs = plt.subplots(3,1, sharex=True, figsize=(figsize[0],figsize[1]*2))

c = (125,275)
label = "\SI{{{}}}{{px}}\\approx\SI{{{}}}{{\micro\meter}}$"
axs[0].plot(x[c[0]:c[1]], sample(1, 200, rope_sigma_px[0], x)[c[0]:c[1]], label="$\min: "+label.format(rope_sigma_px[0], rope_sigma[0]))
axs[0].plot(x[c[0]:c[1]], sample(1, 200, rope_sigma_px[1], x)[c[0]:c[1]], label="$\max: "+label.format(rope_sigma_px[1], rope_sigma[1]))
axs[0].set_title("A: Spread of the sample", weight="bold", loc="left")
axs[0].legend(title="$\mathit{{ROPE}}$", loc="lower right");

axs[0].set_yticks([])

#
d = 92 # frames
rope_sigma_mean = (rope_sigma_px[1]+rope_sigma_px[0])/2
#label = "\SI{{{}}}{{px/frame}}\\approx\SI{{{}}}{{\micro\meter\per\second}}$"
label = "\SI{{{}}}{{\micro\meter\per\second}}$"
cent = 100
c = (cent-50, cent+50)
axs[1].plot(x[c[0]:c[1]], sample(1, 100, rope_sigma_mean, x)[c[0]:c[1]], "black")
axs[1].annotate("at $t_0$", xy=(80, 0.4),xytext=(50, 0.6),horizontalalignment="center", arrowprops=dict(arrowstyle='->'))

cent = 100+rope_v_px[0]*d
c = (int(cent)-50, int(cent)+50)
axs[1].plot(x[c[0]:c[1]], sample(1, cent, rope_sigma_mean, x)[c[0]:c[1]], label="$\min: "+label.format(rope_v[0]))

cent = 100+rope_v_px[1]*d
c = (int(cent)-50, int(cent)+50)
axs[1].plot(x[c[0]:c[1]], sample(1, cent, rope_sigma_mean, x)[c[0]:c[1]], label="$\max: "+label.format(rope_v[1]))

axs[1].set_title("B: Isotachophoretic velocity", loc="left", weight="bold")
axs[1].set_xlim(0, 512)
axs[1].legend(title="$\mathit{{ROPE}}$", loc="lower right")
#axs[1].set_ylabel("intensity (-)")

axs[1].annotate("at $t_0+\SI{2}{\second}$", xy=(265, 0.5),xytext=(400, 0.9),horizontalalignment="center", arrowprops=dict(arrowstyle='->'), color="white")
axs[1].annotate("at $t_0+\SI{2}{\second}$", xy=(340, 0.5),xytext=(400, 0.9),horizontalalignment="center", arrowprops=dict(arrowstyle='->'));

axs[1].set_yticks([])

#
cent = 200
c = [cent-50, cent+50]
s = sample(1, cent, rope_sigma_mean, x)

a = s.max() # standardized

sigma = a/snr
s_noisy = s + np.random.normal(0,sigma,len(s))
axs[2].plot(x, s_noisy, label="$\mathit{{SNR}}={}$ (AWGN)".format(snr), alpha=0.8)

axs[2].plot(x[c[0]:c[1]], s[c[0]:c[1]], label="sample only", alpha=0.8)

axs[2].set_xticks(np.linspace(0, pixel, 5))
#axs[2].set_xticklabels(np.linspace(0, pixel, 5))
axs[2].set_yticks([])
axs[2].set_xlabel("$x$ (\si{\pixel})")
axs[2].set_title("C: Signal-to-noise ratio", loc="left", weight="bold")
axs[2].legend(loc="upper right")

fig.tight_layout()
fig.savefig("ropes.pdf")

# +
mean_v = 160
std_v = 8 
print(std_v)

rope_v = (mean_v-3*std_v, mean_v+3*std_v)
print(rope_v)

x = np.linspace(rope_v[0], rope_v[1], 100)
plt.fill_between(x, ss.norm.pdf(x, mean_v, std_v));

x = np.linspace(mean_v-4*std_v, mean_v+4*std_v, 100)
plt.plot(x, ss.norm.pdf(x, mean_v, std_v));

# +
#mean_s = 9/px/1e6
#std_s = 1/px/1e6
mean_s = 6.5
std_s = 0.75

rope_s = (mean_s-3*std_s, mean_s+3*std_s)
print(rope_s)

x = np.linspace(rope_s[0], rope_s[1], 100)
plt.fill_between(x, ss.norm.pdf(x, mean_s, std_s), alpha=0.5);

x = np.linspace(mean_s-4*std_s, mean_s+4*std_s, 100)
plt.plot(x, ss.norm.pdf(x, mean_s, std_s), alpha=0.5);

###
mean_s = 8.5
std_s = 0.75

rope_s = (mean_s-3*std_s, mean_s+3*std_s)
print(rope_s)

x = np.linspace(rope_s[0], rope_s[1], 100)
plt.fill_between(x, ss.norm.pdf(x, mean_s, std_s), alpha=0.5);

x = np.linspace(mean_s-4*std_s, mean_s+4*std_s, 100)
plt.plot(x, ss.norm.pdf(x, mean_s, std_s), alpha=0.5);
# -






