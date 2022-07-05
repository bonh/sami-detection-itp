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

import numpy as np
import scipy.special as sp
import scipy.optimize as so
import matplotlib.pyplot as plt
import matplotlib as mpl

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
def normal(x, a, c, w):
    return a*np.exp(-(c - x)**2/2/w**2)

def skewed_normal(x, a, c, w, alpha):
    return normal(x, a, c, w) * (1-sp.erf((alpha*(c - x))/np.sqrt(2)/w))


# +
a = 1
c = 40
x = np.linspace(0, 512/2, 512)
w = 10 
alpha = 10

f = skewed_normal(x, a, c, w, alpha)
g = skewed_normal(x, a, c+56, w, alpha)

X = np.correlate(f, g, mode="same")

popt, pcov = so.curve_fit(normal, x-512/4, X)

# +
fig, axs = plt.subplots(1,2,figsize=(4.5,1.8))

axs[0].plot(x, f, label="$t$");
axs[0].plot(x, g, label="$t+\Delta t$");
axs[0].legend(fontsize=8)
axs[0].spines['left'].set_visible(False)
axs[0].tick_params(axis='y', which='both', labelleft=False, left=False)
axs[0].set_title("A: Sample peaks", loc="left", weight="bold")
axs[0].set_xticks([0, 512/4, 512/2])
axs[0].set_xlabel("$x$ (\si{\pixel})")
axs[0].set_ylabel("$I$ (-)")

axs[1].plot(x-512/4, X, label="Cross-correlation")
axs[1].plot(x-512/4, normal(x-512/4, *popt), label="Fitted Gaussian")
axs[1].legend(fontsize=8)
axs[1].spines['left'].set_visible(False)
axs[1].tick_params(axis='y', which='both', labelleft=False, left=False)
axs[1].set_title("B: Cross-correlation and fit", loc="left", weight="bold")
axs[1].set_xticks([-512/4, 0, 512/4])
axs[1].set_xlabel("$\Delta x$ (\si{\pixel})")
axs[1].set_ylabel("$X$ (-)")

fig.tight_layout()
fig.savefig("skewedgauss_crosscorrelation.pdf")
# -
# ### 





