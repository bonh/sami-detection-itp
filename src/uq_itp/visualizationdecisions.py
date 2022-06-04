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
labels = ["$10^1$", "$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "", 0]
n = [5, 5, 5, 4, 0.05, 0, 5]
n_s = [5, 4, 0.05, 0.05, 0.05, 0, 5]
c_s = ["#4477AA"]#BBBBBB']#, "royalblue", "royalblue", "royalblue", "royalblue", "lightsteelblue", "lightsteelblue"]
c = ["#EE6677"]#66CCEE"]#228833"]#EE6677"]#coral", "coral", "coral", "coral", "coral", "mistyrose", "mistyrose"]

#x = np.arange(len(labels))  # the label locations
x = np.array([0, 1, 2, 3, 4, 4.6, 5.6])
width = 0.4  # the width of the bars

fig, ax = plt.subplots(1,1, figsize=figsize)
ax.bar(x[:-1] - width/2-0.02, n_s[:-1], width, label='Single frame', color=c_s)
ax.bar(x[:-1] + width/2+0.02, n[:-1], width, label='SAMI w. 200 frames', color=c)

ax.bar(x[-1] - width/2-0.02, n_s[-1], width,  color=c_s, alpha=0.6)
ax.bar(x[-1] + width/2+0.02, n[-1], width, color=c, alpha=0.6)

ax.vlines(x[5], 0, 5, ls="dashed", color="black")
ax.set_ylim(-0.1, 5.1)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of correct decisions')
ax.set_xlabel(r"$c$ (\si{\nano\gram\per\liter})")
ax.legend(bbox_to_anchor=(0.0, 0.99, 1., .102), loc=3,
       ncol=2, borderaxespad=0., frameon=False, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_yticks([0, 1, 2, 3, 4, 5])

ax.text(x[5]+0.08, 0.8, "Control without sample", rotation=90, fontsize=8)

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
left=False, right=False) # labels along the bottom edge are off
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
left=False, right=False) # labels along t


fig.tight_layout()

fig.savefig("decision.pdf")
# -






