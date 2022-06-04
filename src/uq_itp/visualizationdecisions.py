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

mpl.rcParams['figure.dpi'] = 150
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
figsize = np.array([3.42,3.00])
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
    "font.sans-serif": ["Helvetica"],
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ])
    }

plt.rcParams.update(pgf_with_latex)

# +
labels = ["$10^1$", "$10^0$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "", 0]
n = [5, 5, 5, 4, 0.05, 0, 5]
n_s = [5, 4, 0.05, 0.05, 0.05, 0, 5]
c = ["royalblue", "royalblue", "royalblue", "royalblue", "royalblue", "lightsteelblue", "lightsteelblue"]
c_s = ["coral", "coral", "coral", "coral", "coral", "mistyrose", "mistyrose"]

#x = np.arange(len(labels))  # the label locations
x = np.array([0, 1, 2, 3, 4, 4.6, 5.6])
width = 0.4  # the width of the bars

fig, ax = plt.subplots(1,1, figsize=figsize)
rects1 = ax.bar(x - width/2-0.02, n, width, label='multi', color=c)
rects2 = ax.bar(x + width/2+0.02, n_s, width, label='single', color=c_s)

ax.vlines(x[5], 0, 5, ls="dashed", color="black")
ax.set_ylim(-0.1, 5.1)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of correct decisions')
ax.set_xlabel(r"Concentration $c$ (ngL$^{-1}$)")
ax.legend(bbox_to_anchor=(0.2, 0.99, 1., .102), loc=3,
       ncol=2, borderaxespad=0., frameon=False, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.text(x[5]+0.08, 1.2, "Without sample", rotation=90, fontsize=8)

fig.tight_layout()
# -


