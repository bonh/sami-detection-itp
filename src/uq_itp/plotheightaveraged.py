# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

import os
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from tqdm import tqdm
import input


def rawimages2heightaveraged(inname):
    data_raw = input.load_nd_data(inname, verbose=False)
    data_raw = input.cuttochannel(data_raw, 27, 27)
    background = np.mean(data_raw[:,:,0:10],axis=2)
    data_raw = input.substractbackground(data_raw, background)
    return input.averageoverheight(data_raw)


# +
basepath = "/home/cb51neqa/projects/itp/exp_data/"

innames = []
for root, dirs, files in os.walk(basepath):
    for file in files:
        if(file.endswith(".nd2")):
            innames.append(os.path.join(root,file))

pprint(innames)

# +
fig,axs = plt.subplots(len(innames),1, sharex=True, sharey=True, figsize=(15, len(innames)*8))

for i in tqdm(range(0, len(innames))):
    data = rawimages2heightaveraged(innames[i])
    axs[i].imshow(np.transpose(data), aspect="auto",origin="lower")
    axs[i].set_ylabel("t");

    if "nd2" in innames[i].split("/"):
        title = innames[i].split("/")[-4] + "_" + innames[i].split("/")[-3] + "_" + innames[i].split("/")[-1]
    else:
        title = innames[i].split("/")[-3] + "_" + innames[i].split("/")[-2] + "_" + innames[i].split("/")[-1]

    axs[i].set_title(title)
axs[i].set_xlabel("x");
# -

fig.tight_layout()
fig.savefig("heightaveraged.pdf")


