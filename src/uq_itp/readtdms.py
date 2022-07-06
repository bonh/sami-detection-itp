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

from nptdms import TdmsFile
import matplotlib.pyplot as plt

# +
inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.tdms"

fps = 46 # frames per second (1/s)
px = 1.6e-6 # size of pixel (m/px)
# -

tdms_file = TdmsFile.read(inname)

time = tdms_file.groups()[0].channels()[0]
voltage = tdms_file.groups()[0].channels()[1]
corr = tdms_file.groups()[0].channels()[2]

plt.plot(voltage);

startframe = 420

# +
import numpy as np
x = voltage[startframe:]
x = (x-np.min(x))/(np.max(x)-np.min(x))

y = time[startframe:]
y = (y-np.min(y))/(np.max(y)-np.min(y))
(m, b) = np.polyfit(y, x, 1)
xp = np.polyval([m, b], y)

plt.figure(figsize=(3,3))
plt.scatter(y[::5], x[::5], marker="o")
plt.plot(y, xp, "red");
plt.xlabel("time")
plt.ylabel("voltage");
plt.annotate("linear fit", (0.6, 0.6), (0.6,0.4), arrowprops=dict(arrowstyle="->"));
# -

