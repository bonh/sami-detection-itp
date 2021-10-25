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
import dataprep


def raw2widthaverage(inname, channel):
    data_raw = raw2images(inname, channel)
    return dataprep.averageoverheight(data_raw)


def raw2images(inname, channel, background=None):
    data_raw = dataprep.load_nd_data(inname, verbose=False)
    print(data_raw.shape)
    data_raw = dataprep.cuttochannel(data_raw, channel[0], channel[1])
    background = np.mean(data_raw[:,:,0:50], axis=2)
    return dataprep.substractbackground(data_raw, background)


def raw2frameaverage(inname, channel):
    data = raw2widthaverage(inname, channel)
    return np.mean(data, axis=1)




