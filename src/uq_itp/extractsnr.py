import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import re

import bayesian
import helper
import dataprep

def get_conc_name(concentration):
    conc = concentration.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000
        
    return conc

concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]

N = 7

snr = np.zeros((len(concentrations),N, 5))
snr += np.nan

for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
                
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 

            hdi = az.hdi(idata, hdi_prob=.95, var_names="snr")
            low = float(hdi.snr[0])
            high = float(hdi.snr[1])

            mode = bayesian.get_mode(idata.posterior, ["snr"])[0]

            snr[j,i,:] = np.array([conc, i+1, mode, low, high])
        except FileNotFoundError as e:
            print(e)
            continue 

snr = snr.reshape(-1, 5)
np.savetxt("snr.csv", snr, header="c, n, mode, low, high", delimiter=",", comments='', fmt='%5g, %1.1g, %1.3g, %1.3g, %1.3g')

###
times = [150, 200, 250]

C = len(concentrations)
T = len(times)
snr = np.zeros((C, T, N, 6))
snr += np.nan

for j, c in enumerate(concentrations):
    conc = get_conc_name(c)

    for t, time in enumerate(times):
        for i in range(0, N):
            try:
                inname = "./{}/00{}/idata_single_t{}.nc".format(concentrations[j], i+1, time)
                idata = az.InferenceData.from_netcdf(inname)

                hdi = az.hdi(idata, hdi_prob=.95, var_names="snr")
                low = float(hdi.snr[0])
                high = float(hdi.snr[1])

                mode = bayesian.get_mode(idata.posterior, ["snr"])[0]

                snr[j,t,i,:] = np.array([conc, time, i+1, mode, low, high])

            except FileNotFoundError as e:
                print(e)
                continue

snr = snr.reshape(N*T*C,-1)
np.savetxt("snr_single.csv", snr, header="c, t, n, mode, low, high", delimiter=",", comments='', fmt='%5g, %1.1g, %1.3g, %1.3g, %1.3g, %1.3g')
