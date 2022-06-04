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

N = 6

sigma = np.zeros((len(concentrations),N, 5))
sigma += np.nan

for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
                
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 

            hdi = az.hdi(idata, hdi_prob=.95, var_names="sigma")
            low = float(hdi.sigma[0])
            high = float(hdi.sigma[1])

            mode = bayesian.get_mode(idata.posterior, ["sigma"])[0]

            sigma[j,i,:] = np.array([conc, i+1, mode, low, high])
        except FileNotFoundError as e:
            print(e)
            continue 

sigma = sigma.reshape(-1, 5)
np.savetxt("sigma.csv", sigma, header="c, n, mode, low, high", delimiter=",", comments='', fmt='%5g, %1.1g, %1.3g, %1.3g, %1.3g')
