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

velocities = np.zeros((len(concentrations),N, 6))
velocities += np.nan

for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            conc = get_conc_name(concentrations[j])
                
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 

            hdi = az.hdi(idata, hdi_prob=.95, var_names="velocity")
            low = float(hdi.velocity.values[0][0])
            high = float(hdi.velocity.values[0][1])

            mode = bayesian.get_mode(idata.posterior, ["velocity"])[0]

            with open("./{}/00{}/intervals.dat".format(concentrations[j], i+1)) as f:
                min_, max_ = [int(x) for x in next(f).split()]
            nframes = max_-min_

            velocities[j,i,:] = np.array([conc, i+1, mode, low, high, nframes])
        except FileNotFoundError as e:
            print(e)
            continue 

velocities = velocities.reshape(-1, 6)
np.savetxt("velocities.csv", velocities, header="c, n, mode, low, high, nframes", delimiter=",", comments='', fmt='%5g, %1.1g, %1.3g, %1.3g, %1.3g, %1.3g')
