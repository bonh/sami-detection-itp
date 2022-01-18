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

import arviz as az
import numpy as np
import re

import bayesian

# +
concentrations = ["AF647_10ng_l", "AF647_1ng_l", "AF647_100pg_l", "AF647_10pg_l", "AF647_1pg_l", "AF647_0ng_l"]

rope_velocity = [117, 184]
rope_sigma = [5, 17]
ref_snr = 3

N = 6

times = [200]#[150, 200, 250]


# -

def get_conc_name(concentration):
    conc = concentration.split("_")
    match = re.match(r"([0-9]+)([a-z]+)", conc[1], re.I)
    conc, unit = match.groups()
            
    if unit == "pg":
        conc = int(conc)/1000

    return conc

results = np.zeros((len(concentrations)*N, 5))
results += np.nan

for j in range(0, len(concentrations)):
    for i in range(0, N):
        conc = get_conc_name(concentrations[j])
        results[j*N+i,0:2] = np.array([conc, i+1])
        try:
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], i+1)
            idata_cross = az.InferenceData.from_netcdf(inname) 
            v = int(bayesian.check_rope(idata_cross.posterior["velocity"], rope_velocity)>.95)
            results[j*N+i,2] = np.array([v])

            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            w = int(bayesian.check_rope(idata.posterior["sigma"], rope_sigma)>.95)
            snr = int(bayesian.check_refvalue(idata.posterior["snr"], ref_snr)>.95)
            results[j*N+i,3:5] = np.array([w, snr])

        except FileNotFoundError as e:
            print(e)
            continue

print(results)

np.savetxt("detection.csv", results, header="c, i, v, w, snr", delimiter=",", comments='', fmt='%5g, %1.1g, %1.1g, %1.1g, %1.1g')
