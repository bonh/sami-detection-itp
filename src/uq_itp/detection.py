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


results = np.zeros((len(concentrations), N, 3))
for j in range(0, len(concentrations)):
    for i in range(0, N):
        try:
            inname = "./{}/00{}/idata_cross.nc".format(concentrations[j], i+1)
            idata_cross = az.InferenceData.from_netcdf(inname) 
            
            inname = "./{}/00{}/idata.nc".format(concentrations[j], i+1)
            idata = az.InferenceData.from_netcdf(inname) 
            
            conc = get_conc_name(concentrations[j])
            
            v = bayesian.check_rope(idata.posterior["velocity"], rope_velocity)>.95 
            w = bayesian.check_rope(idata3.posterior["sigma"], rope_sigma)>.95
            snr = bayesian.check_refvalue(idata3.posterior["snr"], snr_ref)>.95
            
            results[conc, i] = np.array([v, w, snr])
        except FileNotFoundError as e:
            print(e)
            results[conc, i] = np.array([np.nan, np.nan, np.nan])
            continue


