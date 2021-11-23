import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

import dataprep
import bayesian
import helper

def main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity):
    data_raw = helper.raw2images(inname, channel)

    data_raw = data_raw[:,:,frames[0]:frames[1]]

    height = data_raw.shape[0]
    length = data_raw.shape[1]
    nframes = data_raw.shape[2]

    data = dataprep.averageoverheight(data_raw)
    data = dataprep.standardize(data)

    lagstep = 30 
    corr = dataprep.correlate_frames(data, lagstep)
    corr = dataprep.standardize(corr)

    corr_mean = np.mean(corr[:,:-lagstep], axis=1)
    x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

    corr_mean[int(corr_mean.shape[0]/2)] = 0
    corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

    x_lag = x_lag[0:int(corr_mean.shape[0])]

    corr_mean = dataprep.standardize(corr_mean)

    window = 7
    corr_mean_smoothed = dataprep.simplemovingmean(corr_mean, window, beta=6)
    x_lag_smoothed = x_lag[int(window/2):-int(window/2)]

    with bayesian.signalmodel_correlation(corr_mean_smoothed, -x_lag_smoothed, px, lagstep, fps) as model:
        trace = pm.sample(4000, return_inferencedata=False, cores=4, target_accept=0.9)
      
        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 
        summary = az.summary(idata, var_names=["sigma_noise", "sigma", "centroid", "amplitude", "c", "velocity"])

    v = bayesian.get_mode(idata.posterior, ["velocity"])[0]*1e-6
    print(v*1e6)
    mode_velocity_in_rope = (v*1e6 > rope_velocity[0] and v*1e6 < rope_velocity[1])
    if not mode_velocity_in_rope:
        return idata, None

    data_shifted = dataprep.shift_data(data, v, fps, px)

    data_mean = np.mean(data_shifted, axis=1)

    data_mean = dataprep.standardize(data_mean)

    x = np.linspace(0, len(data_mean), len(data_mean))
    with bayesian.signalmodel(data_mean, x) as model:
        trace2 = pm.sample(4000, return_inferencedata=False, cores=4, target_accept=0.9)
    
        ppc2 = pm.fast_sample_posterior_predictive(trace2, model=model)
        idata2 = az.from_pymc3(trace=trace2, posterior_predictive=ppc2, model=model) 

    return idata, idata2

if __name__ == "__main__":
    # same for every experiment
    fps = 46 # frames per second (1/s)
    px = 1.6e-6# size of pixel (m/px)  

    rope_sigma = (5,15)
    rope_velocity = (200,250)

    channel = [27, 27]
    lagstep = 30
    
    inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"
    frames = [150,250]

    idata_cross, idata_multi = main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity)

    idata_cross.to_netcdf("idata_cross.nc")
    idata_multi.to_netcdf("idata_multi.nc")
