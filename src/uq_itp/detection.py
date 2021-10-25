import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

import dataprep
import bayesian

def main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity):
    data_raw = dataprep.load_nd_data(inname, verbose=False)
    data_raw = dataprep.cuttochannel(data_raw, channel[0], channel[1])
    background = np.mean(data_raw[:,:,0:50],axis=2)
    data_raw = dataprep.substractbackground(data_raw, background)
    data = dataprep.averageoverheight(data_raw)
    
    corr = dataprep.correlate_frames(data, lagstep)
    
    corr_mean = np.mean(corr[:,frames[0]:frames[1]], axis=1).reshape(-1, 1)
    
    #corr_mean[int(corr_mean.shape[0]/2)] = 0
    corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
    
    scaler = preprocessing.MinMaxScaler().fit(corr_mean)
    corr_mean = scaler.transform(corr_mean).flatten()
    
    x = np.linspace(0, len(corr_mean), len(corr_mean))
    model, trace = bayesian.fit_crosscorrelationmodel(corr_mean, x, px, lagstep, fps)
    
    idata = az.from_pymc3(trace=trace, model=model) 
    detected = bayesian.check_rope(idata.posterior["sigma"], rope_sigma) > .95 and bayesian.check_rope(idata.posterior["velocity"], rope_velocity) > .95
    
    return detected.astype(int), idata
    
if __name__ == "__main__":
    # same for every experiment
    fps = 46 # frames per second (1/s)
    px = 1.6e-6# size of pixel (m/px)  
    rope_sigma = (5,15)
    rope_velocity = (200,250)
    channel = [27, 27]
    lagstep = 30
    
    inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"
    frames = [150,200]

    detected = main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity)
    print(detected)
