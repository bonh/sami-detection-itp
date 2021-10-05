import numpy as np
from sklearn import preprocessing
import pymc3 as pm

import dataprep
import bayesian

def main(inname, channel, lagstep, frames):
    data_raw = dataprep.load_nd_data(inname, verbose=False)
    data_raw = dataprep.cuttochannel(data_raw, channel[0], channel[1])
    background = np.mean(data_raw[:,:,0:10],axis=2)
    data_raw = dataprep.substractbackground(data_raw, background)
    data = dataprep.averageoverheight(data_raw)
    
    corr = dataprep.correlate_frames(data, lagstep)
    
    corr = corr[:,frames[0]:frames[1]]
    
    corr_mean = np.mean(corr, axis=1).reshape(-1, 1)
    
    x = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])
    corr_mean[int(corr_mean.shape[0]/2)] = 0
    corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]
    x = x[0:int(corr_mean.shape[0])]
    x *= -1
    
    scaler = preprocessing.MinMaxScaler().fit(corr_mean)
    corr_mean = scaler.transform(corr_mean).flatten()
    
    with bayesian.create_signalmodel(corr_mean, x) as model:
        trace_mean = pm.sample(20000, return_inferencedata=True, cores=4)
    
    with bayesian.create_model_noiseonly(corr_mean) as model:
        trace_noiseonly = pm.sample(10000, return_inferencedata=True)
        
    dfwaic = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="waic")
    dfloo = pm.compare({"sample":trace_mean, "noiseonly":trace_noiseonly}, ic="loo")
    
    return dfwaic, dfloo
    
if __name__ == "__main__":
    
    inname = "/home/cb51neqa/projects/itp/exp_data/ITP_AF647_5µA/AF_0.1ng_l/001.nd2"
    channel = [27, 27]
    lagstep = 30
    frames = [100,300]
    dfwaic, dfloo = main(inname, channel, lagstep, frames)
    
    print(dfwaic)
    print(dfloo)
