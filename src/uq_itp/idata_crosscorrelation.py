import os
import sys
from os.path import expanduser
import multiprocessing as mp
from pprint import pprint
import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

from gradient_free_optimizers import *

import dataprep
import bayesian
import helper

class MyCallback:
    def __init__(self, model, every=1000, max_rhat=1.05):
        self.model = model
        self.every = every
        self.max_rhat = max_rhat
        self.traces = {}
        self.uptodate = [False, False, False, False]
        self.count = 0

    def __call__(self, trace, draw):
        if draw.tuning:
            return

        self.count += int(draw.stats[0]['diverging'])
        if self.count > 10:
            raise RuntimeError

        self.traces[draw.chain] = trace
        if len(trace) % self.every == 0:
            multitrace = pm.backends.base.MultiTrace(list(self.traces.values()))
            if pm.stats.rhat(multitrace).to_array().max() < self.max_rhat:
                raise KeyboardInterrupt

def main(inname, channel, lagstep, px, fps, data_raw=None, startframe=None, delta=None, artificial=False):
    if inname and channel:
        data_raw = helper.raw2images(inname, channel)

    if len(data_raw.shape) > 2:
        data = dataprep.averageoverheight(data_raw)
    else:
        data = data_raw
    data = dataprep.standardize(data)

    fft2 = np.fft.fft2(data)
    
    mask = np.zeros(data.shape)
    for y, x in np.ndindex(mask.shape):
        if x>0 and x < 30 and y > 490:
            mask[y, x] = 1
    
    tmp = np.fft.ifft2(fft2*mask)
    tmp = np.abs(tmp)
    tmp = dataprep.standardize(tmp)
    data = tmp

    corr = dataprep.correlate_frames(data, lagstep)
    corr = dataprep.standardize(corr)

    def functional(parameters):
        startframe = parameters["start"]
        delta = parameters["delta"]

        print(startframe, delta)

        corr_mean = np.mean(corr[:,startframe:startframe+delta], axis=1)
        
        x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

        corr_mean[int(corr_mean.shape[0]/2)] = 0
        corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

        x_lag = x_lag[0:int(corr_mean.shape[0])]

        corr_mean = dataprep.standardize(corr_mean)

        window = 7
        corr_mean_smoothed = dataprep.simplemovingmean(corr_mean, window, beta=6)
        x_lag_smoothed = x_lag[int(window/2):-int(window/2)]

        with bayesian.signalmodel_correlation(corr_mean_smoothed, -x_lag_smoothed, px, lagstep, fps, artificial=artificial) as model:
            try:
                trace = pm.sample(10000, tune=4000, return_inferencedata=False, cores=1, chains=4, target_accept=0.9, callback=MyCallback(model))
            except RuntimeError:
                print("Divergence!")
                return -1e5
            except KeyboardInterrupt:
                print("Converged!")

        idata = az.from_pymc3(trace=trace, model=model) 
        
        hdi_velocity = az.hdi(idata, var_names=["velocity"])["velocity"]
        s = hdi_velocity[1] - hdi_velocity[0]
        result = -1/2*np.sqrt(s**2)# + 1e-4*delta
        result = result.item()

        v = bayesian.get_mode(idata.posterior, ["velocity"])[0]*1e-6
        print(v*1e6)

        return result

    if not startframe and not delta:  
        #delta = 200
            
        search_space = {"start": np.arange(0, 300, 10), "delta": np.arange(0, 300, 10)}
        #search_space = {"start": np.arange(100, 300, 10)}

        #opt = RandomSearchOptimizer(search_space)
        opt = BayesianOptimizer(search_space)
        #opt = EvolutionStrategyOptimizer(search_space)
        #opt = SimulatedAnnealingOptimizer(search_space)
        opt.search(functional, n_iter=20, early_stopping={"n_iter_no_change":5})

        startframe = opt.best_para["start"]
        endframe = startframe + opt.best_para["delta"]
        
    endframe = startframe + delta
    print(startframe, endframe)

    corr_mean = np.mean(corr[:,startframe:endframe], axis=1)
        
    x_lag = np.linspace(-corr_mean.shape[0]/2, corr_mean.shape[0]/2, corr_mean.shape[0])

    corr_mean[int(corr_mean.shape[0]/2)] = 0
    corr_mean = corr_mean[0:int(corr_mean.shape[0]/2)]

    x_lag = x_lag[0:int(corr_mean.shape[0])]

    corr_mean = dataprep.standardize(corr_mean)

    window = 7
    corr_mean_smoothed = dataprep.simplemovingmean(corr_mean, window, beta=6)
    x_lag_smoothed = x_lag[int(window/2):-int(window/2)]

    with bayesian.signalmodel_correlation(corr_mean_smoothed, -x_lag_smoothed, px, lagstep, fps, artificial=artificial) as model:
        trace = pm.sample(10000, tune=4000, return_inferencedata=False, cores=1, chains=4, target_accept=0.9, callback=MyCallback(model))

        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        prior_pc = pm.sample_prior_predictive(50000, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model, prior=prior_pc) 

    return idata, startframe, endframe 

def run(inname, channel, lagstep, px, fps):
    sys.stdout = open(str(os.getpid()) + ".out", "w")
    sys.stderr = open(str(os.getpid()) + ".err", "w")
    j = 0
    while True:
        try:
            print(inname)
            idata_cross, min_, max_ = main(inname, channel, lagstep, px, fps)
        except Exception as e:
            print(e)
            if j<3:
                print("retry")
                j+=1
                continue
            else:
                break

        number = (inname.split("_")[-1]).split("/")[-1]
        number = number.replace(".nd2", "")
        folder = (inname.split("/")[-2])

        folder = folder + "/" + number

        from pathlib import Path
        Path(folder).mkdir(parents=True, exist_ok=True)

        idata_cross.to_netcdf(folder+"/idata_cross.nc")
        with open(folder+"/intervals.dat", 'w') as f:
            f.write("{}\t{}".format(min_, max_))
        
        print(folder+"/idata_cross.nc")

        break

if __name__ == "__main__":

    home = expanduser("~")
    basepath = home + "/projects/itp/exp_data/2021-12-20/5µA"

    innames = []
    for root, dirs, files in os.walk(basepath):
        for f in files:
            if "nd2" in f:
                if(f.endswith(".nd2")):
                    innames.append(os.path.join(root,f))

    #innames = list(filter(lambda inname: "10pg_l" in inname or "1pg_l" in inname, innames))
    #innames = innames[20:]
    #innames = innames[-5:-1]
    #pprint(innames)

    # same for every experiment
    fps = 46 # frames per second (1/s)
    px = 1.6e-6# size of pixel (m/px)  

    channel = [27, 27]
    lagstep = 30

    cores = mp.cpu_count()
    threads = int(cores)

    with mp.Pool(threads) as pool:
        multiple_results = [pool.apply_async(run, (inname, channel, lagstep, px, fps)) for inname in innames]
        print([res.get() for res in multiple_results])
