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

#        self.traces[draw.chain] = trace
#        if len(trace) % self.every == 0:
#            multitrace = pm.backends.base.MultiTrace(list(self.traces.values()))
#            if pm.stats.rhat(multitrace).to_array().max() < self.max_rhat:
#                raise KeyboardInterrupt

def main(inname, channel, lagstep, px, fps, data_raw=None, startframe=None, delta=None, artificial=False, cores=1, samples=5000, tune=5000):
    if inname and channel:
        data_raw = helper.raw2images(inname, channel)

    if len(data_raw.shape) > 2:
        data = dataprep.averageoverheight(data_raw)
    else:
        data = data_raw
    data = dataprep.standardize(data, axis=0)

    data, _, _ = dataprep.fourierfilter(data, 100, 40/4, -45, True, True)
    data = dataprep.standardize(data, axis=0)

    global best_trace
    global best_model
    best_trace = None
    best_model = None
    
    def functional(parameters):
        startframe = parameters["start"]
        delta = parameters["delta"]
        endframe = startframe + delta

        print(startframe, delta)

        N = 8
        deltalagstep = 5
        lagstepstart = 30
        x_lag, corr_mean_combined = dataprep.correlation(data, startframe, endframe, lagstepstart=lagstepstart, deltalagstep=deltalagstep, N=N)
        
        result_old = -1e6
        
        j = 0
        while True:
            try:
                print(inname)
                with bayesian.signalmodel_correlation(corr_mean_combined.T, -np.array([x_lag,]*N).T, px, deltalagstep, lagstepstart, fps, artificial=artificial) as model:
                    trace = pm.sample(samples, tune=tune, return_inferencedata=False, cores=cores, chains=4, target_accept=0.9)

                    idata = az.from_pymc3(trace=trace, model=model) 

                    hdi_velocity = az.hdi(idata, var_names="velocity").velocity.values[0]
                    s = hdi_velocity[1] - hdi_velocity[0]
                    result = -1/2*np.sqrt(s**2)# + 1e-4*delta

                    global best_trace
                    global best_model
                    if result > result_old:
                        result_old = result

                        best_trace = trace
                        best_model = model
                        
                    return result
            except Exception as e:
                print(e)
                if j<3:
                    print("retry")
                    j+=1
                    continue
                else:
                    return -1e5
            
    if not startframe and not delta:  
        if True:      
            search_space = {"start": np.arange(50, 300, 25), "delta": np.arange(150, 250, 25)}
            initialize = {"warm_start": [{"start": 100, "delta": 200}]}

            #opt = RandomSearchOptimizer(search_space)
            opt = BayesianOptimizer(search_space, initialize=initialize)
            #opt = EvolutionStrategyOptimizer(search_space)
            #opt = SimulatedAnnealingOptimizer(search_space)
            opt.search(functional, n_iter=1, early_stopping={"n_iter_no_change":5}, max_score=-1)

            startframe = opt.best_para["start"]
            delta = opt.best_para["delta"]
            endframe = startframe + delta
        
        else:
            import nevergrad as ng

            parametrization = ng.p.Instrumentation(
                # an integer from 1 to 12
                start=ng.p.Scalar(lower=50, upper=300).set_integer_casting(),
                delta=ng.p.Scalar(lower=150, upper=250).set_integer_casting()

            )

            def functional_ng(start: int, delta: int) -> float:
                parameters = {"start":start, "delta":delta}
                return -1*functional(parameters)

            from concurrent import futures
            optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=1)
            recommendation = optimizer.minimize(functional_ng)

            startframe = recommendation.kwargs["start"]
            delta = recommendation.kwargs["delta"]
            endframe = startframe + delta

        print("Optimal values: ", startframe, endframe, delta)     
        
    else:
        endframe = startframe + delta

    with best_model as model:
        ppc = pm.fast_sample_posterior_predictive(best_trace, model=model)
        idata = az.from_pymc3(trace=best_trace, posterior_predictive=ppc, model=model) 

    return idata, startframe, endframe 

def run(inname, channel, lagstep, px, fps, cores=1):

        idata_cross, min_, max_ = main(inname, channel, lagstep, px, fps, cores=cores)


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


if __name__ == "__main__":

    home = expanduser("~")
    basepath = home + "/projects/itp/exp_data/2021-12-20/5µA"

    innames = []
    for root, dirs, files in os.walk(basepath):
        for f in files:
            if "nd2" in f:
                if(f.endswith(".nd2")):
                    innames.append(os.path.join(root,f))

    innames = list(filter(lambda inname: "_0ng_l" in inname, innames))
    #innames = list(filter(lambda inname: "001.nd2" in inname, innames))
    #innames = innames[20:]
    #innames = innames[0:1]
    pprint(innames)

    # same for every experiment
    fps = 46 # frames per second (1/s)
    px = 1.6e-6# size of pixel (m/px)  

    channel = [27, 27]
    lagstep = 30

    for inname in innames:
        run(inname, channel, lagstep, px, fps, cores=4)

    #cores = mp.cpu_count()
    #threads = int(cores)

    #with mp.Pool(threads) as pool:
    #    multiple_results = [pool.apply_async(run, (inname, channel, lagstep, px, fps)) for inname in innames]
    #    print([res.get() for res in multiple_results])
