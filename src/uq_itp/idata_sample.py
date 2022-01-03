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

def main(inname, channel, px, fps, rope_velocity, idata_cross, startframe, endframe, data_raw=None, artificial=False):
    if inname and channel:
        data_raw = helper.raw2images(inname, channel)
        
    if len(data_raw.shape) > 2:
        data = dataprep.averageoverheight(data_raw)
    else:
        data = data_raw
    data = dataprep.standardize(data)
    
    v = bayesian.get_mode(idata_cross.posterior, ["velocity"])[0]*1e-6
    print(v*1e6)
    mode_velocity_in_rope = (v*1e6 > rope_velocity[0] and v*1e6 < rope_velocity[1])
    if not mode_velocity_in_rope:
        return None

    data_shifted = dataprep.shift_data(data, v, fps, px)

    data_mean = np.mean(data_shifted[:,startframe:endframe], axis=1)
    data_mean = dataprep.standardize(data_mean)

    x = np.linspace(0, len(data_mean), len(data_mean))
    with bayesian.signalmodel(data_mean, x, artificial=artificial) as model:
        trace = pm.sample(16000, tune=8000, return_inferencedata=False, chains=4, cores=1, target_accept=0.9)
    
        ppc = pm.fast_sample_posterior_predictive(trace, model=model)
        idata = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model) 

    return idata 

def run(inname, channel, px, fps, rope_velocity):
   j = 0
   while True:
       try:
           print(inname)
           
           number = (inname.split("_")[-1]).split("/")[-1]
           number = number.replace(".nd2", "")
           folder = (inname.split("/")[-2])
           folder = folder + "/" + number

           idata_cross = az.InferenceData.from_netcdf(folder+"/idata_cross.nc")
                       
           with open(folder+"/intervals.dat") as f:
               min_, max_ = [int(x) for x in next(f).split()]

           print(min_, max_)
           idata = main(inname, channel, px, fps, rope_velocity, idata_cross, min_, max_)

           idata.to_netcdf(folder+"/idata.nc")
       except AttributeError as e:
           print(e)
           break
       except Exception as e:
           print(e)
           if j<3:
               print("retry")
               j+=1
               continue
           else:
               break

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

    #innames = list(filter(lambda inname: "AF_1ng" in inname, innames))
    #innames = list(filter(lambda inname: "005" in inname, innames))
    #innames = innames[20:]
    #innames = innames[-5:-1]
    #pprint(innames)

    # same for every experiment
    fps = 46 # frames per second (1/s)
    px = 1.6e-6# size of pixel (m/px)  

    channel = [27, 27]

    rope_velocity = (100,200)

    cores = mp.cpu_count()
    threads = int(cores)

    with mp.Pool(threads) as pool:
        multiple_results = [pool.apply_async(run, (inname, channel, px, fps, rope_velocity)) for inname in innames]
        print([res.get() for res in multiple_results])
