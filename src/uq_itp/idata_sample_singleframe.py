import os
from pprint import pprint
import numpy as np
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

from gradient_free_optimizers import *

import dataprep
import bayesian
import helper

def main(inname, channel, times):
    data_raw = helper.raw2images(inname, channel)
    data = dataprep.averageoverheight(data_raw)
    data = dataprep.standardize(data, axis=0)
    
    number = (inname.split("_")[-1]).split("/")[-1]
    number = number.replace(".nd2", "")
    folder = (inname.split("/")[-2])
    folder = folder + "/" + number
    
    from pathlib import Path
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    for time in times:
        d = data[:,time]
        
        x = np.arange(0, d.shape[0])
        with bayesian.signalmodel(d, x, artificial=True) as model:
            trace = pm.sample(4000, tune=4000, return_inferencedata=False, cores=4, target_accept=0.9)

            ppc = pm.fast_sample_posterior_predictive(trace, model=model)
            idata_single = az.from_pymc3(trace=trace, posterior_predictive=ppc, model=model)       
        
            idata_single.to_netcdf(folder+"/idata_single_t{}.nc".format(time))

if __name__ == "__main__":

    basepath = "/home/cb51neqa/projects/itp/exp_data/"

    innames = []
    for root, dirs, files in os.walk(basepath):
        for f in files:
            if "nd2" in f:
                if(f.endswith(".nd2")):
                    innames.append(os.path.join(root,f))

    innames = list(filter(lambda inname: "AF_10ng" in inname, innames))
    #innames = list(filter(lambda inname: "005" in inname, innames))
    #innames = innames[20:]
    #innames = innames[-5:-1]
    #pprint(innames)

    times = [150, 200, 250]

    channel = [27, 27]

    for inname in innames:
        j = 0
        while True:
            try:
                print(inname)
                            
                main(inname, channel, times)

            except AttributeError:
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
