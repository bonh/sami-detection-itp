import os
from pprint import pprint
import numpy as np

import detection

basepath = "/home/cb51neqa/projects/itp/exp_data/"

innames = []
for root, dirs, files in os.walk(basepath):
    for f in files:
        if "nd2" in f:
            if(f.endswith(".nd2")):
                innames.append(os.path.join(root,f))

innames = list(filter(lambda inname: "AF_0ng" in inname, innames))
#innames = innames[20:]
#innames = innames[-5:-1]
pprint(innames)

# same for every experiment
fps = 46 # frames per second (1/s)
px = 1.6e-6# size of pixel (m/px)  

rope_sigma = (5,15)
rope_velocity = (200,250)

channel = [27, 27]
lagstep = 30

titles, detected = [], []

results = np.empty((len(innames), 2))

for inname in innames:
    # not the same for every experiment.
    # however, we are missing a good automatic decision which frames to include
    frames = [150,250]

    j = 0
    while True:
        try:
            idata_cross, idata_multi = detection.main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity)
        except:
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
        if idata_multi:
            idata_multi.to_netcdf(folder+"/idata_multi.nc")

        break
