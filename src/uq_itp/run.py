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

i = 0
for inname in innames:
    # not the same for every experiment.
    # however, we are missing a good automatic decision which frames to include
    frames = [150,200]

    detected, idata = detection.main(inname, channel, lagstep, frames, px, fps, rope_sigma, rope_velocity)

    title = inname.split("_")[-2]
    title = title.replace("ng", "")

    results[i,:] = [title, detected]
    i += 1

print(results)
np.savetxt("detection.dat", results)
