import numpy as np
from nd2reader import ND2Reader

def load_nd_data(inname, startframe=0, endframe=-1, verbose=False):
    with ND2Reader(inname) as rawimages:
        rawimages.bundle_axes = 'yx' # defines which axes will be present in a single frame
        rawimages.iter_axes = 't' # defines which axes will be the index axis; t-axis is the time axis
    
        # determine metadata of the images
        height = rawimages.metadata["height"]
        width = rawimages.metadata["width"]
        nframes = rawimages.metadata["num_frames"]
        
        if verbose:
            print("\nheight = {}, width = {}, nframes = {}".format(height, width, nframes))
        
        if endframe == -1:
            end = len(rawimages)
        else:
            if endframe < nframes:
                end = endframe
            else:
                end = nframes
                print("endframe < len(rawimages)")
    
        # Y x X x N
        data = np.zeros((height, width, end))
    
        #data = rawimages
        #print(np.mean(data)
    
        # load image data into data array
        for frame in np.arange(startframe, end, 1):
            data[:,:,frame] = rawimages[frame]
    
        if verbose:
            print("\ndata shape = {}".format(data.shape))
        return data

def cuttochannel(data, upperrow, lowerrow):
    height = data.shape[0]
    frow = int(height/2 - upperrow)
    lrow = int(height/2 + lowerrow)
    return data[frow:lrow,:,:]

def substractbackground(data, background):
    return data - background.reshape(background.shape[0], background.shape[1],1)

def averageoverheight(data):
    return np.mean(data, axis=0)
