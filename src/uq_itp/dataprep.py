import numpy as np
from nd2reader import ND2Reader

def load_nd_data(inname, startframe=0, endframe=-1, verbose=False):
    with ND2Reader(inname) as rawimages:
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

def shift_data(data, v, fps, px):
    dx =  v/fps
    data_shifted = np.zeros(data.shape)
    for i in range(0, data.shape[1]):
        shift = data.shape[0] - int(i*dx/px)%data.shape[0]
        data_shifted[:,i] = np.roll(data[:,i], shift)
    
    #data_shifted = np.roll(data_shifted, int(data_shifted.shape[1]/2), axis=1)
    
    return data_shifted

def correlate_frames(data, step):
    corr = np.zeros(data.shape)
    for i in range(0,data.shape[1]-step):
        corr[:,i] = np.correlate(data[:,i], data[:,i+step], "same")

    return corr
