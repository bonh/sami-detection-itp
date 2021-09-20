import numpy as np
from nd2reader import ND2Reader

def load_nd_data(inname, startframe=0, endframe=-1):
    with ND2Reader(inname) as rawimages:
        print(rawimages)
        rawimages.bundle_axes = 'yx' # defines which axes will be present in a single frame
        rawimages.iter_axes = 't' # defines which axes will be the index axis; t-axis is the time axis
    
        # determine metadata of the images
        height = rawimages.metadata["height"]
        width = rawimages.metadata["width"]
        nframes = rawimages.metadata["num_frames"]
        
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
    
        print("\ndata shape = {}".format(data.shape))
        return data