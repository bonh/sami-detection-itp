import numpy as np
from nd2reader import ND2Reader
import scipy as sc

def load_nd_data(inname, startframe=0, endframe=-1, verbose=False, nth=1):
    '''
    Loads the image data from .nd2 file.

    Parameters
    ----------
    inname : str
        File name.
    startframe : int, optional
        First frame to load. The default is 0.
    endframe : int, optional
        Last frame to load. The default is -1.
    verbose : bool, optional
        If True, some information is printed during loading.
        The default is False.
    nth : int, optional
        Load each nth frame. The default is 1.

    Returns
    -------
    data : ndarray
        All image/video data in one array.
    '''
    with ND2Reader(inname) as rawimages:
        # determine metadata of the images
        height = rawimages.metadata["height"]
        width = rawimages.metadata["width"]
        nframes = rawimages.metadata["num_frames"]
        
        if verbose:
            print("\nheight = {}, width = {}, nframes = {}".format(height, width, nframes))
        
        if endframe == -1:
            end = nframes
        else:
            if endframe < nframes:
                end = endframe
            else:
                end = nframes
                print("endframe < len(rawimages)")
    
        # Y x X x N
        #data = np.zeros((height, width, end))
        data = np.zeros((height, width, len(np.arange(startframe, end, nth))))
    
        #data = rawimages
        #print(np.mean(data)
    
        # load image data into data array
        #for frame in np.arange(startframe, end, 1):
        #    data[:,:,frame] = rawimages[frame]
        for i,frame in enumerate(np.arange(startframe, end, nth)):
            data[:,:,i] = rawimages[frame]
        
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
    if fps and px:
        v_px = v/(fps*px)
    else:
        v_px = v
        
    #offset = int(data.shape[0]/3)
    offset = 0
    
    data_shifted = np.zeros(data.shape)

    for i in range(0, data.shape[1]):
        shift = data.shape[0] - int(i*v_px)+offset
        if shift-offset-data.shape[0] == 0:
            print("shift is zero")
        data_shifted[:,i] = np.roll(data[:,i], shift)
    
    #data_shifted = np.roll(data_shifted, int(data_shifted.shape[1]/2), axis=1)
    
    return data_shifted

def correlate_frames(data, step):
    corr = np.zeros((data.shape[0], data.shape[1]-step))
    for i in range(0,data.shape[1]-step):
        corr[:,i] = np.correlate(data[:,i], data[:,i+step], "same")

    return corr

def standardize(data, axis=None):
    return (data-np.mean(data, axis=axis))/np.std(data, axis=axis)

def simplemovingmean(data, window, beta=0):
    window_ = np.kaiser(window, beta)
    window_ = window_ / window_.sum()
    return np.convolve(window_, data, mode='valid')

def fourierfilter(data, rx, ry, rotation, horizontal, vertical):
    ff = np.fft.fft2(data)
    ff = np.fft.fftshift(ff)
       
    X, Y = ff.shape
        
    window_y = sc.signal.windows.gaussian(Y, std=ry)[:,None]
    window_x = sc.signal.windows.gaussian(X, std=rx)[:,None]
    window2d = np.sqrt(np.dot(window_x, window_y.T)) # expand to 2D
    window2d = sc.ndimage.interpolation.rotate(window2d, angle=rotation, reshape=False)
    
    # remove strong horizontal and vertical frequency components
    if horizontal:
        window2d[int(X/2),:] = 0
    if vertical:
        window2d[:,int(Y/2)] = 0
    
    ffw = window2d * ff
    
    iff = np.fft.ifftshift(ffw)
    iff = np.fft.ifft2(iff)
    
    return np.real(iff), window2d, ff

def correlation(data, startframe, endframe, lagstepstart=30, deltalagstep=5, N=8):
    length = int(data.shape[0]/2)
    corr_mean_combined = np.zeros((N, length))

    for i in range(0, N):
        lagstep = lagstepstart + i*deltalagstep
        corr = correlate_frames(data, lagstep)
        corr = standardize(corr)
        
        corr_mean = np.mean(corr[:,startframe:endframe-lagstep], axis=1)
        
        # clean the correlation data
        # remove peak at zero lag
        corr_mean[length] = 0
        #cut everything right of the middle (because we know that the velocity is positiv)
        corr_mean = corr_mean[0:length]
        
        #window = 7
        #corr_mean = simplemovingmean(corr_mean, window, beta=6)
    
        corr_mean_combined[i,:] = standardize(corr_mean)
        
    x_lag = np.arange(-length, 0)
        
    return x_lag, corr_mean_combined