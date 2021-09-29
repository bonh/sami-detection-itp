import numpy as np
from scipy.special import erf

def shift_data(data, v, fps, px):
    dx =  v/fps
    data_shifted = np.zeros(data.shape)
    for i in range(0, data.shape[1]):
        shift = data.shape[0] - int(i*dx/px)%data.shape[0]
        data_shifted[:,i] = np.roll(data[:,i], shift)
    
    data_shifted = np.roll(data_shifted, int(data_shifted.shape[1]/2), axis=1)
    
    return data_shifted

def skewNormal(x, t, v=1.0, sigma=1.0, alpha=0.0):
    '''
    Skew normal distribution.
    
    INPUT:
        x: position
        t: time
        v: velocity
        sigma: width parameter
        alpha: skewness parameter (simple Gaussian for alpha=0.0)
        
    RETURN:
        skew normal
    '''
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-v*t)**2/2/sigma**2) * (1 + erf(alpha*(x-v*t)/np.sqrt(2)/sigma))

def artificalsignal(S0=1.0, sigma=1.0, v=1.0, alpha=1.0, noise='none', nstrength=1.0, width=512, height=54, nframes=500, **kwargs):
    '''
    Creates an ITP signal.
    
    INPUT:
        S0: total intensity
        sigma: signal width
        v: signal velocity
        alpha: skewness (for alpha=0, the signal has Gaussian shape)
        noise: type of background noise ('none', 'uniform', 'log-normal')
        nstrength: noise strength
        width: frame width in px
        height: frame height in px
        nframes: number of frames
    
    RETURN:
        data: numpy array of image frames 
    '''
    # data array
    data = np.zeros((height, width, nframes))
    
    # noise array
    ndata = np.zeros((height, width, nframes))
    ln_mean = kwargs.get('ln_mean', 0.5)
    ln_sigma = kwargs.get('ln_sigma', 0.05)
    if noise == 'uniform':
        ndata = np.random.random_sample((height,width,nframes))
    elif noise == 'log-normal':
        ndata = np.random.lognormal(mean=ln_mean, sigma=ln_sigma, size=(height,width,nframes))
        
    # create signal
    x = np.linspace(0, width, width, endpoint=False)
    for i in range(nframes):
        data[:,:,i] = np.array([skewNormal(x, i, v=v, sigma=sigma, alpha=alpha)]*height)

    return data*S0 + ndata*nstrength
