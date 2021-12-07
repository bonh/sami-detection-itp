import numpy as np
import pymc3 as pm
from pymc3.model import modelcontext
from scipy import dot
from scipy.linalg import cholesky as chol
import warnings
import arviz as az

from scipy import dot
from scipy.linalg import cholesky as chol
import scipy.special as spf
import scipy.stats as st
import warnings

import theano
import theano.tensor as tt

def check_rope(values, rope):
    prob = ((values > rope[0]) & (values <= rope[1])).mean()
    return prob.data

def check_refvalue(values, refvalue):
    prob = (values > refvalue).mean()
    return prob.data

def get_mode(data, var_names):
    _, vals = az.sel_utils.xarray_to_ndarray(data, var_names=var_names)
    return [az.plots.plot_utils.calculate_point_estimate("mode", val) for val in vals]

def signalmodel_correlation(data, x, px, lagstep, fps):
    with pm.Model() as model:
        # background
        # f = b*x + c
        #aa = pm.Normal("a", 0, 0.0001)
        b = pm.Normal('b', 0, 1)
        c = pm.Normal('c', 0, 1)
        
        background = pm.Deterministic("background", b*x+c)
        #background = pm.Deterministic("background", c)

        # sample peak
        amp = pm.HalfNormal('amplitude', 20) 
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.HalfNormal('sigma', 50) # TODO: calculate from physics?
        #alpha = pm.Normal("alpha", 0, 0.1)

        def sample(amp, cent, sig, x):        
            return amp*tt.exp(-(cent - x)**2/2/sig**2)
        
        sample = pm.Deterministic("sample", sample(amp, cent, sig, x))
        
        # background + sample
        signal = pm.Deterministic('signal', background + sample)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        velocity = pm.Deterministic("velocity", cent*px/(lagstep/fps)*1e6)
        snr = pm.Deterministic("snr", amp/sigma_noise)

        return model
    
def model_sample(a, c, w, alpha, x):       
    return a*tt.exp(-(c - x)**2/2/w**2) * (1-tt.erf((alpha*(c - x))/tt.sqrt(2)/w))

def signalmodel(data, x):
    with pm.Model() as model:
        # background
        # f = c
        c = pm.Normal('c', 0, 1)
        b = pm.Normal('b', 0, 1)
        #d = pm.Normal('d', 0, 1)
        
        #background = pm.Deterministic("background", d*x**2+b*x+c)
        background = pm.Deterministic("background", b*x+c)

        # sample peak
        amp = pm.HalfNormal('amplitude', 5)
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.HalfNormal('sigma', 20) # TODO: calculate from physics?
        #sig = pm.Deterministic("sigma", pm.Beta('beta', 2, 2)*20)# TODO: calculate from physics?
        alpha = pm.Normal("alpha", 0, 1e-2)
        
        sample = pm.Deterministic("sample", model_sample(amp, cent, sig, alpha, x))
        
        # background + sample
        signal = pm.Deterministic('signal', background + sample)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        def fmax(A, c, sigma, a):
            erf = tt.erf
            sqrt = tt.sqrt
            pi = np.pi
            exp = tt.exp
            Abs = pm.math.abs_
            sign = pm.math.sgn
            return A*(erf(sqrt(2)*a*(-sqrt(2)*a*(2 - pi/2)/(pi**(3/2)*sqrt(a**2 + 1)*(-2*a**2/(pi*(a**2 + 1)) + 1)**1.0) + sqrt(2)*a/(sqrt(pi)*sqrt(a**2 + 1)) - exp(-2*pi/Abs(a))*sign(a)/2)/2) + 1)*exp(-(-sqrt(2)*a*(2 - pi/2)/(pi**(3/2)*sqrt(a**2 + 1)*(-2*a**2/(pi*(a**2 + 1)) + 1)**1.0) + sqrt(2)*a/(sqrt(pi)*sqrt(a**2 + 1)) - exp(-2*pi/Abs(a))*sign(a)/2)**2/2)

        fmax = pm.Deterministic("fmax", fmax(amp, cent, sig, alpha))
        snr = pm.Deterministic("snr", fmax/sigma_noise)

        return model
