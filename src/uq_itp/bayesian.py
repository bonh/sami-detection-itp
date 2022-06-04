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

def check_rope_hdi(hdi, rope):
    return hdi[0]>=rope[0] and hdi[1]<=rope[1]

def check_refvalue(values, refvalue):
    prob = (values > refvalue).mean()
    return prob.data

def check_refvalue_hdilow(hdilow, refvalue):
    return hdilow>=refvalue

def get_mode(data, var_names):
    _, vals = az.sel_utils.xarray_to_ndarray(data, var_names=var_names)
    return [az.plots.plot_utils.calculate_point_estimate("mode", val) for val in vals]

def signalmodel_correlation(data, x, px, deltalagstep, lagstepstart, fps, artificial=False):
    
    N = x.shape[1]
    length = x.shape[0]
    
    with pm.Model() as model:
        # background
        # f = b*x + c
        #aa = pm.Normal("a", 0, 0.0001)
        c = pm.Normal('c', 0, 1, shape=1)
        if not artificial:
            b = pm.Normal('b', 0, 1, shape=1)
            background = pm.Deterministic("background", b*x+c)
        else:
            background = pm.Deterministic("background", c)

        # sample peak
        amp = pm.HalfNormal('amplitude', 10, shape=1)
        measure = pm.Uniform("measure", 0, 1, shape=1)
        cent = pm.Deterministic('centroid', measure*length)
        if N > 1:
            deltacent = pm.HalfNormal("deltac", 20, shape=1)
        else:
            deltacent = 0
        sig = pm.HalfNormal('sigma', 50, shape=1) # TODO: calculate from physics?
        #alpha = pm.Normal("alpha", 0, 0.1)
        
        def sample(amp, cent, deltacent, sig, x):
            n = np.arange(0,N)
            return amp*tt.exp(-(cent + deltacent*n - x)**2/2/sig**2)
        
        sample = pm.Deterministic("sample", sample(amp, cent, deltacent, sig, x))
        
        # background + sample
        signal = pm.Deterministic('signal', background + sample)

        # prior noise
        sigma_noise = pm.HalfNormal('sigmanoise', 1, shape=1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        velocitypx = pm.Deterministic("velocitypx", (cent + deltacent)/(lagstepstart + deltalagstep))
        velocity = pm.Deterministic("velocity", velocitypx*px*fps*1e6)

        return model
    
def model_sample(a, c, w, alpha, x):       
    return a*tt.exp(-(c - x)**2/2/w**2) * (1-tt.erf((alpha*(c - x))/tt.sqrt(2)/w))

# derived quantities
def fmax(A, c, sigma, a):
    erf = tt.erf
    sqrt = tt.sqrt
    pi = np.pi
    exp = tt.exp
    Abs = pm.math.abs_
    sign = pm.math.sgn
    return A*(erf(sqrt(2)*a*(-sqrt(2)*a*(2 - pi/2)/(pi**(3/2)*sqrt(a**2 + 1)*(-2*a**2/(pi*(a**2 + 1)) + 1)**1.0) + sqrt(2)*a/(sqrt(pi)*sqrt(a**2 + 1)) - exp(-2*pi/Abs(a))*sign(a)/2)/2) + 1)*exp(-(-sqrt(2)*a*(2 - pi/2)/(pi**(3/2)*sqrt(a**2 + 1)*(-2*a**2/(pi*(a**2 + 1)) + 1)**1.0) + sqrt(2)*a/(sqrt(pi)*sqrt(a**2 + 1)) - exp(-2*pi/Abs(a))*sign(a)/2)**2/2)

def signalmodel(data, x, artificial=False):
    with pm.Model() as model:
        # background
        # f = c
        c = pm.Normal('c', 0, 0.01)
        if not artificial:
            b = pm.Normal('b', 0, 0.01)
            #d = pm.Normal('d', 0, 1)
            #background = pm.Deterministic("background", d*x**2+b*x+c)
            background = pm.Deterministic("background", b*x+c)
        else:
            background = pm.Deterministic("background", c)

        # sample peak
        amp = pm.HalfNormal('amplitude', 10)
        measure = pm.Uniform("measure", 0, 1)
        cent = pm.Deterministic('centroid', measure*len(data))
        sig = pm.HalfNormal('sigma', 20) # TODO: calculate from physics?
        #sig = pm.Deterministic("sigma", pm.Beta('beta', 2, 2)*20)# TODO: calculate from physics?
        alpha = pm.Normal("alpha", 0, 0.1)
        
        sample = pm.Deterministic("sample", model_sample(amp, cent, sig, alpha, x))
        
        # background + sample
        signal = pm.Deterministic('signal', background + sample)

        # prior noise
        sigma_noise = pm.HalfNormal('sigmanoise', 1.0) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)

        fmax_ = pm.Deterministic("fmax", fmax(amp, cent, sig, alpha))
        snr = pm.Deterministic("snr", fmax_/sigma_noise)

        return model
