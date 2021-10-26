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

def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

def check_rope(values, rope):
    prob = ((values > rope[0]) & (values <= rope[1])).mean()
    return prob.data

def signalmodel_correlation(data, x, px, lagstep, fps):
    with pm.Model() as model:
        # background
        # f = b*x + c
        #aa = pm.Normal("a", 0, 0.0001)
        b = pm.Normal('b', 0, 1)
        c = pm.Normal('c', 0, 1)
        
        background = pm.Deterministic("background", b*x+c)

        # sample peak
        amp = pm.Uniform('amplitude', 0, 2) 
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.Uniform('sigma', 0, 100) # TODO: calculate from physics?
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

def signalmodel(data, x):
    with pm.Model() as model:
        # background
        # f = b*x + c
        #aa = pm.Normal("a", 0, 0.0001)
        b = pm.Normal('b', 0, 0.1)
        c = pm.Normal('c', 0, 1)
        
        background = pm.Deterministic("background", b*x+c)

        # sample peak
        amp = pm.Uniform('amplitude', 0, 2) 
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.Uniform('sigma', 0, 100) # TODO: calculate from physics?
        alpha = pm.Normal("alpha", 0, 0.01)

        def sample(amp, cent, sig, x):       
            return amp*tt.exp(-(cent - x)**2/2/sig**2) * (1-tt.erf((alpha*(cent - x))/tt.sqrt(2)/sig))
        
        sample = pm.Deterministic("sample", sample(amp, cent, sig, x))
        
        # background + sample
        signal = pm.Deterministic('signal', background + sample)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        snr = pm.Deterministic("snr", amp/sigma_noise)

        return model
    
def fit_crosscorrelationmodel(data, x, px, lagstep, fps):
    with pm.Model() as model:
        # background, f = b*x + c
        b = pm.Normal('b', 0, 1)
        c = pm.Normal('c', 0, 1)
        
        background = pm.Deterministic("background", b*x+c)

        # signalpeak
        amp = pm.Uniform('amplitude', 0, 2) 
        cent = pm.Uniform('centroid', 0, len(data))
        sig = pm.Uniform('sigma', 0, 100) # TODO: calculate from physics?

        # Gaussian
        def model_signal(amp, cent, sig, x):
            return amp*np.exp(-1*(cent - x)**2/2/sig**2)

        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, x))

        # noise
        sigma_noise = pm.HalfNormal('sigma_noise', 1) # TODO: can we estimate a prior value from zero concentration images?

        # likelihood       
        likelihood = pm.Normal('y', mu = background+signal, sd=sigma_noise, observed = data)
        
        # derived quantities
        velocity = pm.Deterministic("velocity", (len(data)-cent)*px/(lagstep/fps)*1e6)
        snr = pm.Deterministic("snr", amp/sigma_noise)

        # sample the model
        trace = pm.sample(3000, tune=2000, return_inferencedata=False, cores=4)
        
        return model, trace

def create_signalmodel(data, x):
    with pm.Model() as model:
        # prior peak
        amp = pm.HalfNormal('amplitude', 1)
        cent = pm.Normal('centroid', 125, 50)
        base = pm.Normal('baseline', 0, 0.5)
        sig = pm.HalfNormal('sigma', 10)

        # forward model signal
        signal = pm.Deterministic('signal', model_signal(amp, cent, sig, base, x))

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = signal, sd=sigma_noise, observed = data)
        
        return model
    
def create_model_noiseonly(data):
    model = pm.Model()
    with model:
        # baseline only
        base = pm.Normal('baseline', 0, 0.5)

        # prior noise
        sigma_noise = pm.HalfNormal('sigma_noise', 0.1)

        # likelihood
        likelihood = pm.Normal('y', mu = base, sd=sigma_noise, observed = data)

        return model

# Based on https://github.com/quentingronau/bridgesampling/blob/master/R/bridge_sampler_normal.
# Copied from https://gist.github.com/junpenglao/4d2669d69ddfe1d788318264cdcf0583
def Marginal_llk(mtrace, model=None, logp=None, maxiter=1000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.

    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    logp : Model Log-probability function, read from the model by default
    maxiter : Maximum number of iterations

    Returns
    -------
    marg_llk : Estimated Marginal log-Likelihood.
    """
    # Bridge Sampling
    r0, tol1, tol2 = 0.5, 1e-10, 1e-4
    
    model = modelcontext(model)
    logp = model.logp_array
    # free_RVs might be autotransformed. 
    # if that happens, there will be a model.deterministics entry with the first part of the name that needs to be used 
    # instead of the autotransformed name below in stats.ess
    # so we need to replace that variable with the corresponding one from the deterministics
    vars = model.free_RVs
    det_names=[d.name for d in model.deterministics]
    det_names.sort(key=lambda s:-len(s)) # sort descending by length
    
    def recover_var_name(name_autotransformed):
        for dname in det_names:
            if dname==name_autotransformed[:len(dname)]:
                return dname
        return name_autotransformed
    
    # Split the samples into two parts  
    # Use the first 50% for fiting the proposal distribution and the second 50% 
    # in the iterative scheme.
    len_trace = len(mtrace)
    nchain = mtrace.nchains
    
    N1_ = len_trace // 2
    N1 = N1_*nchain
    N2 = len_trace*nchain - N1
    
    neff_list = dict() # effective sample size
    
    arraysz = model.bijection.ordering.size
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))
    # matrix with already transformed samples
    for var in vars:
        varmap = model.bijection.ordering.by_name[var.name]
        # for fitting the proposal
        x = mtrace[:N1_][var.name]
        samples_4_fit[varmap.slc, :] = x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int))).T
        # for the iterative scheme
        x2 = mtrace[N1_:][var.name]
        samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0], np.prod(x2.shape[1:], dtype=int))).T
        # effective sample size of samples_4_iter, scalar
        orig_name=recover_var_name(var.name)
        tmp = az.from_pymc3(trace=mtrace[N1_:], model=model)
        neff_list.update(az.ess(tmp, var_names=[orig_name]))
        #neff_list.update(az.ess(mtrace[N1_:], var_names=[orig_name]))
    
    # median effective sample size (scalar)
    neff = np.median(list(neff_list.values()))  # FIXME: Crashes here because of shape sigma > 1!
    
    
    
    # %%
    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = chol(V, lower=True)
    
    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + np.dot(L, st.norm.rvs(0, 1, size=samples_4_iter.shape))
    
    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)
    
    # Evaluate unnormalized posterior for posterior & generated samples
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])
    
    # Iterative scheme as proposed in Meng and Wong (1996) to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar = np.median(l1) # To increase numerical stability, 
                                # subtracting the median of l1 from l1 & l2 later
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)
        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol
    
        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn("""Infinite value in iterative scheme, returning NaN. 
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion=='r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion=='logml':
                criterion_val = np.abs((logml - logmlold)/logml)
    
        if i >= maxiter:
            return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)
    
    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with 
                        adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')
    
    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal", q11 = q11, q12 = q12, q21 = q21, q22 = q22)
