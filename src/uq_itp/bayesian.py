import numpy as np
import pymc3 as pm

def model_signal(amp, cent, sig, baseline, x):
    return amp*np.exp(-1*(cent - x)**2/2/sig**2) + baseline

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


