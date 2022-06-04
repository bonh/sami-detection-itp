# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


# +
def skewedgaussian(amp, cent, sig, alpha, x):        
    return gauss(amp, cent, sig, x) * errorfunction(cent, sig, alpha, x)

def errorfunction(cent, sig, alpha, x):        
    return (1-special.erf((alpha*(cent - x))/np.sqrt(2)/sig))

def gauss(amp, cent, sig, x):
    return amp*np.exp(-(cent - x)**2/2/sig**2)


# +
x = np.linspace(0,200,1000)
alpha = -4
f = skewedgaussian(1, 70, 5, alpha, x)
g = skewedgaussian(1, 30, 5, alpha, x)

alpha = 10
h = skewedgaussian(1, 70, 5, alpha, x)
i = skewedgaussian(1, 30, 5, alpha, x)
# -

plt.plot(f);
plt.plot(g);
plt.plot(h)
plt.plot(i);

plt.plot(np.correlate(f, g, "same"), label="first")
plt.plot(np.correlate(h, i, "same"), label="second")
plt.legend()
plt.xlim(500,900);

# +
x = np.linspace(0,200,1000)
alpha = -10
f = errorfunction(70, 5, alpha, x)
g = errorfunction(30, 5, alpha, x)

alpha = 5
h = errorfunction(70, 5, alpha, x)
i = errorfunction(30, 5, alpha, x)
# -

plt.plot(f);
plt.plot(g);
plt.plot(h)
plt.plot(i);

plt.plot(np.correlate(f, g, "same"), label="first")
plt.plot(np.correlate(h, i, "same"), label="second")
plt.legend()


