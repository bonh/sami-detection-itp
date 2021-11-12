# -*- coding: utf-8 -*-
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

# +
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pymc3 as pm
import arviz as az

import dataprep
import bayesian

import scipy.special as sc

from sympy import *
init_printing()

# +
A, sigma, x, c, a = symbols('A sigma x c a')

f = A*exp(-(x-c)**2/2/sigma**2)*(1+erf(a*(x-c)/sqrt(2)/sigma))
f
# -

delta = a/sqrt(1+a**2)
muz = sqrt(2/pi)*delta
sigmaz = sqrt(1-muz**2)
gamma1 = (4-pi)/2*(delta*sqrt(2/pi)**3)/(1-2*delta**2/pi)**(3/2)
mo = muz - gamma1*sigmaz/2-sign(a)/2*exp(-2*pi/Abs(a))
xmax = c + sigma*mo

fmax = f.subs(x, xmax)
fmax

# +
x_ = np.linspace(0, 512, 2*512)
c_ = x_[int(len(x_)/2)]
a_ = 10
sigma_ = 10
A_ = 5

lam_f = lambdify([A, c, sigma, a, x], f, modules=['numpy', "scipy"])
plt.plot(x_, lam_f(A_, c_, sigma_, a_, x_));

xmax_ = xmax.subs({A:A_, c:c_, sigma:sigma_, a:a_})
plt.vlines(float(xmax_), 0, 12, "r")

fmax_ = fmax.subs({A:A_, c:c_, sigma:sigma_, a:a_})
plt.hlines(float(fmax_), 0, 512, "r");

plt.xlim(200, 400);
# -

str(fmax)


