#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:25:36 2019

@author: fred
"""
import numpy as np
from numpy import random as rdm
import matplotlib.pyplot as plt
import GPy

def proba(x, noise=0):
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
    return np.square(np.sin(x_noise))

def measure(x, underlying_p = proba, nb_measures = 1, noise_gauss = None):
    proba = underlying_p(x)
    if nb_measures == np.inf:
        res = proba 
        if noise_gauss is not None:
            res += np.random.normal(loc=0., scale=noise_gauss, size=proba.shape)
    else:
        res = rdm.binomial(nb_measures, proba)/nb_measures
    return res

landscape = lambda x: proba(x, 0)
observe = lambda x: measure(x, landscape, np.inf, 0.1)

nb_init = 30
x_range = (0, 4)
x_test = np.linspace(*x_range, 1000)[:, np.newaxis]
p_test = landscape(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = observe(x_init)
plt.plot(x_test, p_test, 'b--')
plt.scatter(x_init, y_init)

ls_init = (x_range[1]-x_range[0])/25
v_init = 1.



### ============================================================ ###
# GPY model
### ============================================================ ###
k_classi = GPy.kern.Matern52(input_dim = 1, variance = v_init, lengthscale = ls_init)
#i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Gaussian()
m_gpy = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, likelihood=lik)
_ = m_gpy.optimize_restarts(num_restarts=1) 


mean, var = m_gpy.predict(x_test, likelihood=None, include_likelihood=False)
col_custom = (0.1, 0.2, 0.5)
plt.plot(x_test, mean)
plt.plot(x_test, p_test)
plt.plot(x_test, mean, color = col_custom, linewidth = 0.8, label = r'$model$')
plt.plot(x_test, mean - 1.96 * np.sqrt(var), color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, mean + 1.96 * np.sqrt(var), color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=25)
plt.plot(x_test, p_test, 'r--', label='F')
plt.legend()
