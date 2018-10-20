#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
### ============================================================ ###
# MODEL: restricted qubits with projective measurement
# |phi(x)> = sin(x) |1> + cos(x) |0>
### ============================================================ ###
def proba(x, noise=0):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    return np.square(np.sin(x_noise))

def measure(x, underlying_p = proba, nb_measures = 1):
    return rdm.binomial(nb_measures, underlying_p(x))/nb_measures

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)


### ============================================================ ###
# Data training / test
### ============================================================ ###
x_range = (0, np.pi)
nb_init = 50
x_test = np.linspace(*x_range)[:, np.newaxis]
p_test = proba(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = measure(x_init)

plt.plot(x_test, p_test, label = 'real proba')
plt.scatter(x_init, y_init, label='one shot')
plt.legend()


### ============================================================ ###
# Regression task
# 1- Simple GP with heteroskedastic noise - onem
# 2- Use of a bernouilli Likelihood
### ============================================================ ###
#Same kernel for all
k_one = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)

# ------------------------------------------ #
# 1 Classic set-up
# ------------------------------------------ #
m_reg_one = GPy.models.GPRegression(X = x_init, Y=y_init, kernel=k_one)
m_reg_one.optimize_restarts(num_restarts = 10)
yp_reg_one, _ = m_reg_one.predict(x_test)
error_reg_one = squerror(yp_reg_one, p_test)
print(error_reg_one)
m_reg_one.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])
plt.legend()



# ------------------------------------------ #
# 2 Bernouilli likelihood
# ------------------------------------------ #
i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Bernoulli()
m_classi = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik)
_ = m_classi.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters

y1, v1 = m_classi.predict(x_test) ## After link
y2, v2 = m_classi._raw_predict(x_test) ## Before link
y3, v3 = m_classi.predict_noiseless(x_test) ## same

y = y1
v = v1

plt.plot(x_test, y)
plt.plot(x_test, y+1.96*v)
plt.plot(x_test, y-1.96*v)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])





