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

def proba2(x, noise=0):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
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
nb_init = 1000
x_test = np.linspace(*x_range)[:, np.newaxis]
p_test = proba2(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = measure(x_init, underlying_p = proba2)

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
lik = GPy.likelihoods.Binomial()
m_classi = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
_ = m_classi.optimize_restarts(num_restarts=1) #first runs EP and then optimizes the kernel parameters

y1, v1 = m_classi.predict(x_test, Y_metadata = {'trials':1}) ## After link
y2, v2 = m_classi._raw_predict(x_test) ## Before link
y3, v3 = m_classi.predict_noiseless(x_test) ## same

y = y2
v = v2

plt.plot(x_test, y)
plt.plot(x_test, y+1.96*v)
plt.plot(x_test, y-1.96*v)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])




def predict_p(model, X):
    mean, var = model._raw_predict(X, full_cov=False, kern=None)
    likelihood = model.likelihood
    Nf_samp = 10000
    s = np.random.randn(mean.shape[0], Nf_samp) * np.sqrt(var) + mean
    #ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
    p = likelihood.gp_link.transf(s)
    mean = np.median(p, axis = 1)
    q = np.quantile(p, [0.025, 0.975], axis=1)
    y = np.linspace(1,0)
    density = np.diff(np.array([[np.sum(pp < yy) for pp in p] for yy in y]), axis=0)
    return mean, q[0], q[1], density
    

col_custom = (0.1, 0.2, 0.5)

a, b, c, d = predict_p(m_classi, x_test)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = 'model')
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(-d, cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (0,  np.pi, 0, 1), alpha=0.6)
plt.scatter(x_init, y_init, label = 'Observations', marker = 'o', c='red', s=5)
plt.plot(x_test, p_test, 'r--', label='F')
plt.legend(loc='lower right')
plt.savefig("bernouilli_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

