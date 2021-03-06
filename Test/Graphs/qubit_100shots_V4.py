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
import scipy.stats as stats
import scipy.special as spe


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

def proba2(x, noise=0, scaling = 1.):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        #x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
        x_noise = +np.sin(3*(x))/2 + (x)/1.5
    return np.square(scaling * np.sin(x_noise))


def measure(x, underlying_p = proba, nb_measures = 1):
    return rdm.binomial(nb_measures, underlying_p(x))/nb_measures

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)


def predict_p(model, X):
    mean, var = model._raw_predict(X, full_cov=False, kern=None)
    likelihood = model.likelihood
    Nf_samp = 10000
    s = np.random.randn(mean.shape[0], Nf_samp) * np.sqrt(var) + mean
    #ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
    p = likelihood.gp_link.transf(s)
    p_min = np.min(p)
    p_max = np.max(p)
    mean = np.median(p, axis = 1)
    std = np.std(p, axis = 1)
    q = np.quantile(p, [0.025, 0.975], axis=1)
    y = np.linspace(0, 1)
    density = np.diff(np.vstack([np.array([[np.sum(pp < yy) for pp in p] for yy in y]), np.ones(len(p))*Nf_samp]), axis=0)/Nf_samp
    
    return mean, q[0], q[1], density, (p_min, p_max), std

def invcdf(y, alpha=1):
    return  np.sqrt(2) * spe.erfinv(2*y-1) / alpha 

def changedistrib(y, mu, var, alpha=1):
    inv = invcdf(y,alpha)
    res = stats.norm.pdf(inv, loc=mu, scale=var) / (alpha * stats.norm.pdf( alpha * inv))
    res[np.isnan(res)]=0
    return  res
    
def predict_p_v2(model, X, alpha=1):
    mean, var = model._raw_predict(X, full_cov=False, kern=None)
    #likelihood = model.likelihood
    #Nf_samp = 10000
    #s = np.random.randn(mean.shape[0], Nf_samp) * np.sqrt(var) + mean
    #ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
    #p = likelihood.gp_link.transf(s)
    N_discr = 5000
    y = np.linspace(0, 1,N_discr)
    density = np.array([changedistrib(y, m, np.sqrt(v), alpha) for m, v in zip(mean, var)])
    p_min, p_max = 0, 1
    m_mean = np.array([np.sum(d * y)/len(y) for d in density])
    densitycum = np.cumsum(density, axis=1)/N_discr
    m_median = np.array([y[np.argwhere(dc>=0.5)[0,0]] for dc in densitycum])
    q = np.array([(y[np.argwhere(dc>=0.025)[0,0]], y[np.argwhere(dc<=0.975)[-1,0]]) for dc in densitycum])
    std = np.array([np.sqrt(np.sum(np.square(y-m) * d)/N_discr) for d, m in zip(density, m_mean)])
    return m_median, q[:,0], q[:,1], np.transpose(density), (p_min, p_max), std


def next_ucb_bounds(model, loc, w=4):
    a, b, c, d, d_range, s = predict_p(model, loc)
    acq = (a + w * (c-b))
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_std(model, loc, w=4):
    a, b, c, d, d_range, s = predict_p(model, loc)
    acq = (a + w * s)
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_bounds_v2(model, loc, w=4, alpha=1):
    a, b, c, d, d_range, s = predict_p_v2(model, loc, alpha)
    acq = (a + w * (c-b))
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_std_v2(model, loc, w=4, alpha=1):
    a, b, c, d, d_range, s = predict_p_v2(model, loc, alpha)
    acq = (a + w * s)
    x_next = loc[np.argmax(acq)]
    return x_next



### ============================================================ ###
# Data training / test
### ============================================================ ###

version = 'v6_'
save=False
weight=6
col_custom = (0.1, 0.2, 0.5)
nb_init = 30
weight = 4
scale = 1


x_range = (0, 4)
x_test = np.linspace(*x_range, 2000)[:, np.newaxis]
p_test = proba2(x_test, scaling=scale)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
#x_init = rdm.uniform(*[1.65, 2.15], nb_init)[:, np.newaxis]
#x_init = np.row_stack((x_init, rdm.uniform(*[1.70, 2.1], 6*nb_init)[:, np.newaxis]))
y_init = measure(x_init, underlying_p = lambda x: proba2(x, scaling=scale))


alpha=1.
k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1.* alpha, lengthscale = (x_range[1]-x_range[0])/25)
i_meth = GPy.inference.latent_function_inference.Laplace()
l_func = GPy.likelihoods.link_functions.Probit2(alpha)
lik = GPy.likelihoods.Binomial(gp_link=l_func)
m_classi = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
#m_classi['.*lengthscale'].constrain_bounded(0.10, 1., warning = False)
_ = m_classi.optimize_restarts(num_restarts=5) 


a, b, c, d, d_range, s = predict_p(m_classi, x_test)
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test, alpha)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1.25), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
plt.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=5)
plt.plot(x_test, p_test, 'r--', label='F')
plt.legend(loc='lower right', fontsize=12)
acq = (a + weight * s)
acq_scale = 10*(np.max(acq) - np.min(acq))
acq_shift = np.mean(acq)
plt.plot(x_test, (acq - acq_shift) / acq_scale - 0.1, color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight, alpha)
plt.vlines(x_next, -1,2, colors='r')
plt.ylim([-0.2, 1.099])
plt.ylabel(r'$p_{tgt}(\theta)$', fontsize=16)
plt.xlabel(r'$\theta$', fontsize=16)
#plt.text(-0.5, -0.3,r'$(a)$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
print(m_classi)


if (save):
    plt.savefig(version+"init25.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



n_iter = 1
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, weight))
    y_next = measure(x_next, underlying_p = proba2)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8)
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 0.75), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
plt.scatter(m_classi.X, m_classi.Y, marker = 'o', c='red', s=5)
plt.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = '*', c='g', s=30)
plt.plot(x_test, p_test, 'r--')
plt.legend(loc='lower right')
acq = (a + weight * s)
acq_scale = 10*(np.max(acq) - np.min(acq))
acq_shift = np.mean(acq)
plt.plot(x_test, (acq - acq_shift) / acq_scale - 0.1, color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
plt.vlines(x_next, -1,2, colors='r')
plt.ylim([-0.2, 1.099])
#x_best = next_ucb_std(m_classi, m_classi.X, 0)
#plt.vlines(x_best, -1,2, colors='g')
print(m_classi)
if (save):
    plt.savefig(version+"init26.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



n_iter = 75
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, weight))
    y_next = measure(x_next, underlying_p = proba2)
    print(x_next)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8)
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 0.75), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
plt.scatter(m_classi.X, m_classi.Y, marker = 'o', c='red', s=5)
plt.plot(x_test, p_test, 'r--')
#plt.legend(loc='lower right')
acq = (a + weight * s)
acq_scale = 10*(np.max(acq) - np.min(acq))
acq_shift = np.mean(acq)
plt.plot(x_test, (acq - acq_shift) / acq_scale - 0.1, color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
plt.vlines(x_next, -1,2, colors='r')
plt.ylim([-0.2, 1.099])
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
plt.vlines(x_best, -1,2, colors='g', label = r'$\theta_{best}$')
plt.legend(loc='lower right')
print(m_classi)
if (save):
    plt.savefig(version+"init106.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
x_100=proba2(x_best)
print(x_100)

n_iter = 75
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, weight))
    print(x_next)
    y_next = measure(x_next, underlying_p = proba2)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8)
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 0.75), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
plt.scatter(m_classi.X, m_classi.Y, marker = 'o', c='red', s=5)
plt.plot(x_test, p_test, 'r--')
#plt.legend(loc='lower right')
acq = (a + weight * s)
acq_scale = 10*(np.max(acq) - np.min(acq))
acq_shift = np.mean(acq)
plt.plot(x_test, (acq - acq_shift) / acq_scale - 0.1, color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
plt.vlines(x_next, -1,2, colors='r')
plt.ylim([-0.2, 1.099])
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
plt.vlines(x_best, -1,2, colors='g', label = r'$\theta_{best}$')
plt.legend(loc='lower right')
print(m_classi)
if (save):
    plt.savefig(version+"init175.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

x_175=proba2(x_best)


n_iter = 25
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, 0))
    print(x_next)
    y_next = measure(x_next, underlying_p = proba2)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
plt.figure()
plt.plot(x_test, a, color = col_custom, linewidth = 0.8)
plt.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
plt.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 0.75), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
plt.scatter(m_classi.X, m_classi.Y, marker = 'o', c='red', s=5)
plt.plot(x_test, p_test, 'r--')
#plt.legend(loc='lower right')
acq = (a + weight * s)
acq_scale = 10*(np.max(acq) - np.min(acq))
acq_shift = np.mean(acq)
plt.plot(x_test, (acq - acq_shift) / acq_scale - 0.1, color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
plt.ylim([-0.2, 1.099])
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
plt.vlines(x_best, -1,2, colors='g', label = r'$\theta_{best}$')
plt.legend(loc='lower right')
print(m_classi)
if (save):
    plt.savefig(version+"init200.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

x_200=proba2(x_best)