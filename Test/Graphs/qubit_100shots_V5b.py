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
from matplotlib.ticker import MultipleLocator

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
seed = np.random.randint(1,1000000000)
seed=732972113
np.random.seed(seed)

version = 'v2_'+str(seed)
save=True
weight=6
col_custom = (0.1, 0.2, 0.5)
nb_init = 30
weight = 4

x_range = (0, 4)
x_test = np.linspace(*x_range, 1000)[:, np.newaxis]
p_test = proba2(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = measure(x_init, underlying_p = proba2)

k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/25)
i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Binomial()
m_classi = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
#m_classi['.*lengthscale'].constrain_bounded(0.10, 1., warning = False)
_ = m_classi.optimize_restarts(num_restarts=5) 


a, b, c, d, d_range, s = predict_p(m_classi, x_test)
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
ax2.vlines(x_next, np.min(acq), np.max(acq), colors='r', linestyles='dashed')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax2.set_xlim([-0.02, 4.02])
ax2.xaxis.tick_top()
ax1.set_xticklabels([], visible= False)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.set_yticks(np.arange(0,1.01, 0.2))
ax1.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_xticks(np.arange(0,4.01, 0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.set_yticklabels([], visible= False)

#ax1.set_xticklabels(, fontsize=16)
#ax1.set_yticklabels(fontsize=16)
#plt.legend(loc='lower right', fontsize=12)
#acq_scale = (np.max(acq) - np.min(acq))
#acq_shift = np.max(acq)
#plt.plot(x_test, (acq - acq_shift) *0.3 / acq_scale , color='r')
#ax2.scatter(x_next, 0, marker = '*', c='r', s=80)
#ax2.set_yticklabels('a', visible=False)
#plt.ylabel(r'$p_{tgt}(\theta)$', fontsize=16)
#plt.xlabel(r'$\theta$', fontsize=16)
#plt.text(-0.5, -0.3,r'$(a)$')
#plt.xticks(fontsize=16)
#plt.yticks(np.arange(0, 1.01, step=0.2), fontsize=16)
print(m_classi)


if (save):
    plt.savefig(version+"init30.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



n_iter = 1
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, weight))
    y_next = measure(x_next, underlying_p = proba2)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=5)


a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
ax2.vlines(x_next, np.min(acq), np.max(acq), colors='r', linestyles='dashed')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax2.set_xlim([-0.02, 4.02])
ax2.xaxis.tick_top()
ax1.set_xticklabels([], visible= False)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.set_yticks(np.arange(0,1.01, 0.2))
ax1.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_yticklabels([], visible= False)

#ax1.set_xticklabels(, fontsize=16)

#x_best = next_ucb_std(m_classi, m_classi.X, 0)
#plt.vlines(x_best, -1,2, colors='g')
print(m_classi)
if (save):
    plt.savefig(version+"init31.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


n_iter = 60
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
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(m_classi.X, m_classi.Y, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
ax2.vlines(x_next, np.min(acq), np.max(acq), colors='r', linestyles='dashed')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax2.set_xlim([-0.02, 4.02])
ax2.xaxis.tick_top()
ax1.set_xticklabels([], visible= False)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.set_yticks(np.arange(0,1.01, 0.2))
ax1.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_yticklabels([], visible= False)
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')

pres = proba2(x_best)
if (save):
    plt.savefig(version+"init91"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)


n_iter = 9
for n in range(n_iter):
    x_next = np.atleast_1d(next_ucb_std_v2(m_classi, x_test, 0))
    y_next = measure(x_next, underlying_p = proba2)
    print(x_next)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = predict_p_v2(m_classi, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(m_classi.X, m_classi.Y, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = next_ucb_std_v2(m_classi, x_test, weight)
ax2.vlines(x_next, np.min(acq), np.max(acq), colors='r', linestyles='dashed')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax2.set_xlim([-0.02, 4.02])
ax2.xaxis.tick_top()
ax1.set_xticklabels([], visible= False)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.set_yticks(np.arange(0,1.01, 0.2))
ax1.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_yticklabels([], visible= False)
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')

pres = proba2(x_best)
if (save):
    plt.savefig(version+"init100"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

