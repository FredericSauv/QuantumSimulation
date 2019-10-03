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
from matplotlib.ticker import MultipleLocator
import toyproba as toy



### ============================================================ ###
# Init
### ============================================================ ###
col_custom = (0.1, 0.2, 0.5)
seed = np.random.randint(1,1000000000)
seed=732972113
np.random.seed(seed)

version = 'v5_gpy'+str(seed)
save=True
nb_init = 30
weight = 4

x_range = (0, 4)
x_test = np.linspace(*x_range, 1000)[:, np.newaxis]
p_test = toy.proba(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = toy.measure(x_init, underlying_p = toy.proba)


### ============================================================ ###
# Start Optim
### ============================================================ ###
k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/25)
i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Binomial()
m_classi = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
#m_classi['.*lengthscale'].constrain_bounded(0.10, 1., warning = False)
_ = m_classi.optimize_restarts(num_restarts=5) 


a, b, c, d, d_range, s = toy.predict_p(m_classi, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
#fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, sharey=True,grid        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groupsspec_kw={'hspace': 0.3,'wspace':0.05,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = toy.next_ucb_std(m_classi, x_test, weight)
ax2.vlines(x_next, 0., np.max(acq), colors='r', linestyles='dashed')
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
#ax2.set_yticklabels([], visible= True)
ax2.set_yticks([])
ax2.set_yticks([0., 2.])
ax1.text(-0.4, 1.1, '(a)', fontsize=18)
ax2.set_ylabel(r'$a.u.$', fontsize=16)
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
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
ax2.set_yticks([0.0, 2.0])
ax1.set_ylabel('Landscape', fontsize=18)
ax2.set_ylabel('Acqu.', fontsize=18)
ax2.yaxis.set_label_coords(-0.09,0.5)


if (save):
    plt.savefig(version+"init30_nothet.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

ax2.text(1., -0.8, r'Control parameter $\theta$', fontsize=18)
if (save):
    plt.savefig(version+"init30.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


n_iter = 1
for n in range(n_iter):
    x_next = np.atleast_1d(toy.next_ucb_std(m_classi, x_test, weight))
    y_next = toy.measure(x_next, underlying_p = toy.proba)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=5)


a, b, c, d, d_range, s = toy.predict_p(m_classi, x_test)
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
x_next = toy.next_ucb_std(m_classi, x_test, weight)
ax2.vlines(x_next, 0, np.max(acq), colors='r', linestyles='dashed')
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
#ax2.set_yticklabels([], visible= False)
ax2.set_yticks([0.0, 2.0])
#ax2.set_yticks([])
ax1.text(-0.4, 1.1, '(b)', fontsize=18)
#ax2.set_ylabel(r'$ a.u.$', fontsize=16)
#ax1.set_xticklabels(, fontsize=16)

#x_best = next_ucb_std(m_classi, m_classi.X, 0)
#plt.vlines(x_best, -1,2, colors='g')
print(m_classi)
if (save):
    plt.savefig(version+"init31_nothet.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

ax2.text(1., -0.8, r'Control parameter $\theta$', fontsize=18)
if (save):
    plt.savefig(version+"init31.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

n_iter = 60
for n in range(n_iter):
    x_next = np.atleast_1d(toy.next_ucb_std(m_classi, x_test, weight))
    y_next = toy.measure(x_next, underlying_p = toy.proba)
    print(x_next)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = toy.predict_p(m_classi, x_test)
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
x_next = toy.next_ucb_std(m_classi, x_test, weight)
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
x_best = toy.next_ucb_std(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
ax2.set_yticks([])
ax1.text(-0.4, 1.1, '(c)', fontsize=18)
#ax2.set_ylabel(r'$ a.u.$', fontsize=16)

pres = toy.proba(x_best)
if (save):
    plt.savefig(version+"init91"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)


n_iter = 9
for n in range(n_iter):
    x_next = np.atleast_1d(toy.next_ucb_std(m_classi, x_test, 0))
    y_next = toy.measure(x_next, underlying_p = toy.proba)
    print(x_next)
    X_new = np.vstack([m_classi.X, x_next])
    Y_new = np.vstack([m_classi.Y, y_next])
    m_classi.Y_metadata = {'trials':np.ones_like(Y_new)*1}
    m_classi.set_XY(X_new, Y_new)    
    _ = m_classi.optimize_restarts(num_restarts=1)
    
a, b, c, d, d_range, s = toy.predict_p(m_classi, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(m_classi.X, m_classi.Y, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
acq = (a + 0 * s)
ax2.plot(x_test, acq,color='r')
x_next = toy.next_ucb_std(m_classi, x_test, 0)
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
#ax2.set_yticklabels([], visible= False)
ax2.set_yticks([0.0, 2.0])
x_best = toy.next_ucb_std(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
ax1.text(-0.4, 1.1, '(c)', fontsize=18)
#ax2.set_ylabel(r'$ a.u.$', fontsize=16)

pres = toy.proba(x_best)
print(m_classi)
if (save):
    plt.savefig(version+"init100_nothet"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

ax2.text(1., -0.8, r'Control parameter $\theta$', fontsize=18)
if (save):
    plt.savefig(version+"init100"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

############## FULL
pres = toy.proba(x_best)
if (save):
    plt.savefig(version+"init100"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

a, b, c, d, d_range, s = toy.predict_p(m_classi, x_test)
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
x_next = toy.next_ucb_std(m_classi, x_test, weight)
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
x_best = toy.next_ucb_std_v2(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')


