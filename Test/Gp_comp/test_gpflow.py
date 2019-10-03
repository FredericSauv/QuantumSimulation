#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import gpflow
from matplotlib.ticker import MultipleLocator
import toyproba as toy



### ============================================================ ###
# Init
### ============================================================ ###
col_custom = (0.1, 0.2, 0.5)
seed = np.random.randint(1,1000000000)
seed=732972113
np.random.seed(seed)

version = 'v5_gpflow'+str(seed)
save=True
nb_init = 30
weight = 4

x_range = (0, 4)
x_test = np.linspace(*x_range, 1000)[:, np.newaxis]
p_test = toy.proba(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = toy.measure(x_init, underlying_p = toy.proba)

guess_v = 1. 
guess_ls = (x_range[1]-x_range[0])/25


### ============================================================ ###
# Start Optim
# Add mini batch
### ============================================================ ###
m = gpflow.models.SVGP(x_init, y_init, kern=gpflow.kernels.Matern52(input_dim = 1, 
                            lengthscales=guess_ls, variance = guess_v, ARD=False), 
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=x_init)
optimizer = gpflow.train.ScipyOptimizer()
#optimizer = gpflow.training.AdamOptimizer()
#optimizer_tensor = optimizer.make_optimize_tensor(m)
#session = gpflow.get_default_session()
#for _ in range(2):
#session.run(optimizer_tensor)
m.feature.set_trainable(False)
optimizer.minimize(m, maxiter=200, initialize=True) # 
# Unfix the hyperparameters.x_init = torch.linspace(0, 1, 10)
#opt = gpflow.train.AdamOptimizer()
#m.feature.set_trainable(True)
#opt.minimize(gpf_m)

toy.plotl_gpflow(m,(x_test, p_test), sd=0.2)

    
#toy.train_gpflow(m, optim, train_loc=False, train_kernel=True)
#t_m, t_v = m.predict_y(x_test)
#toy.plotl_gpflow(m,(x_test, p_test), sd=0.2)


## first step
x_next = np.atleast_1d(toy.next_ucb_std_gpflow(m, x_test, weight))
y_next = toy.measure(x_next, underlying_p = toy.proba)
toy.add_data_gpflow(m, x_next, y_next, fixZ=True)
#toy.train_gpflow(m, optimizer, train_loc=False, train_kernel=True)
optimizer.minimize(m, maxiter=200, initialize=True)
toy.plotl_gpflow(m,(x_test, p_test), sd=0.2)

#60 more
n_iter = 10
for n in range(n_iter):
    x_next = np.atleast_1d(toy.next_ucb_std_gpflow(m, x_test, weight))
    print(x_next)
    y_next = toy.measure(x_next, underlying_p = toy.proba)
    toy.add_data_gpflow(m, x_next, y_next, fixZ=True)
    optimizer.minimize(m, maxiter=200, initialize=True)
    #toy.train_gpflow(m, optim, train_loc=False, train_kernel=True)
    
toy.plotl_gpflow(m,(x_test, p_test), sd=0.2)

test_x = m.X.value
test_y = m.Y.value
m2 = gpflow.models.SVGP(test_x, test_y, kern=gpflow.kernels.Matern52(input_dim = 1, 
                            lengthscales=guess_ls, variance = guess_v, ARD=False), 
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=test_x, minibatch_size=50)
optim = gpflow.train.ScipyOptimizer()
toy.train_gpflow(m2, optim, train_loc=True, train_kernel=True)
toy.plotl_gpflow(m2,(x_test, p_test), sd=0.2)







pres = proba(x_best)
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
acq = (a + 0 * s)
ax2.plot(x_test, acq,color='r')
x_next = next_ucb_std_v2(m_classi, x_test, 0)
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
x_best = next_ucb_std_v2(m_classi, m_classi.X, 0)
ax1.vlines(x_best, 0,1, colors='g', label = r'$\theta_{best}$')
ax1.scatter(m_classi.X[-1], m_classi.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
ax1.text(-0.4, 1.1, '(c)', fontsize=18)
#ax2.set_ylabel(r'$ a.u.$', fontsize=16)

pres = proba2(x_best)
print(m_classi)
if (save):
    plt.savefig(version+"init100_nothet"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

ax2.text(1., -0.8, r'Control parameter $\theta$', fontsize=18)
if (save):
    plt.savefig(version+"init100"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

############## FULL
pres = proba2(x_best)
if (save):
    plt.savefig(version+"init100"+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)

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


