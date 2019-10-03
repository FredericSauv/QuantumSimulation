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
save=False
nb_init = 100
nb_extra = 100
weight = 4

dico_range = {'all':(0, 4), 'tenpc':(1.73, 2.04), 'onepc':(1.83, 1.92), 'tenthpc':(1.86, 1.89), 'hdredthpc':(1.87, 1.88)}
range_all = (0,4)
type_range = 'hdredthpc'
range_test = dico_range[type_range]
x_test_all = np.linspace(*range_all, 5000)[:, np.newaxis]
p_test_all = toy.proba_1(x_test_all)
x_test = np.linspace(*range_test, 5000)[:, np.newaxis]

n_repeat = 10
res = np.zeros((n_repeat,6))

for i in range(n_repeat):    
    x_init = rdm.uniform(*range_all, nb_init)[:, np.newaxis]
    x_init = np.vstack((x_init, rdm.uniform(*range_test, nb_extra)[:, np.newaxis]))
    y_init = toy.measure(x_init, underlying_p = toy.proba_1)
    
    ### ============================================================ ###
    # Start Optim
    ### ============================================================ ###
    #Matern52
    k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.Laplace()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_LP = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_LP.optimize_restarts(num_restarts=5) 
    
    k_classi_EP = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.EP()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_EP = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_EP.optimize_restarts(num_restarts=5) 
    
    
    k_classi = GPy.kern.Matern32(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.Laplace()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_LP_32 = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_LP_32.optimize_restarts(num_restarts=5) 
    
    
    k_classi_EP = GPy.kern.Matern32(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.EP()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_EP_32 = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_EP_32.optimize_restarts(num_restarts=5) 
    
    
    k_classi = GPy.kern.Exponential(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.Laplace()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_LP_Exp = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_LP_Exp.optimize_restarts(num_restarts=5) 
    
    
    k_classi_EP = GPy.kern.Exponential(input_dim = 1, variance = 1., lengthscale = (range_all[1]-range_all[0])/25)
    i_meth = GPy.inference.latent_function_inference.EP()
    lik = GPy.likelihoods.Bernoulli()
    m_classi_EP_Exp = GPy.core.GP(X=x_init, Y=y_init, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
    _ = m_classi_EP_Exp.optimize_restarts(num_restarts=5) 
    
    
    x_best_EP = toy.next_ucb_std(m_classi_EP, x_test, 0)
    pres_EP = toy.proba_1(x_best_EP)
    x_best_LP = toy.next_ucb_std(m_classi_LP, x_test, 0)
    pres_LP = toy.proba_1(x_best_LP)
    print('res Mat52 (EP then LP)', pres_EP, pres_LP)
    
    x_best_EP_32 = toy.next_ucb_std(m_classi_EP_32, x_test, 0)
    pres_EP_32 = toy.proba_1(x_best_EP_32)
    x_best_LP_32 = toy.next_ucb_std(m_classi_LP_32, x_test, 0)
    pres_LP_32 = toy.proba_1(x_best_LP_32)
    print('res Mat32 (EP then LP)', pres_EP_32, pres_LP_32)
    
    
    x_best_EP_Exp = toy.next_ucb_std(m_classi_EP_Exp, x_test, 0)
    pres_EP_Exp = toy.proba_1(x_best_EP_Exp)
    x_best_LP_Exp = toy.next_ucb_std(m_classi_LP_Exp, x_test, 0)
    pres_LP_Exp = toy.proba_1(x_best_LP_Exp)
    print('res Exp (EP then LP)', pres_EP_Exp, pres_LP_Exp)
    res[i] = np.array([pres_EP, pres_LP, pres_EP_32, pres_LP_32, pres_EP_Exp, pres_LP_Exp]).reshape(-1)



#10pct
#        52_EP       52_LP         32_EP      32_LP       EXP_EP     EXP_LP
#array([0.9899005 , 0.98977848, 0.99025309, 0.99058795, 0.98461679, 0.99111406])
#array([0.99407037, 0.99317094, 0.99547118, 0.99432037, 0.99481707, 0.9954619 ])
#array([0.01188907, 0.01041588, 0.01318738, 0.01104681, 0.01816396, 0.00928515])

#1pct
# array([0.99900814, xx0.99961831, x0.99941858, xxx 0.99978502, 0.99883937,0.99886459])
# array([0.99955108, x 0.99985032, xx 0.99987103, xxx 0.99992634, 0.99968516, 0.99936158])
# array([0.00136344, 0.00040108, 0.00098134, 0.00032588, 0.00165528,0.00103888])

#0.1pct
#array([0.99947969, 0.99951994, 0.99952799, 0.99961506, 0.99994802,0.99993174])
#array([0.9995245 , 0.99957853, 0.99956807, 0.99980703, 0.99997259,0.99995659])
#array([4.06960678e-04, 4.03567891e-04, 4.03097321e-04, 3.94076752e-04,6.28209046e-05, 5.83262682e-05])

# 0.01pct
#array([0.99992767, 0.9999285 , 0.99992741, 0.99993659, 0.99999654,0.99999611])
#array([0.99990904, 0.99990904, 0.99990904, 0.99993705, 0.99999643,0.99999669])
#array([4.98003793e-05, 5.54923105e-05, 5.47198117e-05, 5.80203911e-05,2.86241719e-06, 3.57639584e-06])

res_all_median = np.array([[0.99407037, 0.99317094, 0.99547118, 0.99432037, 0.99481707, 0.9954619 ],
                           [0.99955108, 0.99985032, 0.99987103,  0.99992634, 0.99968516, 0.99936158],
                           [0.9995245 , 0.99957853, 0.99956807, 0.99980703, 0.99997259,0.99995659],
                           [0.99990904, 0.99990904, 0.99990904, 0.99993705, 0.99999643,0.99999669]
                           ])
res_all_mean = np.array([[0.9899005 , 0.98977848, 0.99025309, 0.99058795, 0.98461679, 0.99111406],
                         [0.99900814, 0.99961831, 0.99941858, 0.99978502, 0.99883937,0.99886459],
                         [0.99947969, 0.99951994, 0.99952799, 0.99961506, 0.99994802,0.99993174],
                         [0.99990904, 0.99990904, 0.99990904, 0.99993705, 0.99999643,0.99999669]
                         ])

    
fig, ax1 = plt.subplots(1, 1)
ax1.plot(1-res_all_median[:,0], label='52_EP',  marker='s')
ax1.plot(1-res_all_median[:,1], label='52_LP',  marker='s')
ax1.plot(1-res_all_median[:,2], label='32_EP',  marker='s')
ax1.plot(1-res_all_median[:,3], label='32_LP',  marker='s')
ax1.plot(1-res_all_median[:,4], label='EXP_EP',  marker='s')
ax1.plot(1-res_all_median[:,5], label='EXP_LP',  marker='s')
ax1.set_yscale('log')
ax1.legend()
    
## Comp both
fig, ax1 = plt.subplots(1, 1)
a, b, c, d, d_range, s = toy.predict_p(m_classi_LP, x_test)
ax1.plot(x_test, a, color = 'b', linewidth = 0.8, label = r'$LP$')
ax1.scatter(x_init, y_init, marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test_all, 'r--')
a, b, c, d, d_range, s = toy.predict_p(m_classi_EP, x_test)
ax1.plot(x_test, a,'g', linewidth = 0.8, label = r'$EP$')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax1.legend()
ax1.set_title('Mat52')


## Comp both
fig, ax1 = plt.subplots(1, 1)
a, b, c, d, d_range, s = toy.predict_p(m_classi_LP_32, x_test_all)
ax1.plot(x_test_all, a, color = 'b', linewidth = 0.8, label = r'$LP$')
ax1.scatter(x_init, y_init, marker = 'o', c='red', s=25)
ax1.plot(x_test_all, p_test_all, 'r--')
a, b, c, d, d_range, s = toy.predict_p(m_classi_EP_32, x_test_all)
ax1.plot(x_test_all, a,'g', linewidth = 0.8, label = r'$EP$')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax1.legend()
ax1.set_title('Mat32')


## Comp both
fig, ax1 = plt.subplots(1, 1)
a, b, c, d, d_range, s = toy.predict_p(m_classi_LP_Exp, x_test_all)
ax1.plot(x_test_all, a, color = 'b', linewidth = 0.8, label = r'$LP$')
ax1.scatter(x_init, y_init, marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test_all, 'r--')
a, b, c, d, d_range, s = toy.predict_p(m_classi_EP_Exp, x_test_all)
ax1.plot(x_test_all, a,'g', linewidth = 0.8, label = r'$EP$')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax1.legend()
ax1.set_title('Exp')




##################
# Reproduce example rasmussen
##################
train_size = 50
rng = np.random.RandomState(0)
X = rng.uniform(0, 4, 100)[:, np.newaxis]
y = np.array(X > 2, dtype=int)

k_classi = GPy.kern.Matern52(input_dim = 1, variance = 0.5, lengthscale = (x_range[1]-x_range[0])/10)
i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Bernoulli()
m_classi_LP = GPy.core.GP(X=X, Y=y, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
#m_classi['.*lengthscale'].constrain_bounded(0.10, 1., warning = False)
_ = m_classi_LP.optimize_restarts(num_restarts=5) 


k_classi_EP = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/25)
i_meth = GPy.inference.latent_function_inference.EP()
lik = GPy.likelihoods.Bernoulli()
m_classi_EP = GPy.core.GP(X=X, Y=y, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * 1})
#m_classi['.*lengthscale'].constrain_bounded(0.10, 1., warning = False)
_ = m_classi_EP.optimize_restarts(num_restarts=5) 


x_best_EP = toy.next_ucb_std(m_classi_EP, x_test, 0)
pres_EP = toy.proba_1(x_best_EP)
x_best_LP = toy.next_ucb_std(m_classi_LP, x_test, 0)
pres_LP = toy.proba_1(x_best_LP)
print(pres_EP, pres_LP)



## Comp both
fig, ax1 = plt.subplots(1, 1)
a, b, c, d, d_range, s = toy.predict_p(m_classi_LP, x_test)
ax1.plot(x_test, a, color = 'b', linewidth = 0.8, label = r'$model_LP$')
ax1.scatter(X, y, label = r'$Observations$', marker = 'o', c='red', s=25)
a, b, c, d, d_range, s = toy.predict_p(m_classi_EP, x_test)
ax1.plot(x_test, a,'g', linewidth = 0.8, label = r'$model$')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])










m = m_classi_LP
a, b, c, d, d_range, s = toy.predict_p(m, x_test)
fig, (ax1, ax2) = plt.subplots(2, 1,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
#fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, sharey=True,grid        params (iterable): iterable of parameters to optimize or dicts defining
ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
#plt.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$\bar{f}$')
ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', aspect='auto', interpolation='spline16', extent = (x_range[0],  x_range[1], d_range[0], d_range[1]), alpha=1)
ax1.scatter(x_init, y_init, label = r'$Observations$', marker = 'o', c='red', s=25)
ax1.plot(x_test, p_test, 'r--', label='F')
acq = (a + weight * s)
ax2.plot(x_test, acq,color='r')
x_next = toy.next_ucb_std(m, x_test, 0)
ax2.vlines(x_next, 0., np.max(acq), colors='r', linestyles='dashed')
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlim([-0.02, 4.02])
ax2.set_xlim([-0.02, 4.02])
#ax2.xaxis.tick_top()
#ax1.set_xticklabels([], visible= False)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.set_yticks(np.arange(0,1.01, 0.2))
ax1.set_xticks(np.arange(0,4.01, 0.5))
ax2.set_xticks(np.arange(0,4.01, 0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.set_yticks([])
ax2.set_yticks([0., 2.])
ax1.text(-0.4, 1.1, '(a)', fontsize=18)
ax2.set_ylabel(r'$a.u.$', fontsize=16)
ax1.scatter(m.X[-1], m.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
print(m)
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


