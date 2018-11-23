#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:27:22 2018

@author: fred
"""

import sys
import GPy
sys.path.append('/home/fred/Desktop/GPyOpt/')
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mv
from matplotlib import cm
#==============================================================================
# Utilities
#==============================================================================
nb_points = 100
nb_obs = 5
def cost_func(X, noise = None):
    if hasattr(X, '__iter__'):
        res = np.array([cost_func(x, noise) for x in X])
    else:
        res = -np.cos(np.pi*X) +np.sin(4*np.pi*X)
        if (noise is not None):
            res += np.random.randn() * noise

    return  res

X = np.linspace(0.0, 1.0, nb_points)
#X_obs = np.random.uniform(low = 0.0, high = 1.0, size = nb_obs)
Y_ideal = cost_func(X)
#Y_obs = cost_func(X_obs, noise =0.2)


def get_plot_infos(data, cmap):
    list_color_value = (data - np.min(data)) / (np.max(data) - np.min(data))
    return [np.array(cmap(l))[:3] for l in list_color_value]
    
save_plt = False
col_ref = np.array([0.03137255, 0.18823529, 0.41960784])
#==============================================================================
# Init optim
#==============================================================================
myf = lambda x: cost_func(x, noise = 0.0)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]

#Init (how to select how many we want)
myOpt = GPyOpt.methods.BayesianOptimization(myf,bounds, initial_design_numdata = 5, acquisition_type ='LCB')
myOpt.run_optimization()
Y_obs = np.squeeze(myOpt.Y)
X_obs = np.squeeze(myOpt.X)
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
Y_obs_resc = (Y_obs - shift)/sd
X_best = 0.36
Y_best = cost_func(X_best, 0)
Y_best_resc = (Y_best-shift)/sd


myOpt.plot_acquisition(flo_mode = True)
plt.plot(X, Y_tmp, 'r--', label = 'F')
plt.scatter(X_obs, Y_obs_resc, marker = 'o', color='red', label = r'$Observations$')
#plt.scatter(X_best, Y_best_resc, marker = '*', s = 120, color='green', label = r'$To\;find$')
plt.ylabel(r"$F(x)$", fontsize=16)
plt.xlabel(r"$x$", fontsize=16)
#plt.legend(loc=4)
if(save_plt):
    plt.savefig("init_custom_b_plustwo.pdf",bbox_inches='tight')
myOpt.run_optimization(1)


plt.plot(X, Y_tmp, '--', color = col_ref, label = 'F')
plt.scatter(X_obs, Y_obs_resc, marker = 'o', color='red', label = r'$Observations$')
plt.scatter(X_best, Y_best_resc, marker = '*', s = 100, color='green', label = r'$To\;find$', zorder = 100)
plt.ylabel(r"$F(x)$", fontsize=20)
plt.xlabel(r"$x$", fontsize=20)
#plt.legend()
if(save_plt):
    plt.savefig("goal_init.pdf",bbox_inches='tight')
    
    
#==============================================================================
# Prior Samples
#==============================================================================
ker_mimic = GPy.kern.Matern52(variance = 1.0, lengthscale = 0.147, input_dim =1)
nb_points_path = 100
nb_func = 50
X_kernel = np.linspace(0.,1., nb_points_path)[:,None]
mu_kernel = np.zeros((nb_points_path))
C_kernel = ker_mimic.K(X_kernel,X_kernel)
sample_paths = np.squeeze(np.random.multivariate_normal(mu_kernel, C_kernel, nb_func))
mv0 = mv(mean = mu_kernel, cov = C_kernel, allow_singular=True)
log_p = mv0.logpdf(sample_paths)
tmp =  np.power(log_p / np.sum(log_p), 8)
scale = 2 /np.max(tmp)
p_norm = scale * tmp
color_tmp = get_plot_infos(p_norm, cm.Blues)
index_tmp = np.argsort(p_norm)

for n in index_tmp:
    plt.plot(X_kernel[:], sample_paths[n], color = color_tmp[n], linewidth = p_norm[n])
plt.ylabel(r"$f(x)$")
plt.xlabel(r"$x$")
if(save_plt):
    plt.savefig("prior_samples.pdf",bbox_inches='tight')


#==============================================================================
# Likelihood samples
#==============================================================================

index_obs = np.array([ int(a) for a in (X_obs * nb_points_path)])
eps = np.array([s - np.squeeze(Y_obs_resc) for s in sample_paths[:, index_obs]])

logL = np.sum(- np.square(eps)/(2* np.sqrt(0.2)),1)
width = -4.2/logL * 2
color_tmp = get_plot_infos(width, cm.Blues)
index_tmp = np.argsort(width)

for n in index_tmp:
    plt.plot(X_kernel[:], sample_paths[n], color = color_tmp[n], linewidth = width[n], alpha=0.9)
plt.scatter(X_obs, Y_obs_resc, color='r', marker='o', alpha=1, label=r'$Observations$', zorder = 100)
plt.ylabel(r"$f(x)$")
plt.xlabel(r"$x$")
plt.legend(loc=4, fontsize=12)
if(save_plt):
    plt.savefig("likelihood_samples.pdf",bbox_inches='tight')
#color_most = color_tmp[n]

#==============================================================================
# Posterior samples
#==============================================================================
post = myOpt.model.model.posterior_samples_f(X_kernel, nb_func)
for n, zz in enumerate(np.transpose(post)):
    plt.plot(X_kernel[:], zz, color = col_ref, linewidth = 1)
plt.scatter(X_obs, Y_obs_resc, color='r', marker='o', alpha=1, label = r"$Observations$", zorder=100)
plt.legend(loc=4, fontsize=12)
plt.ylabel(r"$f(x)$")
plt.xlabel(r"$x$")
if(save_plt):
    plt.savefig("posterior_samples.pdf",bbox_inches='tight')
#plt.savefig('posterior_more_2.pdf')


m, v = myOpt.model.predict(X[:, None])
lower = m - 1.96*np.sqrt(v)
upper = m + 1.96*np.sqrt(v)
plt.plot(X[:], m, 'b', label=r"$\mu(x) = E_{f| \mathcal{D}}[f(x)] $", color = col_ref)
plt.plot(X[:], lower, 'b', linewidth=0.2)
plt.plot(X[:], upper, 'b', linewidth=0.2)
plt.fill_between(np.squeeze(X), np.squeeze(upper), np.squeeze(lower), alpha=0.2,label=r"$\mu(x) + 1.96 \sigma(X)$")
plt.scatter(X_obs, Y_obs_resc, color='r',  marker='o', label = r"$Observations$" )
plt.legend()
if(save_plt):
    plt.savefig("prediction.pdf",bbox_inches='tight')


#==============================================================================
# Next steps 1
#==============================================================================
myOpt.run_optimization(1)

Y_obs_one = np.squeeze(myOpt.Y)
X_obs_one = np.squeeze(myOpt.X)
shift_one = np.average(Y_obs_one)
sd_one = np.std(Y_obs_one)
Y_tmp_one = (Y_ideal - shift_one)/sd_one
Y_obs_resc_one = (Y_obs_one - shift_one)/sd_one
Y_best_resc_one = (Y_best-shift_one)/sd_one



myOpt.plot_acquisition()
plt.plot(X, Y_tmp_one, 'r--', label = 'F')
plt.scatter(X_best, Y_best_resc_one, marker = '*', s = 120, color='green', label = r'$To\;find$')
plt.scatter(X_obs_one, Y_obs_resc_one, marker = 'o', color='red', label = r'$Observations$')
plt.ylabel(r"$F(x)$")
plt.xlabel(r"$x$")
plt.legend()
if(save_plt):
    plt.savefig("fitting_plusone.pdf",bbox_inches='tight')



#==============================================================================
# Next steps
#==============================================================================
myOpt.run_optimization(1)

Y_obs_five = np.squeeze(myOpt.Y)
X_obs_five = np.squeeze(myOpt.X)
shift_five = np.average(Y_obs_five)
sd_five = np.std(Y_obs_five)
Y_tmp_five = (Y_ideal - shift_five)/sd_five
Y_obs_resc_five = (Y_obs_five - shift_five)/sd_five
Y_best_resc_five = (Y_best-shift_five)/sd_five
myOpt.plot_acquisition()
plt.plot(X, Y_tmp_five, 'r--', label = 'F')
plt.scatter(X_best, Y_best_resc_five, marker = '*', s = 120, color='green', label = r'$To\;find$')
plt.scatter(X_obs_five, Y_obs_resc_five, marker = 'o', color='red', label = r'$Observations$')
plt.ylabel(r"$F(x)$")
plt.xlabel(r"$x$")
plt.legend()
if(save_plt):
    plt.savefig("fitting_plussix.pdf",bbox_inches='tight')



#5 Next
myOpt.run_optimization(5)
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)
if(save_fig):
    plt.savefig('plot_GP/BO_init_plus_eleven_noisy.pdf'.format(l))

myOpt.plot_convergence()
if(save_fig):
    plt.savefig('plot_GP/BO_convergence_noisy.pdf'.format(l))
