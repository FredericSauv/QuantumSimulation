#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:28:25 2019

@author: fred
"""
import numpy as np
import numpy.random as rdm
import scipy.stats as stats
import scipy.special as spe
import matplotlib.pylab as plt
#import gpflow
#import torch
#import pyro
from matplotlib.ticker import MultipleLocator

col_custom = (0.1, 0.2, 0.5)

### ============================================================ ###
# MODEL: restricted qubits with projective measurement
# |phi(x)> = sin(x) |1> + cos(x) |0>
### ============================================================ ###
def proba_1(x, noise=0):
    #generate underlying proba p(|1>) landscape
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
    return np.square(np.sin(x_noise))

def proba_2(x, noise=0):
    xx = 2 * (x-2)
    return 1 - (2*np.power(xx, 2) - 1.05 * np.power(xx,4) + 1/6*np.power(xx, 6))/450


def measure(x, underlying_p = proba_1, nb_measures = 1, noise_gauss = None):
    """ perform projective measurements |1><1|"""
    proba = underlying_p(x)
    if nb_measures == np.inf:
        res = proba 
        if noise_gauss is not None:
            res += np.random.normal(loc=0., scale=noise_gauss, size=proba.shape)
    else:
        res = rdm.binomial(nb_measures, proba)/nb_measures
    return res

def squerror(p_pred, p_true):
    """ Square error"""
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)

def invcdf(y, alpha=1):
    return  np.sqrt(2) * spe.erfinv(2*y-1) / alpha 

def changedistrib(y, mu, var, alpha=1):
    """
    Y~N(mu, var)
    Z~
    """
    inv = invcdf(y,alpha)
    res = stats.norm.pdf(inv, loc=mu, scale=var) / (alpha * stats.norm.pdf( alpha * inv))
    res[np.isnan(res)]=0
    return  res



### ============================================================ ###
#Main functions working on each model
### ============================================================ ###   
def which_type(model):
    type_m = str(type(model))
    if 'GPy' in type_m:
        return 'gpy'
    elif 'gpflow' in type_m:
        return 'gpflow'
    elif 'sklearn' in type_m:
        return 'sklearn'
    else:
        return 'unknown'

def which_lik(model):
    """ Discriminate between gaussian likelihood and non-gaussian
    """
    res = 'unk'
    type_m = which_type(model)
    if type_m == 'gpy':
        if 'Gaussian' in model.likelihood.name:
            res = 'gauss'
        else:
            res = 'nongauss'
    elif type_m == 'gpflow':
        if 'Gaussian' in model.likelihood.name:
            res = 'gauss'
        else:
            res = 'nongauss'
    elif type_m == 'sklearn':
        if 'Gaussian' in str(type(model)):
            res = 'gauss'
        else:
            res = 'nongauss'
    return res

def get_data(model):
    if which_type(model) == 'gpy':
        X, Y = model.X, model.Y
    elif which_type(model) == 'gpflow':
        X, Y = model.X.value, model.Y.value
    elif which_type(model) == 'sklearn':
        X, Y = model.X_train_, model.y_train_
    else:
        X, Y = None, None
    return X, Y

def predict_p(model, X, alpha=1):
    """
        Output:
        + median value
        + quantile 2.5%
        + quantile 97.5%
        + density
        + p_min
        + p_max
        + std
    """
    N_discr = 5000
    type_m = which_type(model)
    if type_m == 'gpy':
        mean, var = model.predict(X, likelihood=None, include_likelihood=False)
    elif type_m == 'gpflow':
        mean, var =model.predict_f(X, alpha)
    elif type_m == 'sklearn':
        mean, std = model.predict(X, return_std=True)
        var = np.square(std)[:, np.newaxis]
    
    if which_lik(model) == 'nongauss':    
        p_min, p_max = 0, 1
        y = np.linspace(p_min, p_max,N_discr)
        density = np.array([changedistrib(y, m, np.sqrt(v), alpha) for m, v in zip(mean, var)])
        mean = np.array([np.sum(d * y)/len(y) for d in density])
        densitycum = np.cumsum(density, axis=1)/N_discr
        #m_median = np.array([y[np.argwhere(dc>=0.5)[0,0]] for dc in densitycum])
        q = np.array([(y[np.argwhere(dc>=0.025)[0,0]], y[np.argwhere(dc<=0.975)[-1,0]]) for dc in densitycum])
        qf, ql = q[:,0], q[:,1]
        std = np.array([np.sqrt(np.sum(np.square(y-m) * d)/N_discr) for d, m in zip(density, mean)])
        density = np.transpose(density)
    else:    
        std = np.sqrt(var)
        qf, ql = mean -1.96 * std, mean + 1.96 * std
        p_min, p_max = np.min(mean -2.58 * std), np.max(mean +2.58 * std) 
        y = np.linspace(p_min, p_max, N_discr)
        density = np.transpose([stats.norm.pdf(y, loc = m, scale = s) for m, s in zip(mean, std)])

    return mean, qf, ql, density, (p_min, p_max), std


#Examples acquisitions
def next_ucb_bounds(model, loc, w=4):
    """ UCB acquisition with bou"""
    a, b, c, _, _, s = predict_p(model, loc)
    acq = (a + w * (c-b))
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_std(model, loc, w=4):
    a, _, _, _, _, s = predict_p(model, loc)
    acq = (a + w * s)
    x_next = loc[np.argmax(acq)]
    return x_next

#
def add_new_obs(model, X_new, Y_new, fit = True, **args):
    type_m = which_type(model)
    if type_m == 'gpy':
        X_updated = np.vstack([model.X, X_new])
        Y_updated = np.vstack([model.Y, Y_new])
        #model.Y_metadata = {'trials':np.ones_like(Y_new)*1}
        model.set_XY(X_updated, Y_updated)
        if fit > 0:
            model.optimize_restarts(num_restarts=fit)

def next_step(model, measure, acq = next_ucb_std, print_next=True):
    x_next = acq(model)
    if print_next: print('x_next: '+str(x_next))
    y_next = measure(x_next)
    add_new_obs(model, x_next, y_next, fit=1)
    


### ============================================================ ###
# GPy utility
### ============================================================ ###    
#def predict_p_gpy(model, X, alpha=1):
#    """ Based on the underlying GPymodel return details of the surrogate model
#    Remarks in case of lik != Gaussian plots over p-space (rather than f-space) 
#    
#    Output:
#        + median value
#        + quantile 2.5%
#        + quantile 97.5%
#        + density
#        + p_min
#        + p_max
#        + std
#    
#    """
#    mean, var = model._raw_predict(X, full_cov=False, kern=None)
#    N_discr = 5000
#    y = np.linspace(0, 1,N_discr)
#    density = np.array([changedistrib(y, m, np.sqrt(v), alpha) for m, v in zip(mean, var)])
#    p_min, p_max = 0, 1
#    m_mean = np.array([np.sum(d * y)/len(y) for d in density])
#    densitycum = np.cumsum(density, axis=1)/N_discr
#    #m_median = np.array([y[np.argwhere(dc>=0.5)[0,0]] for dc in densitycum])
#    q = np.array([(y[np.argwhere(dc>=0.025)[0,0]], y[np.argwhere(dc<=0.975)[-1,0]]) for dc in densitycum])
#    std = np.array([np.sqrt(np.sum(np.square(y-m) * d)/N_discr) for d, m in zip(density, m_mean)])
#    return m_mean, q[:,0], q[:,1], np.transpose(density), (p_min, p_max), std
#





### ============================================================ ###
# plot 
# Should be agnostic of the underlying model
### ============================================================ ### 
def plot_model(model, ideal, sd=None):
    """ Simply plot the model, obs, uncertainty
    """
    x_test, p_test = ideal
    X_obs, Y_obs = get_data(model)
    a, b, c, d, _, s = predict_p(model, x_test)
    fig, ax1 = plt.subplots(1, 1)# ,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})

    ax1.plot(x_test, a, color = col_custom, linewidth = 0.8, label = r'$model$')
    ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.4)
    ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.4)
    ax1.scatter(X_obs, Y_obs, label = r'$Observations$', marker = 'o', c='red', s=25)
    ax1.scatter(X_obs[-1], Y_obs[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
    ax1.plot(x_test, p_test, 'r--', label='F')
    if which_lik(model) == 'gaussian':
        ax1.set_ylim([-0.2, 1.2])
    else:
        ax1.set_ylim([-0.02, 1.02])
    
    ax1.set_xlim([-0.02, 4.02])
    ax1.set_xticklabels([], visible= False)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_yticks(np.arange(0,1.01, 0.2))
    ax1.set_xticks(np.arange(0,4.01, 0.5))
    #ax1.text(-0.4, 1.1, '(a)', fontsize=18)
    
    # bounds are either mean +- alpha std or quantiles returned by predict
    if sd is not None:
        binf = a - sd * s 
        bsup = a + sd * s 
    else:
        binf = b 
        bsup = c
    ax1.fill_between(x_test.reshape(-1), binf.reshape(-1), y2=bsup.reshape(-1), alpha=0.1)
    if is_sparse(model) and model.feature.Z.value.shape != model.X.shape:
        ax1.scatter(model.feature.Z.value, np.zeros(model.feature.Z.value.shape),c='r', marker="|")

def plot_extensive(model, ideal, w_acq = 4, save = None, proba = None, best = False, subfig = None, y_bounds = None,y_bounds2 = None, ylabel1 = None, ylabel2=None, xlabel=None, remove_tick=False, weight_lines = 3, font_size = 20):
    """ Plot for nice figures with acauisition function
    """
    version = 'V7'
    col_custom = (0.1, 0.2, 0.5)
    x_test = ideal[0]
    p_test = ideal[1]
    model_type = which_type(model)
    a, b, c, d, d_range, s = predict_p(model, x_test)
    y_bounds = y_bounds if y_bounds is not None else [-0.2, 1.2]
    y_bounds2 = y_bounds2 if y_bounds2 is not None else [0., 2.]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
    
    # First graph plot model with confidence bounds
    ax1.plot(x_test, a, color = col_custom, linewidth = weight_lines, label = r'$model$')
    ax1.plot(x_test, b, color = col_custom, alpha = 0.5, linewidth = 0.5 * weight_lines)
    ax1.plot(x_test, c, color = col_custom, alpha = 0.5, linewidth = 0.5 * weight_lines)
    # plot observations
    ax1.scatter(model.X, model.Y, label = r'$Observations$', marker = 'o', c='red', s=35)
    # Add underlying landscape
    ax1.plot(x_test, p_test, 'r--', label='F', linewidth = weight_lines)
    ax1.scatter(model.X[-1], model.Y[-1], label = r'$New\;Obs$', marker = 's', c='g', s=100)
    # Add density
    if d is not None:    
        ax1.imshow(np.power(d[np.arange(len(d)-1,-1, -1)]/np.max(d), 1), cmap = 'Blues', 
                   aspect='auto', interpolation='spline16', extent = (x_test[0],  
                    x_test[-1], d_range[0], d_range[1]), alpha=1)

    ax1.set_ylim(y_bounds)
    ax1.set_xlim([-0.02, 4.02])
    
    # Acquisition function
    acq = (a + w_acq * s)
    x_next = x_test[np.argmax(acq)]
    ax2.plot(x_test, acq, color='r', linewidth = weight_lines)
    ax2.vlines(x_next, 0, np.max(acq), colors='r', linestyles='dashed', linewidth = weight_lines)
    ax2.xaxis.tick_top()
    ax1.set_xticklabels([], visible= False)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.set_yticks(np.arange(0,1.01, 0.2))
    ax1.set_xticks(np.arange(0,4.01, 0.5))
    ax2.set_xticks(np.arange(0,4.01, 0.5))
    #ax2.set_yticklabels([], visible= False)
    ax2.set_yticks(y_bounds2)
    ax2.set_xlim([-0.02, 4.02])
    if remove_tick:
        ax1.set_yticklabels([], visible= False)
        ax2.set_yticklabels([], visible= False)

    ## Extra options    
    if subfig is not None:
        ax1.text(-0.4, 1.1, str(subfig), fontsize=font_size)
    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1, fontsize=font_size+3)
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2, fontsize=font_size+3)
    if xlabel is not None:
        ax2.text(1., -0.8, xlabel, fontsize=font_size+3)
    #'Control parameter $\theta$'
    x_best = x_test[np.argmax(a)]
    if best:    
        ax1.vlines(x_best, y_bounds[0],1, colors='g', label = r'$\theta_{best}$', linewidth = weight_lines)             
    if proba is not None:
        pres = proba(x_best)
        print(pres)
    else: 
        pres = 'unk'
    if save is not None:
        plt.savefig(save + version + model_type+""+str(pres)+".pdf", bbox_inches='tight', transparent=True, pad_inches=0)
    


### ============================================================ ###
# GPflow
### ============================================================ ###  
def is_sparse(model):
    return (which_type(model)=='gpflow') and ('SVGP' in model.name)


#def predict_p_gpflow(model, X, alpha=1):
#    """ Based on the underlying GPymodel return details of the surrogate model
#    Remarks in case of lik != Gaussian plots over p-space (rather than f-space) 
#    
#    Output:
#        + median value
#        + quantile 2.5%
#        + quantile 97.5%
#        + density
#        + p_min
#        + p_max
#        + std
#    
#    """
#    t_m, t_v = model.predict_y(X)
#    N_discr = 5000
#    y = np.linspace(0, 1,N_discr)
#    density = np.array([changedistrib(y, m, np.sqrt(v), alpha) for m, v in zip(mean, var)])
#    p_min, p_max = 0, 1
#    m_mean = np.array([np.sum(d * y)/len(y) for d in density])
#    densitycum = np.cumsum(density, axis=1)/N_discr
#    #m_median = np.array([y[np.argwhere(dc>=0.5)[0,0]] for dc in densitycum])
#    q = np.array([(y[np.argwhere(dc>=0.025)[0,0]], y[np.argwhere(dc<=0.975)[-1,0]]) for dc in densitycum])
#    std = np.array([np.sqrt(np.sum(np.square(y-m) * d)/N_discr) for d, m in zip(density, m_mean)])
#    return m_mean, q[:,0], q[:,1], np.transpose(density), (p_min, p_max), std
#    
#    
    
#def plotl_gpflow(model, ideal, sd=None):
#    """ To do get density/quantiles etc.."""
#    x_test = ideal[0]
#    p_test = ideal[1]
#    t_m, t_v = model.predict_y(x_test)
#    fig, ax1 = plt.subplots(1, 1)# ,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
#    ax1.plot(x_test, t_m, color = col_custom, linewidth = 0.8, label = r'$model$')
#    ax1.scatter(model.X.value, model.Y.value, label = r'$Observations$', marker = 'o', c='blue', s=25)
#    ax1.scatter(model.X.value[-1], model.Y.value[-1], label = r'$New\;Obs$', marker = 's', c='g', s=80)
#    ax1.plot(x_test, p_test, 'r--', label='F')
#    ax1.set_ylim([-0.02, 1.02])
#    ax1.set_xlim([-0.02, 4.02])
#    ax1.set_xticklabels([], visible= False)
#    ax1.tick_params(axis='both', which='major', labelsize=16)
#    ax1.set_yticks(np.arange(0,1.01, 0.2))
#    ax1.set_xticks(np.arange(0,4.01, 0.5))
#    ax1.text(-0.4, 1.1, '(a)', fontsize=18)
#    if sd is not None:
#        ax1.fill_between(x_test.reshape(-1), (t_m - sd * np.sqrt(t_v)).reshape(-1), y2=(t_m + sd * np.sqrt(t_v)).reshape(-1), alpha=0.1)
#    if is_sparse(model) and model.feature.Z.value.shape != model.X.shape:
#        ax1.scatter(model.feature.Z.value, np.zeros(model.feature.Z.value.shape),c='r', marker="|")


def train_gpflow(model, optim, train_loc=True, train_kernel=True):
    """ Fit the parameters of the model"""
    model.kern.set_trainable(train_kernel)
    if(is_sparse(model)):
        if train_loc:
            # Pre-train 
            model.feature.set_trainable(False)
            gpflow.train.ScipyOptimizer().minimize(model, maxiter=20)
        model.feature.set_trainable(train_loc)
    gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(model)

def add_data_gpflow(model, newX, newY, fixZ=False):
        """
        Update the underlying models of the acquisition function with new data.
        :param newX: samples, size N x D
        :param newY: values obtained by evaluating the objective and constraint functions, size N x R
        """
        model.X = np.vstack((model.X.value, newX))
        model.Y = np.vstack((model.Y.value, newY))
        if fixZ:
            model.Z =model.X.value


#def next_ucb_std_gpflow(model, loc, w=4):
#    a, v = model.predict_y(loc)
#    acq = (a + w * np.sqrt(v))
#    x_next = loc[np.argmax(acq)]
#    return x_next

def next_step_gpflow(model, loc, n_iters=1, w=4):
    for n in range(n_iters):
        pass


### ============================================================ ###
# Pyro
### ============================================================ ###  
def plot_pyro(model, ideal, sd_val=1):
    #plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
    x_test = ideal[0]
    p_test = ideal[1]
    plt.figure(figsize=(12, 6))
    plt.plot(model.X.numpy(), model.y.numpy(), 'kx')
    Xtest = torch.Tensor(x_test)  # test inputs
    with torch.no_grad():
        if type(model) in (pyro.contrib.gp.models.VariationalSparseGP, pyro.contrib.gp.models.VariationalGP):
            mean, cov = model(Xtest, full_cov=True)
        else:
            mean, cov = model(Xtest, full_cov=True, noiseless=False)
    sd = cov.diag().sqrt()  # standard deviation at each input point x
    mean_p = model.likelihood.response_function(mean)
    inf, sup = mean -sd_val * sd, mean + sd_val * sd
    inf_p = model.likelihood.response_function(inf)
    sup_p = model.likelihood.response_function(sup)
    plt.plot(Xtest.numpy(), mean_p.numpy(), 'b', lw=2)  # plot the mean
    plt.plot(Xtest.numpy(), p_test, 'r--', lw=2)  # plot real landscape
    plt.fill_between(Xtest.numpy().reshape(-1),  # plot the two-sigma uncertainty about the mean
                     inf_p.numpy().reshape(-1),
                     sup_p.numpy().reshape(-1),
                     color='C0', alpha=0.3)
        






### ============================================================ ###
# Comp
### ============================================================ ###  
def plot_comp(ideal, obs, models):
    """ To do get density/quantiles etc.."""
    x_test = ideal[0]
    p_test = ideal[1]
    x_obs = obs[0]
    y_obs = obs[1]
    fig, ax1 = plt.subplots(1, 1)# ,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
    ax1.scatter(x_obs, y_obs, label = r'$Observations$', marker = 'o', c='blue', s=25)
    ax1.plot(x_test, p_test, 'r--', label='F')
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlim([-0.02, 4.02])
    ax1.set_xticklabels([], visible= False)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_yticks(np.arange(0,1.01, 0.2))
    ax1.set_xticks(np.arange(0,4.01, 0.5))
    ax1.text(-0.4, 1.1, '(a)', fontsize=18)
    
    for m in models:        
        if 'GPy' in str(type(m)):
            name = 'GPy'
            a, b, c, d, d_range, s = predict_p(m, x_test)
            ax1.plot(x_test, a, linewidth = 0.8, label = name)
        elif 'gpflow' in str(type(m)):
            name = 'gpflow_inducedby_'+str(len(m.feature))
            t_m, t_v = m.predict_y(x_test)
            ax1.plot(x_test, t_m, linewidth = 0.8, label = r'$gpflow$')
        elif 'pyro' in str(type(m)):
            #plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
            Xtest = torch.Tensor(x_test)  # test inputs
            with torch.no_grad():
                if type(m) in (pyro.contrib.gp.models.VariationalSparseGP,pyro.contrib.gp.models.VariationalGP):
                    mean, cov = m(Xtest, full_cov=True)
                else:
                    mean, cov = m(Xtest, full_cov=True, noiseless=False)
            #sd = cov.diag().sqrt()  # standard deviation at each input point x
            mean_p = m.likelihood.response_function(mean)
            #inf, sup = mean -sd_val * sd, mean + sd_val * sd
            #inf_p = model.likelihood.response_function(inf)
            #sup_p = model.likelihood.response_function(sup)
            plt.plot(Xtest.numpy(), mean_p.numpy(), 'b', lw=2)  # plot the mean
            #plt.plot(Xtest.numpy(), p_test, 'r--', lw=2)  # plot real landscape
            #plt.fill_between(Xtest.numpy().reshape(-1),  
                #inf_p.numpy().reshape(-1), sup_p.numpy().reshape(-1), color='C0', alpha=0.3)
        
        
        
        
