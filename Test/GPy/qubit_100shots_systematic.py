#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import copy , sys
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
sys.path.append('/home/fred/Desktop/GPyOpt/')
import GPyOpt


### ============================================================ ###
# Preparation:
### ============================================================ ###
def proba(x, noise=0, model = 0):
    #generate underlying proba p(|1>)
    n = 3
    s = 2
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    if model == 0:
        p = np.square(np.sin(x))
    elif model == 1:    
        p = np.abs(np.sin(n * x_noise) * np.exp(- np.square((x_noise-np.pi/2) / s)))
    else:
        raise NotImplementedError()
    return p

def measure(x, nb_measures = 1, verbose = False, model = 0, noise_input = 0):
    p = proba(x = x, noise= noise_input, model = model)
    if(nb_measures == np.inf):
        res = p
    elif(nb_measures < 1):
        res = p +np.random.normal(0, nb_measures, size = np.shape(p))
    else:
        res = rdm.binomial(nb_measures, p)/nb_measures
        
    if(verbose):
        print(np.around(np.c_[res, p, x], 3))
    return res

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)

def get_bests_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y and best expt for
    X already visited"""
    y_seen = np.min(bo.Y)
    x_seen = bo.X[np.argmin(bo.Y)]
    Y_pred = bo.model.predict(bo.X)
    y_exp = np.min(Y_pred[0])
    x_exp = bo.X[np.argmin(Y_pred[0])]
    
    return (x_seen, y_seen), (x_exp, y_exp)

def do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', X_init = None, Y_init = None, 
                    cost = None, cost_test = None, x_range = (0, np.pi), 
                    nb_iter = 50, nb_measures = None):
    """ Wrap up a BO optimization 
    """
    # basis parameter of the BO optimizer
    bounds_bo = [{'name': '0', 'type': 'continuous', 'domain': x_range}]
    if type_acq == 'EI':
        base_dico = {'acquisition_type':'EI', 'domain': bounds_bo, 
                     'optim_num_anchor':15, 'optim_num_samples':10000} 
    elif type_acq == 'LCB':
        base_dico = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                     'optim_num_anchor':15, 'optim_num_samples':10000, 
                     'acquisition_weight':2, 'acquisition_weight_lindec':True} 
    
    base_dico['X'] = X_init
    base_dico['Y'] = Y_init

    if type_gp == 'binomial':
        base_dico.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                            'likelihood':'Binomial_' + str(nb_measures), 
                            'normalize_Y':False})

    BO = GPyOpt.methods.BayesianOptimization(cost, **base_dico)
    BO.run_optimization(max_iter = nb_iter)
    
    xy, xy_exp = get_bests_from_BO(BO)
    test = cost_test(xy[0])
    test_exp = cost_test(xy_exp[0])
    
    return test, test_exp, BO

def test_four_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures):
    """ Run four flavors of BO Gaussian/Binomial likelihood x EI/LCB"""
    res = np.zeros((4, 2))
    x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
    y_init = model(x_init)
    
    test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)

    res[0, 0] = test
    res[0, 1] = test_exp
    
    test, test_exp, BO = do_one_BO_optim(type_acq = 'LCB', type_gp='gaussian', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)

    res[1, 0] = test
    res[1, 1] = test_exp
    if(nb_measures != np.inf):
        test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='binomial', 
            X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
            x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)
    
        res[2, 0] = test
        res[2, 1] = test_exp
             
        test, test_exp, BO = do_one_BO_optim(type_acq = 'LCB', type_gp='binomial', 
            X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
            x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)
    
        res[3, 0] = test
        res[3, 1] = test_exp
    else:
        na = 0
        res[2, 0] = na
        res[2, 1] = na
        res[2, 0] = na
        res[2, 1] = na
    
    return res

### ============================================================ ###
# Main results
### ============================================================ ###
x_range = (0, np.pi)
# ------------------------------------------ #
# models to simulate
# ------------------------------------------ #
verbose = True
f0_perfect = lambda x: measure(x, nb_measures = np.inf, verbose = verbose, model = 0, noise_input = 0)
f0_gaussian = lambda x: 1- measure(x, nb_measures = 0.2, verbose = verbose, model = 0, noise_input = 0)
f0_bin1 = lambda x: 1- measure(x, nb_measures = 1, verbose = verbose, model = 0, noise_input = 0)
f0_bin3 = lambda x: 3 * (1- measure(x, nb_measures = 3, verbose = verbose, model = 0, noise_input = 0))
f0_bin50 = lambda x: 50 * (1- measure(x, nb_measures = 50, verbose = verbose, model = 0, noise_input = 0))

f0_gaussian_noise = lambda x: 1- measure(x, nb_measures = 0.2, verbose = verbose, model = 0, noise_input = 0.05)
f0_bin1_noise = lambda x: 1- measure(x, nb_measures = 1, verbose = verbose, model = 0, noise_input = 0.05)
f0_bin3_noise = lambda x: 3 * (1- measure(x, nb_measures = 3, verbose = verbose, model = 0, noise_input = 0.05))
f0_bin50_noise = lambda x: 50 * (1- measure(x, nb_measures = 50, verbose = verbose, model = 0, noise_input = 0.05))

f1_perfect = lambda x: measure(x, nb_measures = np.inf, verbose = verbose, model = 1, noise_input = 0)
f1_gaussian = lambda x: 1- measure(x, nb_measures = 0.2, verbose = verbose, model = 1, noise_input = 0)
f1_bin1 = lambda x: 1- measure(x, nb_measures = 1, verbose = verbose, model = 1, noise_input = 0)
f1_bin3 = lambda x: 3 * (1- measure(x, nb_measures = 3, verbose = verbose, model = 1, noise_input = 0))
f1_bin50 = lambda x: 50 * (1- measure(x, nb_measures = 50, verbose = verbose, model = 1, noise_input = 0))

f1_gaussian_noise = lambda x: 1- measure(x, nb_measures = 0.2, verbose = verbose, model = 1, noise_input = 0.05)
f1_bin1_noise = lambda x: 1- measure(x, nb_measures = 1, verbose = verbose, model = 1, noise_input = 0.05)
f1_bin3_noise = lambda x: 3 * (1- measure(x, nb_measures = 3, verbose = verbose, model = 1, noise_input = 0.05))
f1_bin50_noise = lambda x: 50 * (1- measure(x, nb_measures = 50, verbose = verbose, model = 1, noise_input = 0.05))


# model, nb_measures, model_test, nb_init, nb_iter, name
list_test = [(f0_bin1, 1, f0_perfect, 50, 50, 'f0_bin1')]

list_to_simul_f0 = [(f0_gaussian, np.inf, f0_perfect, 50, 50, 'f0_gaussian'),
                    (f0_bin1, 1, f0_perfect, 50, 50, 'f0_bin1'),
                    (f0_bin3, 3, f0_perfect, 16, 17, 'f0_bin3'),
                    (f0_bin50, 100, f0_perfect, 15, 85, 'f0_bin50')]

list_to_simul_f0_noise = [(f0_gaussian_noise, np.inf, f0_perfect, 50, 50, 'f0_gaussian_noise'),
                    (f0_bin1_noise, 1, f0_perfect, 50, 50, 'f0_bin1_noise'),
                    (f0_bin3_noise, 3, f0_perfect, 16, 17, 'f0_bin3_noise'),
                    (f0_bin50_noise, 100, f0_perfect, 15, 85, 'f0_bin50_noise')]

list_to_simul_f1 = [(f1_gaussian, np.inf, f1_perfect, 50, 50, 'f1_gaussian'),
                    (f1_bin1, 1, f1_perfect, 50, 50, 'f1_bin1'),
                    (f1_bin3, 3, f1_perfect, 16, 17, 'f1_bin3'),
                    (f1_bin50, 100, f1_perfect, 15, 85, 'f1_bin50')]

list_to_simul_f1_noise = [(f1_gaussian_noise, np.inf, f1_perfect, 50, 50, 'f1_gaussian_noise'),
                    (f1_bin1_noise, 1, f1_perfect, 50, 50, 'f1_bin1_noise'),
                    (f1_bin3_noise, 3, f1_perfect, 16, 17, 'f1_bin3_noise'),
                    (f1_bin50_noise, 100, f1_perfect, 15, 85, 'f1_bin50_noise')]

# ------------------------------------------ #
# GOGOGOGOGOGOGOGOGGOGOGHOGGOOG
# ------------------------------------------ #
nb_repetitions = 1

# simple rabbi dynamics
dico_res_f0 = {}
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0:
    res_tmp = np.array([test_four_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res_f0[name] = res_tmp

# simple rabbi dynamics
dico_res_f0_noise = {}
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_noise:
    res_tmp = np.array([test_four_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res_f0_noise[name] = res_tmp


# more complicado: three local minimas
dico_res_f1 = {}
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1:
    res_tmp = np.array([test_four_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res_f1[name] = res_tmp

# more complicado: three local minimas and noise
dico_res_f1_noise = {}
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1_noise:
    res_tmp = np.array([test_four_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res_f1_noise[name] = res_tmp    



### ============================================================ ###
# Process results get graphs
# 0 - plot data used
# 1 - plot gaussian processes
### ============================================================ ###
# ------------------------------------------ #
# 0 Preparation for f0 et f1
# ------------------------------------------ #


# ------------------------------------------ #
# 1 Gaussian Noisey0_EI_gaussian_test = measure(xy0_EI_gaussian[0], nb_measures = np.inf, verbose = True)
# ------------------------------------------ #


# ------------------------------------------ #
# 2 Binomial obs with a classical BO
# ------------------------------------------ #
