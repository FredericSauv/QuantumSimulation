#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:39:09 2020

@author: fred
"""

import test_GHZ as qcirc
import sys
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/') ## Fork from https://github.com/FredericSauv/GPyOpt
import GPyOpt
import numpy as np

NB_INIT = 50
NB_ITER = 100
DOMAIN_DEFAULT = [(0, 2*np.pi) for i in range(6)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_DEFAULT)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata':NB_INIT,
                   'model_update_interval':1, 'hp_update_interval':5, 
                   'acquisition_type':'LCB', 'acquisition_weight':5, 
                   'acquisition_weight_lindec':True, 'optim_num_anchor':5, 
                   'optimize_restarts':1, 'optim_num_samples':10000, 'ARD':False}

    
def f_optim(params):
    """ Wrapper around f: 
        + ensure that it can deal with several set of parameters
        + return 1 - f to recast it as a minimization task
        + ensure that the output is a 1d array
    """
    if np.ndim(params) >1 :
        res = np.array([qcirc.F(p, sample_meas=True) for p in params])
    else:
        res = qcirc.F(params, sample_meas=True)
    print(res)
    return 1 - np.atleast_1d(res)
        
Bopt = GPyOpt.methods.BayesianOptimization(f_optim, **BO_ARGS_DEFAULT)    

Bopt.run_optimization(max_iter = NB_ITER, eps = 0)


### Look at results
def f_test(params):
    if np.ndim(params) >1 :
        res = np.array([qcirc.F(p, shots=10000) for p in params])
    else:
        res = qcirc.F(params, sample_meas=True)
    print(res)
    
best_x_measured = Bopt.x_opt
best_y_measured = Bopt.X
best_x_predicted
    
    