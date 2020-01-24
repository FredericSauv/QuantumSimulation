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

# Global params
NB_SHOTS = 128

# Wrapper functions
# all ensure that it can deal with several set of parameters (i.e. ndim(params)==2)
def f_average(params, shots = NB_SHOTS):
    """  estimate of the fidelity"""
    if np.ndim(params) >1 : res = np.array([f_average(p, shots=shots) for p in params])
    else: 
        res = qcirc.F(params, shots = shots)
        print(res)
        res = 1 - np.atleast_1d(res)
    return res


def f_test(params):
    if np.ndim(params) > 1 :
        res = np.array([f_test(p) for p in params])
    else:
        res = qcirc.F(params, shots=10000)
        print(res)
    return res

# BO setup
NB_INIT = 50
NB_ITER = 100
DOMAIN_DEFAULT = [(0, 2*np.pi) for i in range(6)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_DEFAULT)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata':NB_INIT,
                   'model_update_interval':1, 'hp_update_interval':5, 
                   'acquisition_type':'LCB', 'acquisition_weight':5, 
                   'acquisition_weight_lindec':True, 'optim_num_anchor':5, 
                   'optimize_restarts':1, 'optim_num_samples':10000, 'ARD':False}

            


### Run optimization        
Bopt = GPyOpt.methods.BayesianOptimization(f_average, **BO_ARGS_DEFAULT)    
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)


### Look at results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
print(f_test(x_seen))
print(f_test(x_exp))

Bopt.plot_convergence()