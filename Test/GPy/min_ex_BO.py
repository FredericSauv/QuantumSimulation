#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:12:14 2018

@author: fred
"""
import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
import GPyOpt
import numpy as np
import pdb


# define a class with a method which takes an array or arguuments and return a value
# here f
class my_f:
    _best = np.array([0.0898, -0.7126])
    def __init__(self, sd=None):
        self.input_dim = 2		
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd
        self.bounds = [(0,1)]

    # same function but with input/output shaped for BO
    def f_for_BO(self,X):
        X = np.array(X).reshape((len(X),self.input_dim))
        n = X.shape[0]
        fval = (4-2.1*X[:,0]**2+(X[:,0]**4)/3)*X[:,0]**2 +X[:,0]*X[:, 1]+(-4+4*X[:,1]**2) * X[:,1]**2
        
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        print(fval)
        return fval.reshape(n,1) + noise
    
    # same function but with input/output shaped for DE
    def f_for_DE(self, X):
        
        x1 = X[0]
        x2 = X[1]
        res = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2
        if self.sd !=0:
            res += np.random.normal(0,self.sd)
        print(res)
        return res


#First optim
myf =my_f(sd=0)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-3, 3)}, 
           {'name': 'var_2', 'type': 'continuous', 'domain': (-2, 2)}]

max_iter = 50
myOpt = GPyOpt.methods.BayesianOptimization(myf.f_for_BO, bounds, initial_design_numdata = 10, acquisition_type ='LCB')

# (self, f, domain = None, constraints = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None,
# initial_design_numdata = 5, initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
# exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential',
# batch_size = 1, num_cores = 1, verbosity=False, verbosity_model = False, maximize=False, de_duplication=False, **kwargs):
myOpt.run_optimization(max_iter)

# bestt values results
print(myOpt.fx_opt)
print(myOpt.x_opt)

#learning process
myOpt.plot_convergence()

import scipy.optimize as sco
resultOptim = sco.differential_evolution(myf.f_for_DE, [(-3,3), (-2,2)], popsize=5)
print(resultOptim)
