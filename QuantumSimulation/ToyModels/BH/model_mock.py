#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:34:39 2018

@author: fred
"""
import logging
logger = logging.getLogger(__name__)
import sys, pdb
import numpy as np 


if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ...Utility.Optim import pFunc_base as pf
    from .. import ModelBase as mod


## NOTES: before implementing such a class need to refactor modelBase models
##        track_learning functionalities should be implemented earlier on
##        i.e. on c_model or model

# class bowl_example(mod.pcModel_base):
#     """ Toy functions to try the different optimization routines.
#     Can be used with Learner object - has the mandatory attributes and methods:
#         + n_params: nb of parameters
#         + params_bounds: define the bounds for each params
#         + __call__(x): function to minimize
    
#     Extra:
        
#     """    

#     def setup_MOCK_model(self, model_config, test_config):
#         """ A simple toy function to test the optimizers 
#         Bowl function (x - x_tgt).(x - x_tgt)
#         Convex -> such that gradient methods should always converge to the minima
#         """
#         logger.info("+++ Mock function IN USE +++")
#         # potential arguments
#         verbose = model_config.get('verbose', False)
#         self.n_params = model_config.get('n_params', 4)
#         self.noise_output = model_config.get('noise_output', 0)
#         # fixed
#         self.nb_output, self.n_meas = 1, 1        
#         self.domain = [(0, 1) for _ in range(self.n_params)]
#         self.shift = np.random.uniform(0,1, self.n_params)
#         self.x_tgt = self.shift
#         self.model, self.model_test = None, None 
#         self.phi_0 = None
#         self.phi_tgt = None
#         self.T = None
#         self.e_tgt = None
#         self.p_tgt = None
#         self.f_tgt = None 
#         self.fid_zero = None
        
#         def f(x, verbose = verbose, trunc_res = True, **args_call):
#             """ evaluate the mock function"""
#             if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
#                 if(verbose): print(np.shape(x))
#                 res = np.array([f(x_one, verbose, trunc_res, **args_call) for x_one in x])
#             else:
#                 res = np.dot(x - self.shift, x-self.shift)
#                 if(self.noise_output > 0):
#                     res += np.random.normal(0, self.noise_output)
#                 self.call_f += 1
#                 if(verbose): print(x, res)
#             return np.atleast_1d(res)
        
#         def f_test(x, verbose = verbose, trunc_res = True, **args_call):
#             """ evaluate the mock function"""
#             if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
#                 if(verbose): print(np.shape(x))
#                 res = np.array([f_test(x_one, verbose, trunc_res, **args_call) for x_one in x])
#             else:
#                 res = np.dot(x - self.shift, x-self.shift)
#                 self.call_f_test += 1
#                 if(verbose): print(x, res)
#             return np.atleast_1d(res)
        
#         return f, f_test, self.x_tgt

        
#     def __init__(self, params_bounds, noise_out = 0, noise_in = 0, bests= None):
#         """ """
#         self.params_bounds = params_bounds
#         self.n_params = len(self.params_bounds)
#         self.noise_in = noise_in
#         self.noise_out = noise_out
#         self.best_params = bests
    
#     def __call__(self, x, **args):
#         """ rely on underlying _f to be implemented in the subclasses
#         On top some noise is added 
#         """
#         xnoise = model_examples._add_noise(x, self.noise_in)
#         y = self._f(xnoise, **args)
#         ynoise = model_examples._add_noise(y, self.noise_out)
#         return ynoise
        
#     def _f(self, x, **args):
#         """
        
#         """
#         raise NotImplementedError()

#     def gen_random(self, nb_obs = 1, method = 'unif'):
#         """ Generate nb_obs based on some random"""
#         if(method == 'unif'):
#             #could do better
#             res = [np.random.uniform(b[0], b[1], nb_obs) for b in self.params_bounds]
#             res = np.transpose(res)
#         else:
#             raise NotImplementedError()
#         return res
    
#     def dist_to_a_min(self, x):
#         """ """
#         return np.min(np.abs(x - self.best_params))
    
    
#     @staticmethod
#     def _add_noise(z, noise = 0):
#         """ Add noise with the right dimension - Later: implement different noise"""
#         if noise > 0:
#             return z + np.random.normal(0, noise, size = z.shape) 
#         else:
#             return z