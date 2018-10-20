#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:34:39 2018

@author: fred
"""
import logging, pdb
logger = logging.getLogger(__name__)
import numpy as np
#if(__name__ == '__main__'):
#    import sys
#    sys.path.append("../../../")
#    from QuantumSimulation.ToyModels import ModelBase as mod
#    from QuantumSimulation.Utility.Optim import pFunc_base as pf
#    
#else:
#    from ...Utility.Optim import pFunc_base as pf
#    from .. import ModelBase as mod


class model_examples():
    """ Toy functions to try the different optimization routines.
    Can be used with Learner object - has the mandatory attributes and methods:
        + n_params: nb of parameters
        + params_bounds: define the bounds for each params
        + __call__(x): function to minimize
    
    Extra:
        
    """    
    def __init__(self, params_bounds, noise_out = 0, noise_in = 0, bests= None):
        """ """
        self.params_bounds = params_bounds
        self.n_params = len(self.params_bounds)
        self.noise_in = noise_in
        self.noise_out = noise_out
        self.best_params = bests
    
    def __call__(self, x, **args):
        """ rely on underlying _f to be implemented in the subclasses
        On top some noise is added 
        """
        xnoise = model_examples._add_noise(x, self.noise_in)
        y = self._f(xnoise, **args)
        ynoise = model_examples._add_noise(y, self.noise_out)
        return ynoise
        
    def _f(self, x, **args):
        """
        
        """
        raise NotImplementedError()

    def gen_random(self, nb_obs = 1, method = 'unif'):
        """ Generate nb_obs based on some random"""
        if(method == 'unif'):
            #could do better
            res = [np.random.uniform(b[0], b[1], nb_obs) for b in self.params_bounds]
            res = np.transpose(res)
        else:
            raise NotImplementedError()
        return res
    
    def dist_to_a_min(self, x):
        """ """
        return np.min(np.abs(x - self.best_params))
    
    
    @staticmethod
    def _add_noise(z, noise = 0):
        """ Add noise with the right dimension - Later: implement different noise"""
        if noise > 0:
            return z + np.random.normal(0, noise, size = z.shape) 
        else:
            return z
    
    
class restricted_qubit(model_examples):
    """ MODEL: restricted qubits with projective measurement
    |phi(x)> = sin(x) |1> + cos(x) |0>
    with x \in [0, np.pi]
    
    """
    def __init__(self, nb_measures = 1, noise_out = 0, noise_in = 0, discrete = None, target = 1):
        bests = np.arcsin(target)
        model_examples.__init__(params_bounds = [(0, np.pi)], noise_out = noise_out, bests = bests)
        
        
    def proba(self, x, noise = None):
        """ generate underlying proba p(|1>) """
        noise = self.noise_out if noise is None else noise
        if(noise>0):
            x_noise = x + np.random.normal(0, noise, size = x.shape)
        else:
            x_noise = x
        return np.square(np.sin(x_noise))

    def measure(self, x):
        """ frequency of the projective measurement on |1>"""
        return np.random.binomial(self.nb_measures, self.proba(x))/self.nb_measures
        
    def _f(self, x, **args):
        return self.measure(x)
    
    def _true_call(self, x):
        return self.proba(x, noise = 0)
        
    
class localglobal(model_examples):
    """ A model which allows to fine tune the numbers of local global minimas in n-dim
    
    """

    def __init__(self, dim_input = 1, global_deg = 1, local_deg = 3, val_loc_degen = 0.5, tilt = 0, **args):
        mins = [[(0.4 + n) / (global_deg) for n in range(int(global_deg))] for d in range(dim_input)]
        model_examples.__init__(params_bounds = [(0, 1)] * dim_input, bests = mins, **args)
        self.local_deg = local_deg
        self.global_deg = global_deg
        self.tilt = tilt
        self.val_loc_degen = val_loc_degen
        

    def _f(self, x):
        """ f(X=[x1,.., xd]) = g(x1) x ... x g(xd)
        with g(x) = sin^2((x-0.1) * global_degen *pi) + cos^2((x-0.1) * local_degen * pi) + tilt * x 
        """
        freq_ppal = np.pi * self.global_deg
        freq_scnd = self.local_deg * freq_ppal
        res = 1
        for i in range(self.nb_params):
            XX = x[:,i] + 0.1  
            res *= self.tilt * XX - np.square(np.sin(XX * freq_ppal)) + self.val_loc_degen * np.square(np.cos(XX*freq_scnd))
        return res
