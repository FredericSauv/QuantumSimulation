#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:59 2018

@author: fred
"""
import sys
sys.path.append('/home/fred/Desktop/GPyOpt/')
import GPyOpt
import numpy as np
from BouncingAtom import model
import scipy.optimize as sco

#-----------------------------------------------------------------------------#
# Prepare everything
#-----------------------------------------------------------------------------#
# Parameters of the optimizer with some comments
options_bo ={'initial_design_numdata': 20, 'acquisition_type':'EI', 
                  'optim_num_samples':1000, 'optim_num_anchor':10,
                  'acquisition_weight':2, 'acquisition_weight_lindec':False}
max_iter = 100

# Initialize an bouncing atom model
atom_model = model()
h_ref, s, noise = 10.0, 40, 0.1
omega_ref = atom_model.get_omega_perfect(h=h_ref, s=s)

# Define parameter(s) to optimize
# Here we optimize the ratio omega / omega_ref
limits = (0.8, 1.2)
bounds = [{'name': 'omega_ratio', 'type': 'continuous', 'domain': limits}]

# define functions FoM functions
f = lambda x: atom_model.simulate_noisy_h0(omega = x * omega_ref, h_ref = h_ref, 
             noise_h0 = 0, nb_repeat = 1, s = s, nb_period = 1, verbose = True)
f_noisy = lambda x: atom_model.simulate_noisy_h0(omega = x * omega_ref, h_ref = h_ref, 
            noise_h0 = 0.1, nb_repeat = 20, s = 40, nb_period = 1, verbose = True)
f_testing = lambda x: atom_model.simulate_noisy_h0(omega = x * omega_ref, h_ref = h_ref, 
            noise_h0 = 0.1, nb_repeat = 1000, s = 40, nb_period = 1, verbose = True)

def _get_best_exp_from_BO(bo):
    """ From BO extract the best x"""
    Y_pred = bo.model.predict(bo.X)
    return bo.X[np.argmin(Y_pred[0])], np.min(Y_pred)

#-----------------------------------------------------------------------------#
# Optimization of the noiseless set-up
#-----------------------------------------------------------------------------#
myOpt = GPyOpt.methods.BayesianOptimization(f, bounds, **options_bo)
myOpt.run_optimization(max_iter)
x_no_noise, _ = _get_best_exp_from_BO(myOpt)

## Another optimizer (differential evolution)
resultOptim = sco.differential_evolution(f, [limits], popsize=5)
print(resultOptim)
x_no_noise_de = resultOptim['x']

#-----------------------------------------------------------------------------#
# Optimization of the noisy set-up
#-----------------------------------------------------------------------------#
myOpt = GPyOpt.methods.BayesianOptimization(f_noisy, bounds, **options_bo)
myOpt.run_optimization(max_iter)
x_noisy, _ = _get_best_exp_from_BO(myOpt)
 
#optim with DE (struggle to converge in the noisy set-up.. I limit it to 300 iterations)
resultOptim = sco.differential_evolution(f_noisy, [(0.8, 1.2)], popsize=5, maxiter = 300)
x_noisy_de = resultOptim['x']

#-----------------------------------------------------------------------------#
# Test the different optimal parameters found under the noisy set-up but with way 
# more samples
# Which one is the best
#-----------------------------------------------------------------------------#
res_no_noise_BO = f_testing(x_no_noise)
res_no_noise_DE = f_testing(x_no_noise_de)
res_noisy_BO = f_testing(x_noisy)
res_noisy_DE = f_testing(x_noisy_de)



