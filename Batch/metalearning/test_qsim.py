#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:13:05 2019

@author: fred
"""
import q_simulator as sim
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.insert(0, '/home/fred/Desktop/backup_GPYOpt/')
import GPyOpt

DICT_MODEL = {'L':5,'Nb':5,'mu':0,'sps':None,'kblock':None,'pblock':None, 'verbose':True}
DICT_CONTROL = {'T':6.163814081224145,'nb_params':5, 'alpha':0}

# Test Control
control = sim.Control(**DICT_CONTROL)
x = np.linspace(-0.1, 5.1, 10000)
y = control(x)
plt.plot(x, y)
control.alpha = 2
y = control(x)
plt.plot(x, y)
control.alpha = 0
control.parameters = np.random.uniform(0,1, 5)
y = control(x)
plt.plot(x, y)


# Test simulator
seed = None
np.random.seed(seed)
alpha = 0 #np.random.normal(0,1)
s = sim.Simulator(DICT_CONTROL, DICT_MODEL)

# Test BO
bounds = [{'name': str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(s.nb_params)]
DICT_BO = {'domain': bounds, 'acquisition_type': 'LCB', # Selects the Expected improvement
         'initial_design_numdata': 25, 'acquisition_weight':3,'maximize':True}
myBopt = GPyOpt.methods.BayesianOptimization(f=s, **DICT_BO)
myBopt.run_optimization(250)   

# what is stored
dict_res = {'X':np.array(myBopt.X),'Y':np.array(myBopt.Y), 'x_opt':myBopt.x_opt, 
            'fx_opt':myBopt.fx_opt, 'SEED':seed, 'call_f':s._f_calls,'alpha':s.control_fun.alpha}


# for alpha = 0 good params
# x_opt = np.array([0.72887749, 0.79911432, 0.82320549, 0.95501789, 0.88583126])
# fx_opt = -0.9622391074818919

# test reading results
res0 = sim.read_results('_res/res/alpha_std1/alpha_std1/res0.txt')
print(res0['alpha'])

