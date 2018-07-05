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
from functools import partial
import matplotlib.pylab as plt

# quasi degenerate function
def f(X, noise = 0, tilt = 0):
    X = np.array(X).reshape((len(X),1))
    res = tilt * X - np.square(np.sin(X * 2 * np.pi)) + 0.2 * np.square(np.cos(X*8*np.pi))
    return res


budget = 50
init = 5
final = int(max(5, budget/5))


myf = partial(f, noise=0, tilt=0)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 4)}]



#strat1
maxiter = budget - init 
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, acquisition_type ='EI')
myOpt.run_optimization(maxiter)
myOpt.plot_acquisition()
myOpt.plot_convergence()

print(myOpt.fx_opt)
print(myOpt.x_opt)


#Strat2
maxiter = budget - init - final
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, acquisition_type ='EI')
myOpt.run_optimization(maxiter)
X_already, Y_already = myOpt.X, myOpt.Y
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, X= X_already, Y = Y_already, acquisition_type ='LCB',exploration_weight =0.000001)
myOpt.run_optimization(5)
myOpt.plot_acquisition()
myOpt.plot_convergence()
print(myOpt.fx_opt)
print(myOpt.x_opt)



#Strat3
import copy
percentage = 0.1
init = 5
final = final = int(max(5, budget* percentage))
maxiter = budget - init - final
before_final = budget - final 
restrict = int(max(5, before_final * percentage))
def c_diff(x1, x2):
    diff = np.squeeze(x1) - np.squeeze(x2)
    return np.sqrt(np.dot(diff, diff))

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 4)}]
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, acquisition_type ='EI')
myOpt.run_optimization(maxiter)


X_already, Y_already = myOpt.X, myOpt.Y
index_best = np.argmin(myOpt.Y)
best_Y = Y_already[index_best]
best_X = X_already[index_best]
dist = [c_diff(best_X, x) for n, x in enumerate(X_already)]
index_closest = np.argsort(dist)
index_relevant = index_closest[: restrict]
X_relevant = X_already[index_relevant]
Y_relevant = Y_already[index_relevant]
new_limits = [np.min(X_relevant,0), np.max(X_relevant, 0)]
new_bounds = [copy.copy(b) for n, b in enumerate(bounds)]
for n, m in enumerate(new_bounds):
    m.update({'domain':(new_limits[0][n], new_limits[1][n])})

myOpt2 = GPyOpt.methods.BayesianOptimization(myf, new_bounds, X= X_relevant, Y = Y_relevant, acquisition_type ='EI')
myOpt2.run_optimization(final)
myOpt2.plot_acquisition()
myOpt2.plot_convergence()
print(myOpt.fx_opt)
print(myOpt.x_opt)




    

myOpt2 = GPyOpt.methods.BayesianOptimization(myf, bounds, X= X_already, Y = Y_already, acquisition_type ='LCB',exploration_weight =0.000001)
myOpt2.run_optimization(5)


#learning process
myOpt2.plot_acquisition()
myOpt2.plot_convergence()

print(myOpt.fx_opt)
print(myOpt.x_opt)



