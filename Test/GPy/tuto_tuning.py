#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:47:07 2018

@author: fred
"""
import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')

import GPyOpt
import numpy as np
import matplotlib.pyplot as plt

##================##
# GPyOpt: Run a BO and illustrate
# How to add plots to a graph
# Goal: tune the kernel
# Goal: tune nb of init points
# goal: tune finding 
# goal: tune acqu function
##================##
class my_f(GPyOpt.objective_examples.experiments1d.function1d):
    def __init__(self,sd=None):
        self.input_dim = 1		
        if sd==None: self.sd = 0
        else: self.sd=sd
        #self.min = 0.78 		## approx
        #self.fmin = -6 			## approx
        self.bounds = [(0,1)]

    def f(self, X):
        return self(X)

    def __call__(self,X):
        X = X.reshape((len(X),1))
        n = X.shape[0]
        fval = -np.cos(np.pi*X) +np.sin(4*np.pi*X)
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return fval.reshape(n,1) + noise


#First optim
myf =my_f()
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]
max_iter = 15


#


myOpt = GPyOpt.methods.BayesianOptimization(myf,bounds, exact_feval = True)

#Init (how to select how many we want)
myOpt.run_optimization()
fig = plt.figure()
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)
