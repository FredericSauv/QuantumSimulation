#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:15:04 2018

@author: frederic
"""

import GPy
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt

##================##
# GPY: Generate sample paths from different kernels
##================##
input_dim = 1
nb_paths = 4
nb_points = 500
X = np.linspace(0.0 ,1.0, nb_points)
X = X[:,np.newaxis]

length_scales = [0.01, 0.1, 1]

matern = GPy.kern.Matern52(input_dim=input_dim)
for l in length_scales:
    matern.lengthscale = l
    CXX = matern.K(X, X)
    path = np.random.multivariate_normal(np.zeros(nb_points), CXX, nb_paths)
    fig = plt.figure()
    for p in path:
        plt.plot(X, p)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        

kernels = [GPy.kern.Matern52(input_dim=input_dim), GPy.kern.Matern32(input_dim=input_dim), GPy.kern.RBF(input_dim=input_dim)]
for k in kernels:
    k.lengthscale = 0.1
    CXX = k.K(X, X)
    path = np.random.multivariate_normal(np.zeros(nb_points), CXX, nb_paths)
    fig = plt.figure()
    for p in path:
        plt.plot(X, p)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        

##================##
# GPY: fit the hyperparameters of a GP
##================##
# define the cost function
def cost_func(X, noise = None):
    if hasattr(X, '__iter__'):
        res = np.array([cost_func(x, noise) for x in X])
    else:
        res = -np.cos(np.pi*X) +np.sin(4*np.pi*X)
        if (noise is not None):
            res += np.random.randn() * noise

    return  res

X = np.linspace(0.0, 1.0, nb_points)
Y_ideal = cost_func(X)
Y_noise = cost_func(X, noise =0.1)
fig = plt.figure()
plt.plot(X, Y_ideal, label='f')
plt.plot(X, Y_noise, label='f_noisy(sig=0.1)')
plt.xlabel('x')
plt.ylabel('f(x)')

#Draw some random points, fit the model and visualize it
nb_points_init = 10
X_obs = np.random.uniform(0.0, 1.0, (nb_points_init, 1))
Y_obs = cost_func(X_obs)
k = GPy.kern.Matern52(input_dim=input_dim)
m = GPy.models.GPRegression(X_obs, Y_obs, k)

#Plot predictions based on the observations
m.plot()
plt.plot(X, Y_ideal,'r--', label = 'f')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
print(m)

#Fit the hyperparameters but first impose some consraints on the hyperparameters
m.constrain_positive()
m.optimize()
m.plot()
plt.plot(X, Y_ideal,'r--', label = 'f')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
print(m)
#Next point
    
# after 10 




##================##
# GPyOpt: Run a BO and illustrate
# How to add plots to a graph
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


#Next
myOpt.run_optimization(1)
fig = myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

#5 Next
myOpt.run_optimization(5)
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

#5 Next
myOpt.run_optimization(5)
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

#First optim with noise
myf =my_f(sd=0.2)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]
max_iter = 15
myOpt = GPyOpt.methods.BayesianOptimization(myf,bounds)

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


#Next
myOpt.run_optimization(1)
fig = myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

#5 Next
myOpt.run_optimization(5)
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

#5 Next
myOpt.run_optimization(5)
myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()
print(myOpt.model.model)

myOpt.plot_convergence()

##================##
# To tune:
# Nb points init 
# Kernel
# acq
# acquisition


##================##

myBopt = GPyOpt.methods.BayesianOptimization(f=myf,            # function to optimize       
                                             domain=bounds,        # box-constraints of the problem
                                             acquisition_type='EI',
                                             exact_feval = True) # Selects the Expected improvement


# Run the optimization
max_iter = 15     # evaluation budget
max_time = 60     # time budget 
eps      = 10e-6  # Minimum allows distance between the las two observations

myBopt.run_optimization(max_iter, max_time, eps)   

myOpt.plot_acquisition()
shift = np.average(myOpt.Y)
sd = np.std(myOpt.Y)
Y_tmp = (Y_ideal - shift)/sd
plt.plot(X, Y_tmp, 'r--', label = 'f')
plt.legend()

myOpt.plot

