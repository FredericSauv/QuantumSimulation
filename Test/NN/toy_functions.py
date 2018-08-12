#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:38:44 2018

@author: fred
"""
import numpy as np
import matplotlib.pylab as plt

class ToyFunctions():
    def __init__(self, noise_X= 0, noise_Y = 0, dim_X = 1, dim_y = 1, bounds = None):
        self.noise_X = noise_X
        self.noise_Y = noise_Y
        self.dim_X = dim_X
        self.noise_X = noise_X
        self.bounds = bounds
        self._check_integrity()
        
    def __call__(self, X, **args):
        X = self.reshape_X
        X = self._add_noise_X(X)
        Y = np.array([self.fun(x, **args) for x in X])
        Y = self._add_noise_Y(Y)
        return Y
    
    def fun(self, X, **args):
        """ take a 1D input, vectorization is implemented in __call__() """
        raise NotImplementedError()

    def _add_noise_X(self, X):
        if self.noise_X > 0:
            X = X + np.random.normal(loc=0, scale = self.noise_X, size = X.shape)
        return X
    
    def _add_noise_Y(self, Y):
        if self.noise_Y > 0:
            Y = Y + np.random.normal(loc=0, scale = self.noise_Y, size = Y.shape)
        return Y
    
    def reshape_X(self, X):
        dim = len(np.shape(X))
        if(dim == 1):
            assert len(X) == self.dim_X, 'pb shape of X, dim = 1'
            X = np.array(X).reshape((1, self.dimX))
        elif(dim == 2):
            assert np.shape(X) == self.dim_X, 'pb shape of X, dim = 2'
            X = np.array(X).reshape((len(X), self.dimX))
        else:
            assert True, 'pb shape of X: dim > 2'
        return X

        
    def gen_random_data(self, N = 10):
        X = np.transpose([np.random.uniform(low = b[0], high = b[1], size = N) for b in self.bounds])
        Y = self(X)
        return (X, Y)
    
    def check_integrity(self):
        pass
    
    def plot_func(self, **args_plot):
        if(self.dim_X == 1):
            X = np.linspace(self.bounds[0][0], self.bounds[0][1], 10000)
            Y = self(X)
            plt.plot(X, Y)
            
        elif(self.dim_X == 2):
            x1 = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
            x2 = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
            X = np.meshgrid(x, y)
            Y = self(X)
            plt.plot(X, Y)
        
        
class SHC(ToyFunctions):
    def __init__(self, noise_X= 0, noise_Y = 0):
        ToyFunctions.__init__(self, noise_X= noise_X, noise_Y = noise_Y, 
                              dim_X = 2, dim_y = 1, bounds = [(-3, 3), (-2, 2)])
        self.minima_X = [[0.0898, -0.7126], [-0.0898, 0.7126]]
        self.minima_Y = self(self.minima_X)
    
    def fun(self, X):
        x1 = X[0]
        x2 = X[1]
        Y = (4 - 2.1 * x1**2 + x1**4) * x1**2 + x1 * x2 + (4 * x2**2 - 4) * x2**2 
        return Y
    
    
class FF(ToyFunctions):
    """Highly tunable function with local/global maximas"""
    def __init__(self, noise_X= 0, noise_Y = 0, dim_X = 2):
        ToyFunctions.__init__(self, noise_X= noise_X, noise_Y = noise_Y, 
                              dim_X = dim_X, dim_y = 1, bounds = [(-1, 1) for _ in range(dim_X)])
        self.minima_X = [[0.0898, -0.7126], [-0.0898, 0.7126]]
        self.minima_Y = self(self.minima_X)
    
    def fun(self, X):
        Y = 1

        return Y
    
    
    def genFunction(noise = 0, tilt = 0, degen = 1, dim = 1, local_degen = 3, val_local = 0.5):
    def f(X):
        """generate f(X=[x1,.., xd]) = f(x1) x ... x f(x2)
        with f(x) = sin^2((x-0.1)*degen*pi) + cos^2((x-0.1) * 3 * degen * pi) + tilt * x 
        
        X = ((1+0.2)/2degen,...,1/2degen) is a minimum with f(x) = (-1+titl/2degen)^d
        there is degen^d (quasi) degenerate mimimas and ((local_degen-1) * degen) ^d local minimas
        """
        X = np.array(X).reshape((len(X),dim))
        freq_ppal = np.pi * degen
        freq_scnd = local_degen * freq_ppal
        res = 1
        for i in range(dim):
            XX = X[:,i] + 0.1 /degen 
            res *= tilt * XX - np.square(np.sin(XX * freq_ppal)) + val_local * np.square(np.cos(XX*freq_scnd))
        noise_to_add = np.random.normal(0,noise, len(X))
        return res + noise_to_add

    minimas = [[(0.4 + n) / (degen) for n in range(int(degen))] for d in range(dim)]
    return f, minimas
