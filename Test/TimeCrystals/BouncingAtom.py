#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:58:08 2018

@author: fred
"""
from scipy.integrate import ode
import numpy as np
import matplotlib.pylab as plt
import functools

class model:
    """ Atom subject to gravity interacting with a shaken mirror (hard wall)
    
    Parameters
    ---------
    m: effective mass of the system
    Lambda: Amplitude of the shaking
    
    Notations
    ---------
    h(t) is the height of the particle at time t
    
    In free space the classical dynamics is given by:
        dh(t) = p(t)/2m
        dp(t) = m Lambda Cos(omega t)
    """
    
    def __init__(self, Lambda = 0.01, m = 1, g = 1):        
        self.m = m 
        self.g = g
        self.Lambda = Lambda

    def get_omega_perfect(self, h, s):
        """ for a given h and s get omega"""
        return s * np.pi/np.sqrt(2 * h)

    def get_big_omega_perfect(self, h):
        """ for a given h get big_omega"""
        return np.pi/np.sqrt(2 * h)

    def f(self, omega, h_0 = 10.0, noise_h0 = 0.0, nb_repeat = 10, s = 40):
        """ For a given h_0 and omega how close do we get to href after one cycle
        i.e. return h(T) - href
        
        Parameters
        ----------
        h_0: the ideal initial height
        omega: driving of the miror
        
        noise_h0: noise in the preparation of the initial position 
        nb_repeat
        """
        big_omega = omega / s
        T =  2 * np.pi/ big_omega
        noise = np.random.normal(0.0, scale = noise_h0, size = nb_repeat)
        res = [np.abs(self.simulate_one_period(omega, T, h_0*(1+n))[0] - h_0) for n in noise]
        return np.average(res)
    

    def scan_h(self, delta, href, omega, s = 1.0):
        """ For a given href and omega plot |h(T) - href| for different starting
        heights (h(0) in [href - 2 delta, href+20 delta])
        """
        h0_list = np.linspace(href - 2 * delta, href + 20 * delta, 500)
        T = s * 2*np.pi/omega
        res = [self.simulate_one_period(omega, T, h0)[0] - href for h0 in h0_list]
        plt.plot(h0_list, res)
        plt.scatter(href, 0)
        
    def simulate_one_period(self, omega, T, h_0 = 10.0, p_0 = 0.0):
        """ return x and p after one period (T) for a given driving omega"""
        times = [T]
        der = functools.partial(self.derivative, omega = omega)
        evol = model.evolve_ode(der, [h_0, p_0], t0=0, times=times)
        return evol[0,-1], evol[1,-1]
    
    def derivative(self, t, X, omega):
        """ X = [x, p], X'(t) = [p(t)/m, -m * g * Lambda * cos(Omega * t) * (-1 if X[t]==0])"""
        if(X[0]<=0):
            X[1] = abs(X[1])
        dx = 1 / self.m * X[1] 
        dy = (-self.m * self.g - self.Lambda * np.cos(omega * t))
        return np.array([dx ,dy])  

    @staticmethod
    def evolve_ode(derivative, X0, t0, times, verbose= False, complex_type = False):
        """ wrap the ODE solver
        - derivative: func(t, X)
        
        """
        ## Set-up the solver
        solver = ode(derivative, jac=None)
        solver_args = {'nsteps': np.iinfo(np.int32).max, 'rtol':1e-9, 'atol':1e-9}
        solver.set_integrator('dop853', **solver_args)
        solver.set_initial_value(X0, t0)	
        	
        ## Prepare container for the output
        if complex_type:
            v = np.empty((len(X0),len(times)), dtype=np.complex128)
        else:
            v = np.empty((len(X0),len(times)), dtype=np.float64)
        
        ## Run
        for i,t in enumerate(times):
            if t == t0:
                if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
                v[...,i] = X0
                continue
        
            solver.integrate(t)
            if solver.successful():
                if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
                v[...,i] = solver._y
            else:
                raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))			
        return v




#--------------------------------#
model = model()
# scan for h around an h reference
model.scan_h(delta = 1, href = 1/2 * (3*np.pi)**(2/3), omega = np.pi**(2/3)/(3**(1/3)), s=1.0)


#
h=10.0
omega = 40 * np.pi/np.sqrt(2*h)
model.f(omega, h_0 = h, noise_h0 = 0.2)


