#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:58:08 2018

@author: fred
"""
from scipy.integrate import ode
import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)
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

    def simulate_noisy_h0(self, omega, h_ref = 10.0, noise_h0 = 0.1, nb_repeat = 10, 
                          s = 40, nb_period = 1, verbose = False, return_to_init = False):
        """ For a given h_0 and omega how close do we get to href after one cycle
        i.e. return h(T) - href
        
        Parameters
        ----------
        h_ref: the ideal initial height
        omega: driving of the miror
        noise_h0: noise in the preparation of the initial position 
                  x0_i ~ N(h_ref, noise_h0^2)
        nb_repeat: number of repetition ofthe experiment
        nb_period: number of period (T) after which we measure the height of the
                  atom - with T = s * 2 Pi / omega
        s:
        verbose: <boolean>
                if True print the result
        return_to_init: <boolean>
                instead of using h_ref as the reference we use the real h_0 
                (i.e. h_ref + some noise)
        Output
        ------
            res = 1/nb_repeat * sum |h(nb_period * T) - h_ref|
            shape = (len(nb_period), )
        
        """
        big_omega = omega / s
        T_ref =  2 * np.pi/ big_omega
        T = nb_period * T_ref if(np.ndim(nb_period) == 0) else T_ref * np.array(nb_period)
        noise = np.random.normal(0.0, scale = noise_h0, size = nb_repeat)
        if(return_to_init):
            res = [np.abs(self.simulate_to_t(omega, T, h_ref*(1+n))[0] - h_ref*(1+n)) for n in noise]
        else:
            res = [np.abs(self.simulate_to_t(omega, T, h_ref*(1+n))[0] - h_ref) for n in noise]
        avg = np.average(res, axis = 0)
        if(verbose):
            print(avg)
        return avg
    

    def scan_h(self, delta, href, omega, s = 1.0):
        """ Replicate Arek's Mathematica. For a given h_ref and omega plot 
        |h(T) - href| for different starting heights (h(0) in 
        [href - 2 delta, href+20 delta])
        """
        h0_list = np.linspace(href - 2 * delta, href + 20 * delta, 500)
        T = s * 2*np.pi/omega
        res = [self.simulate_to_t(omega, T, h0)[0] - href for h0 in h0_list]
        plt.plot(h0_list, res)
        plt.scatter(href, 0)
        
    def simulate_to_t(self, omega, t, h_0 = 10.0, p_0 = 0.0):
        """ return [X(t_0), ... X(t_n)] where X(t) = [x(t), p(t)]
        shape = (2 x len(t))
        """
        times = [t] if np.ndim(t) == 0 else t
        der = functools.partial(self.derivative, omega = omega)
        evol = model.evolve_ode(der, [h_0, p_0], t0=0, times=times)
        return evol
    
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
        X state of the system (i.e. here [x, p])
        Parameters
        ----------
            + derivative: func(t, X) -> X'(t)
            + X0: an array with the initial conditon
            + t0: starting time
            + times : List of times for which we want state o the system
            + verbose: print some info about the solver
            + complex_type: allow for complex values
        Output
        ------
        State of the system over the requested times shape = (size(X), time)
        i.e. transpose([X(times[0]), ..., X(times[-1])])
        
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


if __name__ == '__main__':
    mod = model()
    #Arek's Mathematica
    mod.scan_h(delta = 1, href = 1/2 * (3*np.pi)**(2/3), omega = np.pi**(2/3)/(3**(1/3)), s=1.0)


    # Look at the impact of changing omega when there is nois ein the initial
    # height (h_ref). Results are obtained for s = 40 (still it doesn't seem
    # that changing it has impact)
    h_ref = 328.85
    s = 40
    print(mod.get_omega_perfect(328.85, 40))
    periods = [1, 6]
    omega_range = np.linspace(0.8, 1.2, 200)
    res_noise_10trials = np.zeros((len(omega_range), len(periods)))
    res_noise_100trials = np.zeros((len(omega_range), len(periods)))
    res_no_noise = np.zeros((len(omega_range), len(periods)))

    omega_ref = s * np.pi/np.sqrt(2 * h_ref)
    for n_om, om in enumerate(omega_ref * omega_range):
        res_noise_10trials[n_om, :] = mod.simulate_noisy_h0(om, h_ref, noise_h0 = 0.2, nb_repeat = 10, s = s, nb_period = periods)
        res_noise_100trials[n_om, :] = mod.simulate_noisy_h0(om, h_ref, noise_h0 = 0.2, nb_repeat = 100, s = s, nb_period = periods)
        res_no_noise[n_om, :] = mod.simulate_noisy_h0(om, h_ref, noise_h0 = 0, nb_repeat = 1, s = s, nb_period = periods)
    
    plt.plot(omega_range,res_noise_10trials[:,0]/h_ref, label = 'noise:0.2, 10trials')
    plt.plot(omega_range,res_noise_100trials[:,0]/h_ref, label = 'noise:0.2, 100trials')
    plt.plot(omega_range,res_no_noise[:,0]/h_ref, label = 'no noise')
    plt.legend()
    plt.legend(fontsize = 12)
    plt.xlabel(r"$\omega / \omega_{ref}$", size = 18)
    plt.ylabel(r"$|h(T)-h_{ref}|/ h_{ref}$", size = 18)
    plt.savefig('noise_0_2_h0_329_lambda_0_01.pdf')


    plt.plot(omega_range,res_noise_10trials[:,1]/h_ref, label = 'noise:0.2, 10trials')
    plt.plot(omega_range,res_noise_100trials[:,1]/h_ref, label = 'noise:0.2, 100trials')
    plt.plot(omega_range,res_no_noise[:,1]/h_ref, label = 'no noise')
    plt.legend(fontsize = 12)
    plt.xlabel(r"$\omega / \omega_{ref}$", size = 18)
    plt.ylabel(r"$|h(6T)-h_{ref}|/ h_{ref}$", size = 18)
    plt.savefig('noise_0_2_h0_329_lambda_0_01_6T.pdf',bbox_inches='tight')