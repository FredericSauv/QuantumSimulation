#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:15:32 2017

@author: frederic
"""
import numpy as np

class Params: 
    """Class defining the parameters of the sytem/simulation"""
    
    def __init__(self,ref_frequency, n_trap, n_int, eta=0, n_trunc = 5, 
                 hbar=1, convention=1, n_period=1, n_pointsPerPeriod = 1, n_fockmax=20, n_intState = 2): 

        # Parameters of the Q-System         
        self.freq = ref_frequency
        self.omega = self.freq * 2 * np.pi
        self.n_period = n_period
        self.n_pointsPerPeriod = n_pointsPerPeriod
        self.n_trap = n_trap # omega_trap = n_trap * omega
        self.n_int = n_int # omega_int = n_int
        self.eta = eta

        # Construction of the floquet state space (internal x motional x Fourier basis)
        self.n_fockmax = n_fockmax
        self.n_intState = n_intState
        self.n_trunc = n_trunc
        self.hbar = hbar
        self.convention = convention

        #Magnus Approximation parameters
        self.orderH = 1
        self.orderK = 0

        # Simulation time parameters
        self.T, self.tmax, self.dt, self.n_dt, self.array_time, self.slice_time, self.array_time_strobo, self.slice_time_strobo, self.slice_time_all = self.genTimeParameters()
        
        
        #Numerical parameters (mostly used to assert/check if truncatures are good enough)
        self.decimalRounding = 10
        self.maxPopLastMotional = 0.0001
        self.maxDevSumProba = 0.00001
        
        #Logging (not used yet)
        self.logging = 0
        self.fullInfo = 1

    def genTimeParameters(self, ref_frequency = None, n_period = None, n_ppp = None):
        if(ref_frequency is None):
            ref_frequency = self.freq
        if(n_period is None):
            n_period = self.n_period
        if(n_ppp is None):
            n_ppp = self.n_pointsPerPeriod       

        T = 1 / ref_frequency
        tmax = n_period * T
        dt = T / n_ppp
        n_dt = int(n_period * n_ppp)
        slice_time_all = slice(0, n_dt + 1)
        array_time_all = np.arange(0, n_dt + 1) * dt
        slice_time_strobo = slice(0, n_dt, max(n_ppp, 1))
        array_time_strobo = np.arange(0, n_dt + 1, max(n_ppp, 1)) * dt

        return T, tmax, dt, n_dt, array_time_all, slice_time_strobo, array_time_strobo, slice_time_strobo, slice_time_all



# ============================================= #
# CUSTOM SETUPS
# ============================================= #
def genSetupInter(setup = None, gauge = 'FME', ref_frequency = 10, int_trap = 2, int_internal=3, eta=0.1, n_period = 5, n_ppp = 25, n_trunc = 75, n_fockmax = 15):
    # Constants
    n_intState = 2
    hbar = 1
    conv = 1

    if (setup is None):
        nInt, nTrap = int_internal, int_trap
    else: 
        if (setup == 0):
          nInt, nTrap = 2, 2  
        elif (setup == 1):
          nInt, nTrap = 3, 2
        elif (setup == 2):
          nInt, nTrap = 11, 2
        elif (setup == 3):
          nInt, nTrap = 5, 2
        elif (setup == 4):
          nInt, nTrap = 2, 4 #Resonnant case
        else:
            raise NotImplementedError()

    # control params
    p0 = Params(ref_frequency, nTrap, nInt, eta, n_trunc, hbar, conv, n_period, n_ppp, n_fockmax, n_intState)
    
    if (gauge is None):
        p0.gauge = None
        p0.orderH = 2
        p0.orderK = 1 ####### ADD FLUCTU TERM
        
    if (gauge == 'HFE'):
        raise NotImplementedError()
        
    elif (gauge == 'VV'):
        ## SPECIAL VANVLECK
        p0.orderH = 2
        p0.orderK = 1 ####### ADD FLUCTU TERM   
        p0.gauge = 'VV'
        
    elif(gauge == 'FME'):
        p0.orderH = 2
        p0.gauge = 'FME'
        p0.orderK = 1 ####### ADD FLUCTU TERM     
    
    else:
        p0.orderH = 2
        p0.gauge = 'FME'
        p0.orderK = 1 ####### ADD FLUCTU TERM     

    return p0

def genSetupStrong(setup = None, ref_freq = 10, real_freq = 1, int_trap = 2, int_internal=3, eta=0.1, n_period = 5, n_ppp = 25, n_trunc = 75, n_fockmax = 15):
    # Constants
    n_intState = 2
    hbar = 1
    conv = 1

    if (setup is None):
        nInt, nTrap = int_internal, int_trap
    else: 
        if (setup == 0):
          nInt, nTrap = 2, 2  
        elif (setup == 1):
          nInt, nTrap = 3, 2
        elif (setup == 2):
          nInt, nTrap = 11, 2
        elif (setup == 3):
          nInt, nTrap = 5, 2
        elif (setup == 4):
          nInt, nTrap = 2, 4 #Resonnant case

    ratio = real_freq / ref_freq
    real_nInt = nInt / ratio
    real_nTrap = nTrap/ratio
          
    # control params
    p0 = Params(real_freq, real_nTrap, real_nInt, eta, n_trunc, hbar, conv, n_period, n_ppp, n_fockmax, n_intState)
    p0.orderK = 1 ####### ADD FLUCTU TERM     


    return p0

def genSetupInterVV(setup = None, ref_frequency = 10, int_trap = 2, int_internal=3, eta=0.1, n_period = 5, n_ppp = 25, n_trunc = 75, n_fockmax = 15):
    # Constants
    n_intState = 2
    hbar = 1
    conv = 1

    if (setup is None):
        nInt, nTrap = int_internal, int_trap
    else: 
        if (setup == 0):
          nInt, nTrap = 2, 2  
        elif (setup == 1):
          nInt, nTrap = 3, 2
        elif (setup == 2):
          nInt, nTrap = 11, 2
        elif (setup == 3):
          nInt, nTrap = 5, 2
        elif (setup == 4):
          nInt, nTrap = 2, 4 #Resonnant case

    # control params
    p0 = Params(ref_frequency, nTrap, nInt, eta, n_trunc, hbar, conv, n_period, n_ppp, n_fockmax, n_intState)
    
    ## SPECIAL VANVLECK
    p0.orderH = 2
    p0.orderK = 1 ####### ADD FLUCTU TERM   
    p0.gauge = 'VV'

    return p0

