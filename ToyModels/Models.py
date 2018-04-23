#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import numpy as np
import matplotlib.pylab as plt
import pdb
import importlib as ilib
from ..Utility.Optim import RandomGenerator as rdm


ilib.reload(rdm)

#TODO: Should it inherit from a state space??

#==============================================================================
#                   Models
#    PURPOSE:
#        Abstract model class (to enforce structure)
#        
#    MAIN METHODS:
#        - init :
#        - changeOfPicture :  
#        - Simulate: 
#        - Evolution:
#
#   MAIN ATTRIBUTES
#        - ss: a state from spaceQuantumSimulation.Utility.StateSpace
#        - initState:
#        - targetState:     
#        - state_t:
#==============================================================================
class Models:
    def __init__(self, **args):
        """  
        Purpose:
            Store params of the toymodel
            generate a StateSpace associated
        """
        self._ss = None
        self.state_init = None
        self.state_tgt = None
        self.state_t = None
        self.pop_t = None
        self.setup = None
        self.time_array = None

        if(args.get('noise') is not None):
            self._noise = args.get('noise')
            self._rdmgen = rdm.RandomGenerator(self._noise.get('seed'))
            
            #will probably change
            #pdb.set_trace()
            self._noise_func = {k:self._rdmgen.gen_rdmfunc_from_string(v) for 
                                k, v in self._noise.items() if k != 'seed' }
            
        else:
            self._rdmgen = rdm.RandomGenerator()
            
# --------------------------------------------------------------------------- #
#   SIMULATION 
#       Evolves the init state and performs other actions (typically computing a figure 
#       of Merit or other statistic)
# --------------------------------------------------------------------------- #
    def Simulate(self, **args):
        """
        Purpose:
            Main entry point to simulate the system
            if fom is not None, will return it, if not the state_t of the system
        """
        raise NotImplementedError()            
        
       
# --------------------------------------------------------------------------- #
#   Evolution
#   Evolve the initial state
# --------------------------------------------------------------------------- #
    def Evolution(self, **args):
        """  
        Purpose:
            Evolve the init_state according to the relevant method and store if
            required
        Output:
            state_t, 
        """
        raise NotImplementedError()        


# --------------------------------------------------------------------------- #
#   Updates
# --------------------------------------------------------------------------- #
    def UpdateStateInit(self, init = None):
        self.state_init = self.GetState(init)


    def UpdateStateTarget(self, tgt = None):
        self.state_tgt = self.GetState(tgt)

    def randomize(self):
        """ Randomize what has to be randomized
        """
        raise NotImplementedError()

# --------------------------------------------------------------------------- #
#   Extras: get the methods from the underlying statespace
# --------------------------------------------------------------------------- #
    def GenPopulation(self):
        if(self.state_t is not None):
            self.pop_t = self._ss.probaFromState(self.state_t)
        else:
            raise NotImplementedError() # Not the right type of error
        
    def PrintPopulation(self, state_t = None, time_array = None, legend = True):
        if(state_t is None):
            if(self.pop_t is None):
                self.GenPopulation()
            pop = self.pop_t
            time = self.time_array
        else:
            pop = self._ss.probaFromState(state_t)
            time = time_array
        
        fig, ax = plt.subplots()
        if((time is not None) and (len(time)==pop.shape[0])):
            for i in range(pop.shape[1]):
                ax.plot(time, pop[:, i], label = str(i))
        else:
            for i in range(pop.shape[1]):
                ax.plot(pop[:, i], label = str(i))
        if(legend):
            ax.legend()
            
    def changeOfPicture(self, state_A, picture_A = 'initial', picture_B = 'initial'):
        return self._ss.changeOfBasis(state_A, picture_A, picture_B)
  
    
    def GetState(self, state = None):
        """
        Purpose:
            Get some specific state from the State Space 
        """
        if(state is None):
            res = None

        elif(isinstance(state, str)):
            res = self._ss.GetStateByName(state)

        else:
            res = np.array(state)

        return res






