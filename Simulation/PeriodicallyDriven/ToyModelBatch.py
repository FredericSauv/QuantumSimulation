#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:41 2018

@author: fred
"""

#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the Simultaion.lightSimulation class)
#  
#
#
#
#============================================================================== 

from ...Utility import Helper as ut
from ...Utility.Optim import Batch as batch
from ...Utility.Optim import Optim as opt
from . import ToyModelOptim as co

import pdb


class ToyModelBatch(batch.Batch):
    """ 
    Purpose:
        Implementation of the abstract batch class for a toyModel Optimizer 
        (which could act an optimizer or simply as a simple simulator)
    
    ThinkAbout:
        Default values??
    
    """
 
    # Build list of keys belonging to the Optim / Simulation / Testing
    # They are all put together in the conf dico (cf. procToRun method)
    # used in order to dispatch entries in the config file to the right parameters
    # objects
    NAME_PARAMS_SIM, NAME_PARAMS_OPTIM = [], []
    NAME_PARAMS_SIM += list(co.ToyModelOptim.LIST_PARAMS_SIM_NEEDED)
    NAME_PARAMS_SIM += list(co.ToyModelOptim.LIST_PARAMS_SIM_OPT)
    NAME_PARAMS_OPTIM += list(opt.AbstractOptim.LIST_PARAMS_OPTIM_NEEDED)
    NAME_PARAMS_OPTIM += list(opt.AbstractOptim.LIST_PARAMS_OPTIM_OPT)
    NAME_PARAMS_TESTING = ['test_' + str(it) for it in NAME_PARAMS_SIM]    

    def __init__(self, inputFile = None, prefixOut = 'res_', folderOut = None, counterName = None):
        batch.Batch.__init__(self, inputFile, prefixOut, folderOut, counterName)
        
        
# ---------------------------
# Implementation of the abstract method
# ---------------------------
    def procToRun(self, config):
        """
        Purpose:
            built the optimizer to run base on the conf dico
        """

        
        # Split the parameters
        paramsSim, paramsOptim, paramsTest = {}, {}, {}
        
        for k, v in config.items():
            v = ut.recastString(v)
                
            if(k in self.NAME_PARAMS_SIM):
                paramsSim[k] = v
            elif(k in self.NAME_PARAMS_OPTIM):
                paramsOptim[k] = v
            elif(k[:5] == "test_"):
                kTmp = ut.removeFromString(k, "test_")
                paramsTest[kTmp] = v
            else:
                print(k + ' not recognized')
        
        # Initialize the Optimizer
        optim_tmp = co.ToyModelOptim(paramsSim, paramsOptim, paramsTest)
        res = optim_tmp.run()
        
        return res

# -----------------------------------------------------------------------------
#                   Auxillary functions
# -----------------------------------------------------------------------------





