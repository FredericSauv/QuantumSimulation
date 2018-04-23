#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
break
import sys
sys.path.append('../../../../')
from  QuantumSimulation.Simulation.Spin.ControlledSpinOptimBatch import ControlledSpinOptimBatch as OptBatch
from QuantumSimulation.Utility.Optim import Batch as b
from QuantumSimulation.Utility import Helper as ut
import importlib as ilib
import pdb


#==============================================================================
# SETUP 1 T
#==============================================================================
batch = OptBatch('Input/test1.txt')
batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

res = OptBatch.read_res(folderName = 'Output', allPrefix ='')

ilib(b)

#import numpy as np
#test_long= {'testing':{'nfev':1,'nit':1,'x':list([-1.57745461,2.98343068,0.42620855,
#  0.54283194,0.7025616,3.53129886,1.43863438,-0.39717741,-2.18991573,-0.61029042,0.22614927
#  ,-0.01802028,-1.67273028,-2.15191426,-1.91033289])}}
#test_short = {'testing':{'a', 'b'}}                                                             
#
#
#ilib.reload(ut)           
#ut.dico_to_text_rep(test_short, 'test_short.txt')
#test_retrieve = ut.file_to_dico('test_short.txt')
#pdb.run("test_retrieve = ut.file_to_dico('test_short.txt')")
#
#import numpy as np
#ut.dico_to_text_rep(test_long_array, 'test_long.txt')
#test_retrieve = ut.file_to_dico('test_long.txt')
#pdb.run("test_retrieve = ut.file_to_dico('test_write.txt')")
#
#from ast import literal_eval as astev
#
