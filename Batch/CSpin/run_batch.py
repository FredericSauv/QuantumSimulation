#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
sys.path.append('../../../')
from  QuantumSimulation.Simulation.Spin.ControlledSpinOptimBatch import ControlledSpinOptimBatch as OptBatch


#==============================================================================
# SETUP 1 T
#==============================================================================
if(len(sys.argv) > 2):
    print("Wrong number of args", file = sys.stderr)
else:
    nameFileInput = sys.argv[1]
    
batch = OptBatch(nameFileInput)
batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

#res = OptBatch.read_res(folderName = 'Output/TestBatch', allPrefix ='')





