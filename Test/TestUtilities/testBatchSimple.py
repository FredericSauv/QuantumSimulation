#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
#sys.path.append('../../../FloquetSimulation')
sys.path.append('../../')

import FloquetSimulation.Simulation.ToyModelBatch as batch
import importlib as ilib
import pdb 

#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
ilib.reload(batch)
batchTest = batch.ToyModelBatch('inputBatch0.txt', counterName = 0, folderOut = 'testBatch')
batchTest.run_procedures(saveFreq = -1, splitRes = True)

#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
# pdb.run("batchTest = batch.ToyModelBatch('inputBatch0.txt', counterName = 0, folderOut = None)")
# pdb.run("batchTest.run_procedures(saveFreq = 0, splitRes = True)")

