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
# 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETERS:
#   + "gen_meta_configs" generate configfiles from a metaconfig file
#   + "run_one_config" run cspinoptim based on a config file
#   + "run_meta_config" run cspinoptim based on a metaconfigfile
#==============================================================================
if(len(sys.argv) > 4):
    print("Wrong number of args", file = sys.stderr)
else:
	type_task = sys.argv[1]
	file_input = sys.argv[2]
    if(type_task == "gen_meta_configs"):
    	# put them in Config/ XXX
    	OptBatch.parse_and_save_meta_config(file_input)

    else(type_task == "run_one_config"):
		batch = OptBatch(file_input)
		batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

    elif(type_task == "run_meta_configs"):
		batch = OptBatch.from_meta_config(file_input)
		batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False, debug = True)

#res = OptBatch.read_res(folderName = 'Output/TestBatch', allPrefix ='')





