#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
sys.path.append('../../../../')
from  QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH


#==============================================================================
# 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETER:
#   + "gen_configs" generate config files from a metaconfig file
#   + "gen_configs_custom" generate config files from a metaconfig file (with extra_processing)
#   + "run_one_config" run cspinoptim based on a config file
#==============================================================================
if(len(sys.argv) > 4 or len(sys.argv) < 3):
    print("Wrong number of args", file = sys.stderr)
else:
    type_task = sys.argv[1]
    file_input = sys.argv[2]
    if(type_task == "gen_configs"):
        if(len(sys.argv) == 4):
            output_f = str(sys.argv[3])
        else:
            output_f = 'Config'
        learner1DBH.parse_and_save_meta_config(file_input, output_folder = output_f)

    elif(type_task == "gen_configs_custom"):
        if(len(sys.argv) == 4):
            output_f = str(sys.argv[3])
        else:
            output_f = 'Config'
        learner1DBH.parse_and_save_meta_config(file_input, output_folder = output_f
                                               , extra_processing = True)

    elif(type_task == "run_one_config"):
        batch = learner1DBH(file_input)
        batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

#    elif(type_task == "run_meta_configs"):
#        # Instantiate Batch with a metaconfig file
#        batch = learner1DBH.from_meta_config(file_input)
#        batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False, debug = True)
#        
    else:
        print("first argument not recognized", file = sys.stderr)
#res = OptBatch.read_res(folderName = 'Output/TestBatch', allPrefix ='')





