#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import logging
logging.basicConfig(level=logging.ERROR,filename='log.log')
logger = logging.getLogger(__name__)
import sys
sys.path.append('../../../QuantumSimulation')
from  QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH


#==============================================================================
# 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETER:
#   + "gen_configs" generate config files from a metaconfig file
#   + "gen_configs_custom" generate config files from a metaconfig file (with extra_processing)
#   + "run_one_config" run cspinoptim based on a config file
#==============================================================================
if __name__ == '__main__':
    if(len(sys.argv) > 4 or len(sys.argv) < 3):
        logger.error("Wrong number of args")
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
                                        , extra_processing = True, update_rules = True)
    
        elif(type_task == "run_one_config"):
            batch = learner1DBH(file_input)
            batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
    

        else:
            logger.error("first argument not recognized")
    #res = OptBatch.read_res(folderName = 'Output/TestBatch', allPrefix ='')
    

    testing = False 
    if(testing):
        learner1DBH.parse_and_save_meta_config(input_file = '_tmp/_configs/config_res330.txt', output_folder = '_tmp/_configs/_tmp', update_rules = True)
        #batch = BatchFS(['_tmp/_configs/_mo5gradient/config_res'+str(i)+'.txt' for i in range(100)])
        batch = learner1DBH('_tmp/_configs/config_res330.txt')
        batch.run_procedures(save_freq = 1)



