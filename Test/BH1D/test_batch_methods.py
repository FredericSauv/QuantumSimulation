# -*- coding: utf-8 -*-
import logging
logging.basicConfig(filename='log.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, pdb
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH



create_configs = True # example of generating and storing configs from a meta-config
run_configs = True # run a config generated 
read_res = True # reading some results stored
    
folder_out = '_test'

if(create_configs):
    """ take a meta config, generate all the configs and store them"""
    learner1DBH.parse_and_save_meta_config('test_meta_config.txt', output_folder = folder_out, extra_processing = True, update_rules = True)

if(run_configs):
    batch = learner1DBH(folder_out + '/config_res1.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

    
if(read_res):
    res = learner1DBH.read_res(folder_out+'/res10.txt')