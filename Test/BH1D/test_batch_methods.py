# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../")
from QuantumSimulation.Simulation.1DBH.learn_1DBH import learner1DBH

# example of generating and storing configs from a meta-config
create_configs = False 
# example of generating and storing configs from a meta-config and extra-rules
create_configs_dispatch=False
# run a config generated 
run_configs = False 
# reading some results stored
read_res = False
    

if(create_configs):
    """ take a meta config, generate all the configs and store them TESTED"""
    learner1DBH.parse_and_save_meta_config('test_meta_config.txt', output_folder = 'ttest_gen_configs')

if(create_configs_dispatch):
    """ take a meta config, generate all the configs and store them"""
    learner1DBH.parse_and_save_meta_config('test_meta_config_dispatch.txt',
                output_folder = 'test_gen_configsNM', extra_processing = True)

if(run_configs):
    batch = learner1DBH('test_gen_configsNM/config_res0.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
    
if(read_res):
    res = learner1DBH.extract_res('TestBatch/res0.txt')