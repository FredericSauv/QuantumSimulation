# -*- coding: utf-8 -*-
import sys
import pdb
sys.path.append("../../../")
from QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH

# example of generating and storing configs from a meta-config
create_configs = False 
# example of generating and storing configs from a meta-config and extra-rules
create_configs_custom=True
# run a config generated 
run_configs = True
# reading some results stored
read_res = True
    

if(create_configs):
    """ take a meta config, generate all the configs and store them TESTED"""
    learner1DBH.parse_and_save_meta_config('test_meta_config.txt', output_folder = 'test_gen_configs')

if(create_configs_custom):
    """ take a meta config, generate all the configs and store them"""
    learner1DBH.parse_and_save_meta_config('test_meta_config_extraprocessing.txt',output_folder = 'test_gen_configs_extra', extra_processing = True, update_rules = True)
    learner1DBH.parse_and_save_meta_config('test_meta_config_extraprocessing_testlinear.txt',output_folder = 'test_gen_configs_extra_benchmark', extra_processing = True, update_rules = True)

if(run_configs):
    batch = learner1DBH('test_gen_configs_extra/config_res10.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
    batch = learner1DBH('test_gen_configs_extra_benchmark/config_benchmark1.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
    
if(read_res):
    res = learner1DBH.extract_one_res('TestBatch/res10.txt')
    res_bench = learner1DBH.extract_one_res('TestBenchmark/benchmark1.txt')