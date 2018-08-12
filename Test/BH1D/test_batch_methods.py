# -*- coding: utf-8 -*-
import sys
import pdb
sys.path.append("../../../")
from QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH

# example of generating and storing configs from a meta-config
create_configs = True 
# example of generating and storing configs from a meta-config and extra-rules
create_configs_custom=False
# run a config generated 
run_configs = False
# reading some results stored
read_res = True
    

if(create_configs):
    """ take a meta config, generate all the configs and store them"""
    learner1DBH.parse_and_save_meta_config('LN5_pcw5.txt', 
                    output_folder = 'test_gen_configs', extra_processing = True, update_rules = True)

if(run_configs):
    batch = learner1DBH('test_gen_configs/config_res1.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

    
if(read_res):
    res = learner1DBH.read_res('TestBatch/res10.txt')
    res_bench = learner1DBH.read_res('TestBenchmark/benchmark1.txt')