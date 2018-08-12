# -*- coding: utf-8 -*-
import sys
import pdb
sys.path.append("../../../")
from  QuantumSimulation.Simulation.Qbits.learn_Qbits import learnerQB

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
    learnerQB.parse_and_save_meta_config('1Q1_pcw5.txt', 
                output_folder = 'test_gen_configs', extra_processing = True, update_rules = True)

if(run_configs):
    batch = learnerQB('test_gen_configs/config_res1.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

if(read_res):
    res = learnerQB.read_res(nameFile = 'res1.txt', allPrefix = '', folderName = 'TestBatch')
    #res_bench = learnerQB.read_res('TestBenchmark/benchmark1.txt')
    