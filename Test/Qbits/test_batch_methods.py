# -*- coding: utf-8 -*-
import sys, logging
sys.path.append("../../../QuantumSimulation/")
logger = logging.getLogger(__name__)
logger.setLevel('ERROR')
from QuantumSimulation.ToyModels.QB.learn_QBits import learnerQB

create_configs = True # example of generating and storing configs from a meta-config
run_configs = True # run a config generated 
read_res = True # reading some results stored
    
out_fld = '_test'

if(create_configs):
    """ take a meta config, generate all the configs and store them TESTED"""
    learnerQB.parse_and_save_meta_config('1Q1_pcw5.txt', 
                output_folder = out_fld, extra_processing = True, update_rules = True)

if(run_configs):
    batch = learnerQB(out_fld+'/config_res1.txt')
    batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)

if(read_res):
    res = learnerQB.read_res(nameFile = 'res1.txt', allPrefix = '', folderName = 'TestBatch')
    #res_bench = learnerQB.read_res('TestBenchmark/benchmark1.txt')
    