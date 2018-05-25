#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:04:21 2018

@author: fred
"""
import numpy as np
import importlib as ilib
import pdb
from functools import partial

if __name__ == "__main__":
    import sys
    sys.path.append("../../../")
    import QuantumSimulation.Simulation.Spin.ControlledSpinOptim as cspinopt
    import QuantumSimulation.ToyModels.ControlledSpin as cs
    import QuantumSimulation.Utility.Optim.Batch as batch
    import QuantumSimulation.Utility.Helper as ut
    from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pf

else:
    from ...Simulation.Spin import ControlledSpinOptim as cspinopt
    from ...Utility.Optim import Batch as batch
    from ...Utility import Helper as ut
    from ...Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pf
    from  ...ToyModels import ControlledSpin as cs

ilib.reload(batch)
ilib.reload(cspinopt)



class ControlledSpinOptimBatch(batch.Batch):
    """ Just implements what running a procedure (based on a config) means
    """
    def __init__(self, inputFile = None, rdm_object = None, procToRun = None, debug = False):
        batch.Batch.__init__(self, inputFile, rdm_object, procToRun, debug)
    
    def run_one_procedure(self, configDico):
        """ Shortcut procToRun
        """
        configSim = configDico['paramsSim']
        configOptim = configDico['paramsOptim']
        configTesting = configDico.get('paramsTesting')
        optim = cspinopt.ToyModelOptim(configSim, configOptim, configTesting)
        _ = optim.Run(writeLogs = True)
        
        #Extract the results
        #pdb.set_trace()
        res={}
        res['optim'] = dict(optim.resOptim)
        if(configTesting is not None):
            res['testing'] = optim.resTest
            optim_fom_tmp = optim.resTest['fun']
            if hasattr(optim_fom_tmp, '__iter__'):
                optim_fom_tmp = optim_fom_tmp[0]
        else:
            optim_fom_tmp = optim.resOptim['fev_fom'][-1][1]
        res['best_fom'] = optim_fom_tmp
        return res



    def _dispatch_configs(self, dico):
        """ AdHoc dispatch rules for the fields in the config dictionnary into the 
        three dicos needed to run optimizations: paramsSimulation, paramsTesting, 
        paramsOptim
        TODO: what about _NB_RANDOM etc..
        """
        #### paramsSimulation
        if('paramsSim' in dico.keys()):
            paramsSim = dico.pop('paramsSim')
        else:
            paramsSim = self._dispatch_configs_sims(dico)
        
        #### paramsTesting         
        if ('paramsTesting' in dico.keys()):
            paramsTesting = dico.pop('paramsTesting')
        else:
            paramsTesting = self._dispatch_configs_sims(dico, prefix = 'test_')
                 
        #### paramsOptim       
        if ('paramsOptim' in dico.keys()):
            paramsOptim = dico.pop('paramsOptim')
        else:
            paramsOptim = dico
    
        dico_out = {'paramsSim': paramsSim, 'paramsTesting': paramsTesting, 'paramsOptim': paramsOptim}
        return dico_out
                
        
    def _dispatch_configs_sims(self, dico, prefix=''):     
        """ AdHoc dispatch rules for the fields in the config dictionnary into the 
        three dicos needed to run optimizations: paramsSimulation, paramsTesting, 
        paramsOptim
        """
        if(prefix != ""):
        # If prefix is not the deault one and if no key start with this 
        # prefix return None
            test_prefix = [k[:len(prefix) == prefix] for k in dico.keys()]
            if(not(np.any(test_prefix))):
                return None
   
        if ((prefix + 'noise') not in dico.keys()):
            noise_fom = dico.pop((prefix + 'noise_fom'), 'normal_0_0.00')
            noise_Ex = dico.pop((prefix + 'noise_fom'), 'normal_0_0.00')
            noise_Ez = dico.pop((prefix + 'noise_fom'), 'normal_0_0.00')
            noise = {'fom':noise_fom, 'Ex':noise_Ex, 'Ez':noise_Ez}
        else:
            noise = dico.pop((prefix + 'noise'))
    
        if ((prefix + 'overall') not in dico.keys()):
            overall_bounds = dico.pop((prefix + 'overall_bounds'), None)
            overall_constraints = dico.pop((prefix + 'overall_constraints'), None)
            overall_constraints_type = dico.pop((prefix + 'overall_constraints_type'), None)
            overall = {'bounds': overall_bounds, 'constraints':overall_constraints, 
            'constraints_type':overall_constraints_type}
        else:
            overall = dico.pop((prefix + 'overall'))
    
        if ((prefix + 'guess') not in dico.keys()):
            guess_func = dico.pop((prefix + 'guess_func'), 'constant')
            guess = {'func':guess_func}
            for k_name, val in dico.items():
                if(k_name[:6+len(prefix)] == (prefix + 'guess_')):
                    guess[k_name[6+len(prefix):]] = val                          
        else:
            guess = dico.pop((prefix +'guess'))
        
        if ((prefix +'control') not in dico.keys()):
            control_func = dico.pop(prefix +'control_func')
            control = {'func':control_func}
            for k_name, val in dico.items():
                if(k_name[:6+len(prefix)] == (prefix +'control_')):
                    control[k_name[6+len(prefix):]] = val                        
        else:
            control = dico.pop(prefix +'control')
        
        setup = dico.pop(prefix +'setup')
        tgt = dico.pop(prefix +'target_state_name')
        init = dico.pop(prefix +'init_state_name')
        T = dico.pop(prefix +'T')
        dt = dico.pop(prefix +'dt')
        fom_name = dico.pop(prefix +'fom_name')
        
        dico_sims = {'setup' : setup, 'target_state_name' : tgt, 
        'init_state_name':init, 'overall': overall, 'guess': guess, 
        'control': control, 'dt': dt, 'T': T, 'fom_name': fom_name,
        'noise': noise}
    
        return dico_sims

# ---------------------------
#   COLLECTING  / PACKAGING RESULTS
# Ad-hoc functions
# ---------------------------
    
    @classmethod
    def process_list_res(cls, listRes, name_listRes = None, printing = False):
        """ Take a list of res and extract interesting informations packaged as
        a dico:
            + evol_fom_stats x
            + evol_ideal_fom_stats x
            + 
            + optimal_function_stats 
            + optimal_fom_stats x
            + optimal_fun1_stats x
        """
        #Path (in the nested structure of the res)
        path_optim_fev_fom = ['optim', 'fev_fom']
        path_optim_func_fom = ['optim', 'func_fom']
        path_test_fom = ['best_fom']
        path_test_fun0 = ['testing', 'fun', 0]
        path_test_fun1 = ['testing', 'fun', 1]
        # path_test_namefom = ['config', 'paramsTesting', 'fom_name']
        path_test_t = ['config', 'paramsTesting', 'T']

        # evol of fom (measured) over number of func evaluation
        optim_fev_fom = [np.array(ut.extract_from_nested(run, path_optim_fev_fom)) for run in listRes]
        evol_fom_stats = ut.merge_and_stats_TS(optim_fev_fom)
        
        # evol of fom (ideal: testing environment) over number of func evaluation
        ideal_fev_foms = ideal_learning_from_list_res(listRes)
        evol_ideal_fom = ideal_fev_foms[0]
        evol_ideal_fom_stats = ut.merge_and_stats_TS(evol_ideal_fom)
        evol_ideal_fom_1 = ideal_fev_foms[1]
        evol_ideal_fom_1_stats = ut.merge_and_stats_TS(evol_ideal_fom_1)


        # pdb.set_trace()
        #stats on the optimal function
        list_T = [ut.extract_from_nested(run, path_test_t) for run in listRes]
        list_array_T = [np.arange(0, T, T/100) for T in list_T]
        list_optim_func = [pf.Repr2Fun(ut.extract_from_nested(run, path_optim_func_fom)[-1][0]) for run in listRes]
        list_optim_func_TS = [np.c_[list_array_T[n], func(list_array_T[n])] for n, func in enumerate(list_optim_func)]
        optimal_function_stats = ut.merge_and_stats_TS(list_optim_func_TS)

        # Based on the final optimal control found get stats for the FoM
        # avg, mini, maxi, std, avg_pstd, avg_mstd
        res_test_fom = [np.array(ut.extract_from_nested(run, path_test_fom)) for run in listRes]
        res_test_fun0 = [np.array(ut.extract_from_nested(run, path_test_fun0)) for run in listRes]
        res_test_fun1 = [np.array(ut.extract_from_nested(run, path_test_fun1)) for run in listRes]
        opt_fom_stats = ut.get_stats(res_test_fom)
        opt_fun0_stats = ut.get_stats(res_test_fun0)
        opt_fun1_stats = ut.get_stats(res_test_fun1)

        if(printing):
            print('TESTED RES:FOM (avg, min, max)')
            print(opt_fom_stats[:3])
            print('TESTED RES:FUN0 (avg, min, max)')
            print(opt_fun0_stats[:3])
            print('TESTED RES:FUN1 (avg, min, max)')
            print(opt_fun1_stats[:3])

        dico_res = {}
        dico_res['evol_fom'] = evol_fom_stats
        dico_res['evol_ideal_fom'] = evol_ideal_fom_stats
        dico_res['evol_ideal_fidelity'] = evol_ideal_fom_1_stats
        dico_res['opt_control'] = optimal_function_stats
        dico_res['best_fom'] = opt_fom_stats
        dico_res['test_fom'] = opt_fun0_stats
        dico_res['test_fidelity'] = opt_fun1_stats 

        return dico_res
# ---------------------------
# Re-run simulations based on control functions
# ---------------------------
def runSimul(dicoArgs, control):
    """ from a dico containing the parameters of the simulations and a control 
        function get the results of the simulation
    """
    setup = dicoArgs['setup']
    init = dicoArgs['init_state_name']
    final = dicoArgs['target_state_name']
    T = dicoArgs['T']
    dt = dicoArgs['dt']
    noise = dicoArgs['noise']
    method = dicoArgs['method_simul']
    fom = dicoArgs['fom_name']
    if(ut.is_str(control)):
        control = pf.Repr2Fun(control)
    
    sim_tmp = cs.ControlledSpin(setup = setup, state_init = init ,controlFun = control,  
                state_target = final ,T = T, dt = dt, noise = noise)
    res = sim_tmp.Simulate(method = method, fom = fom)   
    return res

def ideal_learning_from_res(run):
    """ Compute the fom (under testing conditions) for the different functions 
    found along optimization
    """
    dico_testing = ut.extract_from_nested(run, ['config', 'paramsTesting'])
    func_fom = ut.extract_from_nested(run, ['optim', 'func_fom'])
    fev_fom = ut.extract_from_nested(run, ['optim', 'fev_fom'])
    nb_fom = len(dico_testing['fom_name'])
    run = partial(runSimul, dicoArgs = dico_testing)
    tmp = np.array([[fev_fom[nb][0]] + run(control = item[0]) for nb, item in enumerate(func_fom)])
    res = [tmp[:,[0, n + 1]] for n in range(nb_fom)]
    return res
    

def ideal_learning_from_list_res(simul):
    tmp = [ideal_learning_from_res(run) for run in simul]
    nb_run = len(simul)
    nb_res = len(tmp[0])
    res = [[tmp[r][n] for r in range(nb_run)] for n in range(nb_res)]
    return res




        
        

if __name__ == "__main__":
    pass
    #batch = ControlledSpinOptimBatch()
    #batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)