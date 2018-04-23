#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:04:21 2018

@author: fred
"""
import numpy as np
import importlib as ilib
import pdb


if __name__ == "__main__":
    import sys
    sys.path.append("../../../")
    import QuantumSimulation.Simulation.Spin.ControlledSpinOptim as cspinopt
    import QuantumSimulation.Utility.Optim.Batch as batch

else:
    from ...Simulation.Spin import ControlledSpinOptim as cspinopt
    from ...Utility.Optim import Batch as batch

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

if __name__ == "__main__":
    pass
    #batch = ControlledSpinOptimBatch()
    #batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)