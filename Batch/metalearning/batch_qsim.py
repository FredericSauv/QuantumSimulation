# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""

import logging, sys, pdb 
logger = logging.getLogger(__name__)
import numpy as np
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
else:
    sys.path.append('/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/QuantumSimulation/')
from QuantumSimulation.Utility.Optim.batch_base import BatchBaseParamControl
import q_simulator as sim



class BatchQsim(BatchBaseParamControl):
    """ Warp q_simulator to be run on the cluster """

    def run_one_procedure(self, config):
        """ Define what one run should do """        
        config_model = config['model']
        config_control = config['control']
        config_bo = config['bo']
        seed = config.get('_RDM_SEED', None)
        if(config_model.get('debug', False)): pdb.set_trace()

        #generate randomly alpha
        np.random.seed(seed)
        alpha = config_control.get('alpha', None)
        if  alpha is None:
            alpha_std = config_control.get('alpha_std',1)
            alpha = np.random.normal(0, alpha_std)        
            config_control['alpha'] = alpha

        #setup simulator
        s = sim.Simulator(config_control, config_model)
        nb_params = s.nb_params

        #setup bo
        bounds = [{'name': str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(nb_params)]        
        nb_iter = config_bo.pop('nb_iter')
        config_bo.update({'domain': bounds})
        
        #run bo
        myBopt = GPyOpt.methods.BayesianOptimization(f=s, **config_bo)
        myBopt.run_optimization(nb_iter)   
        
        # generate res
        dict_res = {'X':np.array(myBopt.X),'Y':np.array(myBopt.Y), 'x_opt':myBopt.x_opt, 
                    'fx_opt':myBopt.fx_opt, 'SEED':seed, 'call_f':s._f_calls,'alpha':s.control_fun.alpha}
        
        return dict_res        

    @classmethod
    def _process_collection_res(cls, collection_res, **xargs):
        """ Process the collected res."""
        index = xargs.get('index',0)
        processed = {}
        for k, v in collection_res.items():
            fx_opt = _stats_one_field('fx_opt', v, False, index)
            call_f = _stats_one_field('call_f', v, False)
            Y = _stats_one_field('Y', v, False,0,True)
            processed.update({k:{'fx_opt': fx_opt, 'call_f':call_f, 'Y':Y}})
        return processed

def _stats_one_field(field, list_res, dico_output = False, index=0, extend=False):
    #TODO: Think better about the case when there are arrays of results
    field_values = np.array([res.get(field) for res in list_res])
    mask_none = np.array([f is not None for f in field_values])
    f = field_values[mask_none]
    if(extend): f = _extend(f)
    
    N = len(f)
    if(len(f) > 0):
        field_avg = np.average(f,axis=0)
        field_std = np.std(f,axis=0)
        field_min = np.min(f,axis=0)
        field_max = np.max(f,axis=0)
        field_median = np.median(f,axis=0)
        field_q25 = np.quantile(f, 0.25,axis=0)
        field_q75 = np.quantile(f, 0.75,axis=0)
    else:
        field_avg = np.nan
        field_std = np.nan
        field_min = np.nan
        field_max = np.nan
        field_median = np.nan
        field_q25 = np.nan
        field_q75 = np.nan
    if dico_output:
        res = {'avg':field_avg, 'median': field_median, 'std': field_std, 'min': field_min, 'max':field_max, 'N':N, 'q25':field_q25, 'q75':field_q75}
    else:
        res = [field_avg, field_median, field_std, field_min, field_max, N, field_q25, field_q75]
    return res

def _extend(list_f):
    """ ensure that each element has the same length, if not extend it by replicating its last value"""
    l_max = np.max([len(el) for el in list_f])
    list_ext=[np.array(el) if len(el) == l_max else np.r_[el, np.ones((l_max-len(el),1))*el[-1]] for el in list_f]
    return list_ext

if __name__ == '__main__':
    # 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETER:
    #   + "gen_configs" generate config files from a metaconfig file
    #   + "run_one_config" run cspinoptim based on a config file
    
    if(len(sys.argv) > 5 or len(sys.argv) < 3):
        logger.error("Wrong number of args")
        
    else:
        type_task = sys.argv[1]
        file_input = sys.argv[2]

        if(type_task == "gen_configs"):
            output_f = str(sys.argv[3]) if(len(sys.argv) == 4) else 'Config'
            update_rules = str(sys.argv[4]) if(len(sys.argv) == 5) else True
            BatchQsim.parse_and_save_meta_config(
                    file_input, output_folder = output_f, update_rules = True)

        elif(type_task == "run_one_config"):
            batch = BatchQsim(file_input)
            batch.run_procedures(save_freq = 1)

        else:
            logger.error("first argument not recognized")
   

        



