# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import logging 
logger = logging.getLogger(__name__)

import sys
import numpy as np
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt

if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
else:
    from ..QuantumSimulation.Utility.Optim.batch_base import BatchBase

class BatchFS(BatchBase):
    """
            
    """
    def run_one_procedure(self, config):
        """ implemention of what is a procedure.
        """
        model_config = config['model']
        optim_config = config['optim']

        #setting up the models
        nb_measures = model_config['nb_measures'] 
        model = model_config['model']
        noise_input = model_config.get('noise_input', 0)
        verbose = model_config.get('verbose', False)
        f = lambda x: max(nb_measures,1) * (1-measure(x, nb_measures = nb_measures, 
                            verbose = verbose, model = model, noise_input = noise_input))
        f_test = lambda x:  measure(x, nb_measures = np.inf, verbose = verbose, 
                                    model = model, noise_input = noise_input)

        #setting up the optimizer
        domain_bo = optim_config.get('domain', (0, np.pi))
        type_acq = optim_config['type_acq']
        nb_init = optim_config['nb_init']
        nb_iter = optim_config['nb_iter']
        type_lik = optim_config['type_lik']


        bounds_bo = [{'name': '0', 'type': 'continuous', 'domain': domain_bo}]
        if type_acq == 'EI':
            bo_args = {'acquisition_type':'EI', 'domain': bounds_bo, 
                       'optim_num_anchor':15, 'optim_num_samples':10000} 
        elif type_acq == 'LCB':
            bo_args = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                       'optim_num_anchor':15, 'optim_num_samples':10000, 
                        'acquisition_weight':2, 'acquisition_weight_lindec':True} 
        ### do seeding
        np.random.seed(config['_RDM_SEED'])
        X_init = np.random.uniform(*domain_bo, nb_init)[:, np.newaxis]
        Y_init = f(X_init)
        
        bo_args['X'] = X_init
        bo_args['Y'] = Y_init

        if type_lik == 'binomial':
            bo_args.update({
                'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})

        BO = GPyOpt.methods.BayesianOptimization(f, **bo_args)
        BO.run_optimization(max_iter = nb_iter, eps = 0)
        xy, xy_exp = get_bests_from_BO(BO)
        test = f_test(xy[0])
        test_exp = f_test(xy_exp[0])
        
        return {'test':test, 'test_exp':test_exp, 'params_BO': BO.model.model.param_array, 
                'params_BO_names': BO.model.model.param_names()}




def proba(x, noise=0, model = 0):
    """"generate underlying proba p(|1>)"""
    n = 3
    s = 2
    if(noise>0):
        x_noise = x + np.random.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    if model == 0:
        p = np.square(np.sin(x))
    elif model == 1:    
        p = np.abs(np.sin(n * x_noise) * np.exp(- np.square((x_noise-np.pi/2) / s)))
    else:
        raise NotImplementedError()
    return p

def measure(x, nb_measures = 1, verbose = False, model = 0, noise_input = 0):
    """ Generate projective measurements"""
    p = proba(x = x, noise= noise_input, model = model)
    if(nb_measures == np.inf):
        res = p
    elif(nb_measures < 1):
        res = p +np.random.normal(0, nb_measures, size = np.shape(p))
    else:
        res = np.random.binomial(nb_measures, p)/nb_measures
        
    if(verbose):
        print(np.around(np.c_[res, p, x], 3))
    return res

def get_bests_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y and best expected 
    for Xs already visited"""
    y_seen = np.min(bo.Y)
    x_seen = bo.X[np.argmin(bo.Y)]
    Y_pred = bo.model.predict(bo.X)
    y_exp = np.min(Y_pred[0])
    x_exp = bo.X[np.argmin(Y_pred[0])]
    
    return (x_seen, y_seen), (x_exp, y_exp)

if __name__ == '__main__':

    # 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETER:
    #   + "gen_configs" generate config files from a metaconfig file
    #   + "gen_configs_custom" generate config files from a metaconfig file (with extra_processing)
    #   + "run_one_config" run cspinoptim based on a config file

    if(len(sys.argv) > 5 or len(sys.argv) < 3):
        logger.error("Wrong number of args")
    else:
        type_task = sys.argv[1]
        file_input = sys.argv[2]

        if(type_task == "gen_configs"):
            output_f = str(sys.argv[3]) if(len(sys.argv) == 4) else 'Config'
            update_rules = str(sys.argv[4]) if(len(sys.argv) == 5) else True
            BatchFS.parse_and_save_meta_config(
                    file_input, output_folder = output_f, update_rules = True)

        elif(type_task == "run_one_config"):
            batch = BatchFS(file_input)
            batch.run_procedures(save_freq = 1)

        elif(type_task == "run_meta_config"):
            update_rules = str(sys.argv[3]) if(len(sys.argv) == 4) else True
            batch = BatchFS.from_meta_config(file_input, update_rules = update_rules)
            batch.run_procedures(save_freq = 1)

        else:
            logger.error("first argument not recognized")
   