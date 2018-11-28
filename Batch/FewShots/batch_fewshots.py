# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import logging 
logger = logging.getLogger(__name__)

import sys
import numpy as np
import pdb
from scipy.special import erfinv
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt

if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
else:
    sys.path.append('/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/QuantumSimulation/')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
    
class BatchFS(BatchBase):
    """Implement few shots simulations for batching.
    Provides different methods for optimization / estimation
            
    """
    def run_one_procedure(self, config):
        """ implemention of what is a procedure.
        """
        model_config = config['model']
        optim_config = config['optim']
        if(model_config.get('debug', False) or optim_config.get('debug', False)):
            pdb.set_trace()

        #setting up the models
        nb_measures = model_config['nb_measures'] 
        model = model_config['model']
        noise_input = model_config.get('noise_input', 0)
        verbose = model_config.get('verbose', False)
        f = lambda x: measure(x, nb_measures = nb_measures, verbose = verbose, model = model, noise_input = noise_input)
        f_test = lambda x:  measure(x, nb_measures = np.inf, verbose = verbose, model = model, noise_input = noise_input)

        #setting up the optimizer
        type_optim = optim_config.get('type_optim', 'BO')
        domain = optim_config.get('domain', (0, np.pi))
        nb_init = optim_config['nb_init']
        np.random.seed(config['_RDM_SEED'])
        X_init = np.random.uniform(*domain, nb_init)[:, np.newaxis]
        
        p_target = optim_config.get('p_target')
        if p_target is not None:
            logging.info('USE OF A TARGET FOR THE PROBABILITY')
            f_tmp = optim_config.get('f_target')
            f_target = probit(p_target) if f_tmp is None else f_tmp
            
            logging.info('f_target: {0}'.format(f_target))

        else:
            p_target = 1
            f_target = None

        #MLE
        if(type_optim == 'MLE'):
            max_param = (model // 10) * 2
            domain_mle = (0.5, min(2, max_param))
            Y_init = f(X_init) * nb_measures
            ll = lambda x: loglik(x, X_init, Y_init, model, nb_measures)
            estim = get_max(ll, domain_mle, 7, 1000, 1e-10)
            x_opt = p_to_t(estim, p_target, model)
            test = f_test(x_opt)
            abs_diff = np.abs(p_target - test)
            dico_res = {'test': test, 'test_exp': None, 'estim':estim, 'ptarget':p_target, 
                        'x':x_opt, 'abs_diff':abs_diff}

        #Bayesian Optimization
        elif(type_optim == 'BO'):
            f_BO = lambda x: max(nb_measures,1) * (1 - f(x))
            Y_init = f_BO(X_init)
            type_acq = optim_config['type_acq']
            nb_iter = optim_config['nb_iter']
            type_lik = optim_config['type_lik']
            bounds_bo = [{'name': '0', 'type': 'continuous', 'domain': domain}]
            logger.info('type_acq: {}'.format(type_acq))
            if type_acq == 'EI':
                bo_args = {'acquisition_type':'EI', 'domain': bounds_bo, 
                           'optim_num_anchor':15, 'optim_num_samples':10000} 
            elif type_acq == 'LCB':
                bo_args = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                           'optim_num_anchor':15, 'optim_num_samples':10000, 
                            'acquisition_weight':2, 'acquisition_weight_lindec':True} 
            elif type_acq == 'EI_target':
                bo_args = {'acquisition_type':'EI_target', 'domain': bounds_bo, 
                           'optim_num_anchor':15, 'optim_num_samples':10000,
                           'acquisition_ftarget': f_target} 
            elif type_acq == 'LCB_target':
                bo_args = {'acquisition_type':'LCB_target', 'domain': bounds_bo, 
                           'optim_num_anchor':15, 'optim_num_samples':10000, 
                            'acquisition_weight':2, 'acquisition_weight_lindec':True,
                            'acquisition_ftarget': f_target} 
            else:
                logger.error('type_acq {} not recognized'.format(type_acq))
            if type_lik == 'binomial':
                bo_args.update({
                    'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                    'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
            bo_args['X'] = X_init
            bo_args['Y'] = Y_init

            BO = GPyOpt.methods.BayesianOptimization(f_BO, **bo_args)
            BO.run_optimization(max_iter = nb_iter, eps = 0)
            xy, xy_exp = get_bests_from_BO(BO, f_target)
            test = f_test(xy_exp[0])
            #test_exp = f_test(xy_exp[0])
            abs_diff = np.abs(p_target - test)
            
            dico_res = {'test':test, 'params_BO': BO.model.model.param_array, 
                'params_BO_names': BO.model.model.parameter_names(), 'ptarget':p_target, 
                'x':xy[0], 'xy_exp':xy_exp[0], 'abs_diff':abs_diff} 




        return dico_res 

    @classmethod
    def _process_collection_res(cls, collection_res, **xargs):
        """ Process the collected res. As it is custom to the type of results
        it has to be implemented in the subclasses"""
        processed = {}
        for k, v in collection_res.items():
            #test = _stats_one_field('test', v)
            test_exp = _stats_one_field('test_exp', v)
            test = _stats_one_field('test', v)
            abs_diff = _stats_one_field('abs_diff', v)
            processed.update({k:{'test_exp': test_exp, 'test':test, 'abs_diff':abs_diff}})
        return processed
            
            
def _stats_one_field(field, list_res, dico_output = False):
    field_values = np.array([res.get(field) for res in list_res])
    mask_none = np.array([f is not None for f in field_values])
    f = field_values[mask_none]
    if(len(f) > 0):
        field_avg = np.average(f)
        field_std = np.std(f)
        field_min = np.min(f)
        field_max = np.max(f)
        field_median = np.median(f)
    else:
        field_avg = np.nan
        field_std = np.nan
        field_min = np.nan
        field_max = np.nan
        field_median = np.nan
    if dico_output:
        res = {'avg':field_avg, 'median': field_median, 'std': field_std, 'min': field_min, 'max':field_max}
    else:
        res = [field_avg, field_median, field_std, field_min, field_max]
    return res


def proba(x, noise=0, model = 0):
    """"generate underlying proba p(|1>)"""
    n = 3
    s = 2
    if(noise>0):
        x_noise = x + np.random.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    
    #simple rabbi flip
    if model == 0:
        p = np.square(np.sin(x_noise))        

    elif model == 1:    
        p = np.abs(np.sin(n * x_noise) * np.exp(- np.square((x_noise-np.pi/2) / s)))
    
    #model 0 with decoherence
    elif model // 10 == 1:
        s = model % 10
        p = np.square(np.sin(x_noise)) * np.exp(- 0.5 * np.square(x_noise * s))
        
    #model 0 with decoherence
    elif model // 20 == 1:
        s = model % 10
        p = np.square(np.sin(2 * x_noise)) * np.exp(- 0.5 * np.square(x_noise * s))
    
    elif model // 30 == 1:
        s = model % 20
        p = np.square(np.sin(3 * x_noise)) * np.exp(- 0.5 * np.square(x_noise * s))

    elif model // 50 == 1:
        phase_shift = model % 50
        p = np.square(np.sin(x_noise + phase_shift))
        
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

def get_bests_from_BO(bo, target = None):
    """ From BO optimization extract X giving the best seen Y and best expected 
    for Xs already visited"""
    
    Y_pred = bo.model.predict(bo.X)

    if(target is None):
        y_seen = np.min(bo.Y)
        x_seen = bo.X[np.argmin(bo.Y)]    
        y_exp = np.min(Y_pred[0])
        x_exp = bo.X[np.argmin(Y_pred[0])]
    else:
        if(bo.normalize_Y):
            target = (target - bo.Y.mean())/bo.Y.std() 
        y_seen = np.min(np.abs(bo.Y - target))
        x_seen = bo.X[np.argmin(np.abs(bo.Y - target))]
        y_exp = np.min(np.abs(Y_pred[0] - target))
        x_exp = bo.X[np.argmin(np.abs(Y_pred[0] - target))]

    return (x_seen, y_seen), (x_exp, y_exp)

def probit(p):
    return np.sqrt(2) * erfinv(2 * p -1)



#---------------------------------#
# Parameters estimation
#---------------------------------#
def loglik(params, x, y, model = 0, N = 1):     
    if(model == 1):
        raise NotImplementedError
    else:
        # assume y~Bin(sin^2(param * x), N) / N
        # To be extended to binomial
        phase = (x * params) 
        ll = np.sum(np.log(np.power(np.sin(phase), 2 * N * y)))
        ll += np.sum(np.log(np.power(np.cos(phase), 2 * N * (1 - y))))
    
    return ll


def get_max(f, domain_x, iterative = 3, nb_points = 100, accur_max = 1e-10):
    """ Find the maximum value of a function. Dichotomy style
    """
    logger.info('get_max: iteration {0}'.format(iterative))
    if (np.abs(domain_x[1]- domain_x[0])) < accur_max:
        logger.info('get_max: accuracy_max reached {0}'.format(domain_x[1] - domain_x[0]))
        return (domain_x[1] + domain_x[0])/2
    
    if(iterative>1):
        range_x = np.linspace(*domain_x, nb_points)
        vals = np.array([f(x) for x in range_x])
        ind = np.argsort(-vals)[:3]
        logger.info('get_max: indices found {0}'.format(ind))
        #assert (ind[1]+ind[2]) == 2 * ind[0], 'non consecutive indices {0}'.format(ind)
        x_ind = [range_x[i] for i in ind]
        new_domain = (min(x_ind), max(x_ind))
        res = get_max(f, new_domain, iterative - 1, nb_points, accur_max)
    else:
        logger.info('get_max: Max iter reached - precision = {}'.format(domain_x[1] - domain_x[0]))
        res = (domain_x[1] + domain_x[0])/2
    return res


def p_to_t(estimate, p, model):
    """ For a given omega and p find the associated 
    TODO: Implement model 1
    """
    if(model == 1):
        raise NotImplementedError
    else:
        t = np.arcsin(np.sqrt(p))/estimate

    return t


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
   
    # Just for testing purposes
    testing = False 
    if(testing):
        BatchFS.parse_and_save_meta_config(input_file = 'Inputs/_oneshot100_model10.txt', output_folder = '_configs', update_rules = True)
        batch = BatchFS('_configs/config_res0.txt')
        batch.run_procedures(save_freq = 1)