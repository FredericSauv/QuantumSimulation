# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""

import logging, time, sys, pdb, copy
logger = logging.getLogger(__name__)

import numpy as np
from numpy import inf
import numpy.random as rdm
from scipy.special import erfinv
import scipy.optimize 

sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
import GPy
#import BH1D


if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
else:
    sys.path.append('/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/QuantumSimulation/')
from QuantumSimulation.Utility.Optim.batch_base import BatchBaseParamControl
from QuantumSimulation.ToyModels.BH import BH1D



class BatchSFMI(BatchBaseParamControl):
    """Implement few shots simulations for batching.
    Provides different methods for optimization / estimation
    TODO: Move the logic of the optimization in a parent class
    TODO: Refactor the update of dico_res
    """
    OPTIM_SCIPY = ['DE', 'NM', 'BFGS', 'LBFGSB','SPSA', 'RANDOM','SPSANC']
    OPTIONS_NM = {'disp':True, 'maxiter':200, 'ftol':1e-6, 'maxfev':200, 
                              'adaptative':False, 'max_time':None}
    OPTIONS_DE = {'disp':True, 'maxiter':200, 'popsize':10, 'tol':0.0001}
    OPTIONS_BFGS = {'maxiter': 100, 'disp': False, 'return_all': False}
    OPTIONS_LBFGSB = {'maxiter': 100, 'disp': False, 'return_all': False}
    OPTIONS_SPSA = [{'maxiter': 100, 'disp': False, 'tol': 1e-8, 'a':1, 'b':1, 's':1, 't':1/6,'A':0},
        {'maxiter': 100, 'disp': False, 'tol': 1e-8, 'A':0, 'a':0.5, 'b':0.5, 's':1, 't':1/6},
        {'maxiter': 100, 'disp': False, 'tol': 1e-8, 'A':0, 'a':0.1, 'b':0.1, 's':1, 't':1/6},
        {'maxiter': 100, 'disp': False, 'tol': 1e-8, 'A':0.01*100, 'a':1, 'b':1, 's':0.602, 't':0.101}]
    OPTIONS_RANDOM = {'maxiter': 100}
#    # nb_init':10, 'nb_iter':0, 
#    OPTIONS_BO = {
#        'type_acq':'EI', # Type of acquisition function
#        'ARD':False, # if True different correlation length for each of the input dimension
#        'kernel_type':None, #Will lead to the use of Matern52
#        'type_lik':None, # Type likelihood
#        'mo':None, # Multioutput, i.e. has the output a dim>1
#        'optim_num_anchors':15,
#        'optim_num_samples':10000,
#        'acq_weight':4,
#        'acquisition_weight_lindec':False,
#        'model_update_interval':1, # how often new observations are incorporated
#        'hp_update_interval':1, # how often hyperparameters are updated
#        'num_cores': 1,
#        'max_iters': 1000, # used when updating the hyper-parameters
#        'optimize_restarts':3, # for the hyperparameters fitting
#        'hp_constrains': None,
#        'switch_to_gauss': None,
#        'n_meas':1,
#        'hp_restart':False
#        }
    

    
    def setup_MOCK_model(self, model_config, test_config):
        """ A simple toy function to test the optimizers 
        Bowl function (x - x_tgt).(x - x_tgt)
        Convex -> such that gradient methods should always converge to the minima
        """
        logger.info("+++ Mock function IN USE +++")
        # potential arguments
        verbose = model_config.get('verbose', False)
        self.n_params = model_config.get('n_params', 4)
        self.noise_output = model_config.get('noise_output', 0)
        # fixed
        self.nb_output, self.n_meas = 1, 1        
        self.domain = [(0, 1) for _ in range(self.n_params)]
        self.shift = np.random.uniform(0,1, self.n_params)
        self.x_tgt = self.shift
        self.model, self.model_test = None, None 
        self.phi_0 = None
        self.phi_tgt = None
        self.T = None
        self.e_tgt = None
        self.p_tgt = None
        self.f_tgt = None 
        self.fid_zero = None
        self.warping_input = False # should change
        self.noise_iput = False # should change
        
        def f(x, verbose = verbose, trunc_res = True, **args_call):
            """ evaluate the mock function"""
            if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
                if(verbose): print(np.shape(x))
                res = np.array([f(x_one, verbose, trunc_res, **args_call) for x_one in x])
            else:
                res = np.dot(x - self.shift, x-self.shift)
                if(self.noise_output > 0):
                    res += np.random.normal(0, self.noise_output)
                self.call_f += 1
                if(verbose): print(x, res)
            return np.atleast_1d(res)
        
        def f_test(x, verbose = verbose, trunc_res = True, **args_call):
            """ evaluate the mock function"""
            if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
                if(verbose): print(np.shape(x))
                res = np.array([f_test(x_one, verbose, trunc_res, **args_call) for x_one in x])
            else:
                res = np.dot(x - self.shift, x-self.shift)
                self.call_f_test += 1
                if(verbose): print(x, res)
            return np.atleast_1d(res)
        
        return f, f_test, self.x_tgt
    
    def _get_nb_meas(self, main_fom):
        """ custom (aka not clean) parsing of the fom string"""
        n_meas_each = 1
        n_meas = 1
        n_output = 1
        if ('freqAvgOne' in main_fom) and (len(main_fom)>10):
            n_meas = int(main_fom[10:])
            n_meas_each = self.model.L
            n_output = 1
        elif ('freqMI' in main_fom) and (len(main_fom)>6):
            n_meas = int(main_fom[6:])
            n_meas_each = 1
            n_output = 1
        elif ('freqEachOne' in main_fom) and (len(main_fom)>11):
            n_meas = int(main_fom[11:])
            n_meas_each = 1
            n_output = self.L
        return n_meas, n_meas_each, n_output
    
    def setup_QSPIN_model(self, model_config, test_config):
        """ Setup the model in QuTip allowing to compute the dynamics and FoM
        """        
        #print(zero)
        verbose = model_config.get('verbose', False)
        self.model = BH1D.BH1D(**model_config)
        self.model_test = BH1D.BH1D(**test_config)
        self.n_params = self.model.n_params
        self.main_fom = model_config['fom'][0].split(':')[0]
        self.n_meas, self.n_meas_each, self.nb_output = self._get_nb_meas(self.main_fom)
        self.n_meas_total = self.n_meas * self.n_meas_each

        self.domain = self.model.params_bounds
        self.phi_0 = self.model.state_init
        self.phi_tgt = self.model.state_tgt
        self.T = self.model.T        
        self.x_tgt = None
        self.e_tgt = None
        self.p_tgt = None
        self.f_tgt = None 
        self.fid_zero = None
        # new features possibility to distort input either by stocchastic noise
        # or some deterministic warping
        # Note1 warp is used by f_test but not stocchastic noise (it may change
        #       - maybe should use model_test)
        # Note2 if they are both non None first noise then warp
        if model_config.get('warp_args'): 
            self.warping_input = True 
            self.warping_string = model_config['warp_args']
            self.alpha = read_string_rv(self.warping_string)
            self.warper = Warp(self.alpha)
        else:
            self.warping_input = False
            self.warper = None
            
        if model_config.get('noise_input'):
            self.noise_input = True
            self.noise_input_string = model_config['noise_input']
            self.noise_input_func = read_string_rv(self.noise_input_string, function = True, size = self.n_params)
        else:
            self.noise_input = False
            self.noise_input_func = None
            
        def f(x, verbose = verbose, trunc_res = True, noisy = self.noise_input,
              warped = self.warping_input,  **args_call):
            """ evaluate the  model with parameters x   """
            if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
                print(x)
                print(np.shape(x))
                res = np.array([f(x_one, verbose, trunc_res, **args_call) for x_one in x])
            else:
                if noisy: x += self.noise_input_func()
                if warped: x = self.warper(x)
                res = self.model(x, trunc_res = trunc_res, **args_call)
                if(trunc_res): res = np.atleast_1d(res)[0] 
                self.call_f += self.n_meas
                if(verbose): print(x, res)
            return np.atleast_1d(res)
        
    
        #Use for testing the optimal pulse
        def f_test(x, verbose = verbose, trunc_res = False, noisy = False,
              warped = self.warping_input, **args_call):
            """ evaluate the test model with parameters x   """
            if(np.ndim(x)>1) and (np.shape(x)[0] != self.n_params):
                print(x)
                print(np.shape(x))
                res = np.array([f_test(x_one, verbose, trunc_res, **args_call) for x_one in x])
            else:
                if noisy: x += self.noise_input_func()
                if warped: x = self.warper(x)
                res = self.model_test(x, trunc_res = trunc_res, **args_call)
                if(trunc_res): res = np.atleast_1d(res)[0] 
                self.call_f_test += self.n_meas
                if(verbose): print(x, res)
            return np.atleast_1d(res)
        
        return f, f_test, self.x_tgt


    def run_one_procedure(self, config):
        """ implemention of what is a procedure. Optimizers are implemented such 
        that they are effectively minimizers
        """
        # setting the models
        self.call_f, self.call_f_single, self.call_f_test = 0, 0, 0
        model_config = config['model']
        self._build_control(model_config)  ### TOCHECKKK
        # 2 ways of passing a testing config: either full config or just an update to apply to the model
        test_config = config.get('test', None)
        if(test_config is None):
            test_config = model_config.copy()
            test_update_config = config.get('test_update', {})
            test_config.update(test_update_config)        
        self._build_control(test_config)  ### TOCHECKKK


        if(model_config.get('debug', False)): pdb.set_trace()
        np.random.seed(config.get('_RDM_SEED', None))
        if model_config.get('FAKE', False):
            self.f, self.f_test, self.x_tgt = self.setup_MOCK_model(model_config, test_config)
        else:
            self.f, self.f_test, self.x_tgt = self.setup_QSPIN_model(model_config, test_config)
        
        time_start = time.time()
        dico_res = {}
        if (self.n_params > 0): # Main case: there are some parameters to optimize
            optim_config = config['optim']
            #setting up the optimizer
            type_optim = optim_config.get('type_optim', 'BO')
            params_min, params_max = np.array(self.domain)[:,0], np.array(self.domain)[:,1]

            if(type_optim in self.OPTIM_SCIPY):
                init = np.array([np.random.uniform(*d) for d in self.domain])
                if(type_optim == 'RANDOM'):
                    options = self.OPTIONS_NM.copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    options.update(to_update) 
                    X_init = np.transpose([np.atleast_1d(np.random.uniform(*d, options['maxiter'])) for d in self.domain])    
                    Y_init = self.f(X_init)
                    index_best = np.argmin(np.average(Y_init, 1)) #take the
                    optim = {'x':X_init[index_best], 'message':'RANDOM'}
    
                if(type_optim == 'NM'):
                    """ Implementation of a Nedler Meat optimization  
                    scipy.optimize.minimize(fun, x0, args=(), method='Nelder-Mead', tol=None, 
                    callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 
                    'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 
                    'fatol': 0.0001, 'adaptive': False})"""
                    options = self.OPTIONS_NM.copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    options.update(to_update)
                    init_simplex = np.transpose([np.random.uniform(*d, self.n_params+1) for d in self.domain])
                    options['initial_simplex'] = init_simplex
                    if(optim_config.pop('max_time', None) is not None): logger.warning('NM: max_time is not used')
                    optim = scipy.optimize.minimize(self.f, x0 = init_simplex[0, :], args = () , 
                                method='Nelder-Mead', options = options, callback = None)
                
                elif(type_optim == 'SPSA'):
                    """ Implementation of SPONTANEOUS PERTURBATION STOCHASTIC 
                    APPROXIMATION """
                    config_spsa = optim_config.get('config_spsa',0)
                    options = self.OPTIONS_SPSA[config_spsa].copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    options.update(to_update)
                    
                    params = init
                    alpha = lambda k: options['a']/np.power(k+1+options['A'], options['s'])
                    beta = lambda k: options['b']/np.power(k+1, options['t']) 
                    optim = {}
                    for k in range(int(options['maxiter']/2)):
                        a_k = alpha(k)
                        b_k = beta(k)
                        perturb = np.sign(np.random.uniform(0,1, self.n_params) - 0.5)
                        x_p = np.clip(params + b_k * perturb, params_min, params_max)
                        x_m = np.clip(params - b_k * perturb, params_min, params_max)
                        f_p = self.f(x_p)
                        f_m = self.f(x_m)
                        g_k = (f_p - f_m)/(x_p - x_m) ## Careful
                        params = np.clip(params - a_k * g_k, params_min, params_max)
                        if np.max(np.abs(a_k * g_k)) < options['tol']:
                            optim['message'] = 'Break because less than tol'
                            break
                    optim['x'] = params
                    optim['message'] = optim.get('message', 'SPSA: Stopped because nbfev reached')
                
                #with no clipping
                elif(type_optim == 'SPSANC'):
                    """ Implementation of SPONTANEOUS PERTURBATION STOCHASTIC 
                    APPROXIMATION """
                    config_spsa = optim_config.get('config_spsa',0)
                    options = self.OPTIONS_SPSA[config_spsa].copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    options.update(to_update)
                    
                    params = init
                    alpha = lambda k: options['a']/np.power(k+1+options['A'], options['s'])
                    beta = lambda k: options['b']/np.power(k+1, options['t']) 
                    optim = {}
                    for k in range(int(options['maxiter']/2)):
                        a_k = alpha(k)
                        b_k = beta(k)
                        perturb = np.sign(np.random.uniform(0,1, self.n_params) - 0.5)
                        x_p = params + b_k * perturb
                        x_m = params - b_k * perturb
                        f_p = self.f(x_p)
                        f_m = self.f(x_m)
                        g_k = (f_p - f_m)/(x_p - x_m)
                        params = params - a_k * g_k
                        if np.max(np.abs(a_k * g_k)) < options['tol']:
                            optim['message'] = 'Break because less than tol'
                            break
                    optim['x'] = params
                    optim['message'] = optim.get('message', 'SPSA: Stopped because nbfev reached')
                
                elif(type_optim == 'DE'):
                    """ Implementation of a differential evolution optimizer (need bounds)
                    scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin',
                    maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, 
                    callback=None, disp=False, polish=True, init='latinhypercube') """
                    init = None
                    options = self.OPTIONS_DE.copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    options.update(to_update)
                    bounds = self.domain
                    
                    #workaround to terminate optims when a certain time has been reached
                    # use a callback with two mock arguments 
                    mt = options.get('max_time', None)
                    #TODO: Add fev in the callback
                    #mfev = options.get('maxfev', None)
                    #if(mt is not None) or (maxfev is not None):
                    if mt is not None:
                        time_limit = time.time() + int(options.pop('max_time'))
                        options['callback'] = lambda x, convergence: time.time() > time_limit
                    optim = scipy.optimize.differential_evolution(self.f, bounds, **options)
                    
                    if(options.get('callback') is not None):
                        optim['maxtime'] = options.get('callback')(None, None)
    
                elif(type_optim == 'BFGS'):
                    options = self.OPTIONS_BFGS.copy()
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    optim = scipy.optimize.minimize(self.f, x0=init, method='BFGS', 
                                options=options)
    
                elif(type_optim == 'LBFGSB'):
                    options = self.OPTIONS_LBFGSB
                    to_update = {k: optim_config[k] for k, v in options.items() if k in optim_config}
                    optim = scipy.optimize.minimize(self.f, x0=init, method='L-BFGS-B',
                                bounds = self.domain, options=options)
                    
                cum_time = time.time() - time_start
                x_exp = optim['x']
                x_seen = x_exp
                test = self.f_test(x_exp)
                abs_diff = model_config.get('res_tgt', 1) - test
                dico_res.update({'test':test, 
                            'x':x_seen, 'x_exp':x_exp, 'abs_diff':abs_diff,
                            'time_suggest':cum_time, 'time_fit':0, 
                            'message_optim':optim['message'], 'x_init':init})      
                
            elif(type_optim == 'CRAB'):
                raise NotImplementedError()
                
            elif(type_optim == 'GRAPE'):
                not NotImplementedError()
                 
            #Bayesian Optimization 2 flavors 'BO' and 'BO_NOOPTIM'
            # 'BO' classical bayesian optimization
            # 'BO_NOOPTIM' all the x are randomly generated and GP is fitted 
            #              x_best is decided based on this model
            elif 'BO' in type_optim: 
                ### Main
                self.time_all_bo, self.time_fit_bo, self.time_suggest_bo = 0, 0, 0
                self.set_up_BO(optim_config)
                self.max_time_bo = self.bo_args['max_time_bo']  #Need to be done here as it shouldnot been reset
    
                if(self.save_extra_bo):
                    dico_res.update(tag='init0_', save_full = self.save_extra_full)
                self.BO.run_optimization(max_iter = self.bo_args['nb_iter_bo'], 
                                         eps = 0, max_time = self.max_time_bo)
                self.update_BO_time()
                if(self.save_extra_bo):
                    dico_res.update(self.get_info_BO(i_beg = None, i_end = None, 
                                tag='explor0_', save_full = self.save_extra_full))
        
                nb_exploit = self.bo_args['nb_exploit']
                if nb_exploit>0:
                    self.set_up_BO_exploit()
                    logger.info('Exploitation (i.e. ucb with k=0) for {}'.format(nb_exploit))
                    self.BO.run_optimization(nb_exploit, max_time = self.max_time_bo)
                    self.update_BO_time()
                    if(self.save_extra_bo):
                        dico_res.update(self.get_info_BO(tag='exploit0_', save_full = self.save_extra_full))
                
                ### Extra steps
                nb_polish = self.bo_args['nb_polish']
                nb_to_keep = self.bo_args['nb_to_keep']
                nb_more = self.bo_args['nb_more']
                nb_iter_polish = self.bo_args['nb_iter_polish']
                kernel_list = self.bo_args['kernel_list']
                acq_list = self.bo_args['acq_list']
                polish_step = 0
                X_keep_track = np.c_[(self.BO.X, np.zeros(len(self.BO.X)))]
                Y_keep_track = np.c_[(self.BO.Y, np.zeros(len(self.BO.Y)))]
                self.save_hp_values()
                while nb_polish > 0 and self.max_time_bo > 0:
                    more = nb_more[polish_step]
                    new_iter = nb_iter_polish[polish_step]
                    keep = nb_to_keep[polish_step]
                    kern = kernel_list[polish_step]
                    aw = acq_list[polish_step]
                    logger.info('Polish, nb to keep {}, X times more shots {} '.format(keep, more))
                    nb_polish -= 1  
                    polish_step += 1
                    # change underlying fom
                    n_meas_old = self.n_meas
                    n_meas_new = more * n_meas_old
                    self.main_fom = self.main_fom.replace(str(n_meas_old), str(n_meas_new))
                    fom0 = self.model.fom[0].split(':')
                    fom0[0] = self.main_fom
                    fom0 = ":".join(fom0)
                    self.model.fom[0] = fom0
                        
                    #change BO
                    self.n_meas = n_meas_new
                    self.n_meas_total = self.n_meas * self.n_meas_each
                    self.set_up_BO(optim_config, nb_restrict_data = keep, restrict_domain = True, adapt_shots=more, ow_kernel= kern, ow_acq_weight = aw)   
                    if(self.save_extra_bo):
                        dico_res.update(self.get_info_BO(save_full = self.save_extra_full))
                    self.BO.run_optimization(max_iter = new_iter, eps = 0, max_time = self.max_time_bo)
                    if(self.save_extra_bo):
                        dico_res.update(self.get_info_BO(tag='explor' + str(polish_step) + '_', 
                                                         save_full = self.save_extra_full))
                    if nb_exploit>0:
                        self.set_up_BO_exploit()
                        logger.info('Exploitation (i.e. ucb with k=0) for {}'.format(nb_exploit))
                        self.BO.run_optimization(nb_exploit, max_time=self.max_time_bo)
                        self.update_BO_time()
                        if(self.save_extra_bo):
                            dico_res.update(self.get_info_BO(tag='exploit' + str(polish_step) + '_', 
                                                             save_full = self.save_extra_full))
    
                                    
                    X_keep_track = np.r_[(X_keep_track, np.c_[(self.BO.X, polish_step*np.ones(len(self.BO.X)))])]
                    if(self.BO.Y.shape[1] == (Y_keep_track.shape[1]-1)):
                        Y_keep_track = np.r_[(Y_keep_track, np.c_[(self.BO.Y, polish_step*np.ones(len(self.BO.Y)))])]
                    else:
                        to_add_tmp = [Y_keep_track] * (Y_keep_track.shape[1]-1)
                        to_add_tmp.append(polish_step*np.ones(len(self.BO.Y)))
                
                dico_res.update(self.get_info_BO(tag=''))            
                dico_res.update({'params_BO_names': self.BO.model.model.parameter_names(), 
                    'abs_diff':1- dico_res['test'], 
                    'time_allbo':self.time_all_bo, 'time_fit':self.time_fit_bo, 
                    'time_suggest':self.time_suggest_bo, 'polish_step':polish_step, 
                    'nb_polish':nb_polish, 'nb_more':nb_more, 'nb_to_keep':nb_to_keep})
                x_exp = dico_res['x_exp']
                
            if(test_config.get('gradients',False)):
                logger.info('Test gradients at the final value, with eps = 1e-6')
                nb_output = len(self.f_test(x_exp)) # not clean
                grad_final = np.zeros((self.n_params, nb_output))
                for i in range(self.n_params):
                    perturb_local = np.zeros(self.n_params)
                    perturb_local[i] = 1
                    eps = 1e-6
                    x_clip = np.clip(x_exp, params_min, params_max)
                    x_p = np.clip(x_clip + eps * perturb_local, params_min, params_max)
                    x_m = np.clip(x_clip - eps * perturb_local, params_min, params_max)
#                    if np.allclose(x_p, x_m):
#                        grad_final[i,:] = np.zeros(nb_output)
#                    else:
                    f_p = self.f_test(x_p)
                    f_m = self.f_test(x_m)
                    grad_final[i,:] = (f_p - f_m)/(x_p[i] - x_m[i])
                dico_res.update({'final_grad':grad_final})            
        
        else: #case when there is no params, i.e. nothing to optimize
            test = self.f_test(np.zeros(self.n_params))
            abs_diff = model_config.get('res_tgt', 1) - test
            dico_res.update({'test':test,  'x':None, 'x_exp':None, 
                        'abs_diff':abs_diff, 'time_suggest':0, 'time_fit':0, 
                        'message_optim':'ONLY TESTING', 'x_init':None})      
            
                    
        cum_time = time.time() - time_start        
        dico_res.update({'time_all':cum_time,'x_tgt':self.x_tgt, 
            'call_f':self.call_f, 'call_f_single':self.call_f_single,
            'call_f_test': self.call_f_test, 'p_tgt':self.p_tgt, 
            'f_tgt':self.f_tgt,'fid_zero_field':self.fid_zero, 
            'phi_0': self.phi_0, 'phi_tgt':self.phi_tgt, 
            'nb_output':self.nb_output})
        if hasattr(self.model, '_track_calls'):
            dico_res.update({'tracker':self.model._track_calls})
        if hasattr(self.model_test, '_track_calls'):
            dico_res.update({'tracker_test':self.model_test._track_calls})
        if self.warping_input:
            dico_res.update({'warp_constant':self.alpha})
        if self.noise_input:
            dico_res.update({'noise_descr':self.noise_input_string})
            
            

        return dico_res 

    @classmethod
    def _process_collection_res(cls, collection_res, **xargs):
        """ Process the collected res. As it is custom to the type of results
        it has to be implemented in the subclasses"""
        index = xargs.get('index',0)
        processed = {}
        for k, v in collection_res.items():
            #test = _stats_one_field('test', v)
            test_exp = _stats_one_field('test_exp', v, False, index)
            test = _stats_one_field('test', v, False, index)
            abs_diff = _stats_one_field('abs_diff', v, False, index)
            call_f = _stats_one_field('call_f', v, False)
            time_elapsed = _stats_one_field('time_elapsed', v, False)
            time_fit = _stats_one_field('time_fit', v, False)
            time_suggest = _stats_one_field('time_suggest', v, False)
            best_res = _get_some_res('test', v, np.max)
            median_res = _get_some_res('test', v, np.median)
            processed.update({k:{'test_exp': test_exp, 'test':test, 
                'abs_diff':abs_diff, 'time_elapsed':time_elapsed,'call_f':call_f, 
                'best_res': best_res, 'median_res':median_res, 'time_fit':time_fit,
                'time_suggest':time_suggest}})
        return processed


    def update_BO_time(self):
        self.time_all_bo += self.BO.cum_time
        self.time_fit_bo += self.BO.cum_time_fit
        self.time_suggest_bo += self.BO.cum_time_suggest
        self.max_time_bo -= self.BO.cum_time
            

    def set_up_BO_exploit(self):
        self.BO.acquisition_type = self.BO.acquisition_type.replace('EI', 'LCB')
        self.BO.acquisition_weight = 0.000001
        self.BO.kwargs['acquisition_weight'] = 0.000001
        self.BO.acquisition = self.BO._acquisition_chooser()
        self.BO.evaluator = self.BO._evaluator_chooser()    
        

    def set_up_BO(self, optim_config, nb_restrict_data = None, restrict_domain = False, 
                  adapt_shots=1,ow_kernel=None,ow_acq_weight=None):
        """ set up the BO object. At the end the model is initiated
        Restrict domain // Implement rules for the adaptation of the number of shots // 
        Deal with the fixing of the constraints and the forcing the use of grad
        
        adapt_shots:
            <int> N: multi
            <str> 'Strat1': 
        """
        MAX_INCREASE = 5000
        self.save_extra_bo = optim_config.get('save_extra_bo', False)        
        self.save_extra_full = optim_config.get('save_extra_full', False)
        if ow_kernel is not None:
            optim_config['kernel_type'] = ow_kernel
            logger.info('kernel owritten: {}'.format(ow_kernel))
        if ow_acq_weight is not None:
            optim_config['acq_weight'] = ow_acq_weight
            logger.info('acq_weight owritten: {}'.format(ow_acq_weight))
        if nb_restrict_data is not None:
            (_,_), (x_exp, _) = self.BO.get_best()
            X_to_keep, Y_to_keep = filter_X(self.BO.X, self.BO.Y, x_exp, nb_restrict_data)
        else:
            if(hasattr(self, 'BO')):
                X_to_keep = self.BO.X 
                Y_to_keep = self.BO.Y
            else:
                X_to_keep = None
                Y_to_keep = None

        if restrict_domain:
            self.domain = [(mi, ma) for mi, ma in zip(np.min(X_to_keep, 0), np.max(X_to_keep, 0))]

        if type(adapt_shots) == int: #base case
            if adapt_shots == 1:
                optim_config.update({'X_init':X_to_keep, 'Y_init':Y_to_keep})
            else:
                optim_config.update({'X_init':X_to_keep, 'Y_init':None})

            self.bo_args, f_BO = self.get_BO_args(optim_config)
            self.BO = GPyOpt.methods.BayesianOptimization(f_BO, **self.bo_args)            
            force_grad_acq = optim_config.get('force_grad_acq',False)
            if force_grad_acq: self.BO.acquisition.analytical_gradient_acq = True
            hp_restart = self.bo_args['hp_restart']
            if hp_restart: 
                self.BO._create_model(self.BO.normalization_type)
                self.restore_hp_values() ### What about first step
            if self.hp_constrains is not None: 
                self.BO._create_model(self.BO.normalization_type)
                self.constrain_hp(self.hp_constrains)


        else:
            model_mean, model_std = self.BO.acquisition._compute_acq_splitted(X_to_keep)
            model_std_avg = np.median(model_std)
            model_mean_disp = np.std(model_mean)
            try:
                alpha = np.sqrt(np.median(model_mean, 0) * (1 - np.median(model_mean, 0))/self.n_meas) 
            except:
                alpha = 0.5
            std_data = np.average(np.std(Y_to_keep, 0))/self.n_meas
            
            if adapt_shots == 'strat1':
                coeff = int(np.clip(np.square(alpha/std_data) , 1, MAX_INCREASE))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)
                    
            elif adapt_shots == 'strat2':
                coeff = int(np.clip(np.square(model_mean_disp/model_std_avg), 1,MAX_INCREASE))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)

            elif adapt_shots == 'strat3': 
                coeff =  int(np.clip(np.square(model_mean_disp/model_std_avg), 1, 100))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)
                self.set_up_BO(optim_config, nb_to_keep = None, restrict_domain = False, adapt_shots='strat2')

            else:
                raise NotImplementedError()
                
                

            



    def get_BO_args(self, optim_config):
        nb_init = optim_config['nb_init']
        nb_iter = optim_config['nb_iter']
        type_acq = optim_config['type_acq']
        ARD = optim_config.get('ARD', False)
        kernel_type = optim_config.get('kernel_type', None)
        #is_acq_target = type_acq.find('target') > 0
        logger.info('type_acq: {}'.format(type_acq))
        type_lik = optim_config['type_lik']
        mo = optim_config.get('mo')
        nb_anchors = optim_config.get('nb_anchors', 15)
        optim_num_samples= optim_config.get('optim_num_samples', 10000)
        acq_weight = optim_config.get('acq_weight', 4)
        acquisition_weight_lindec = optim_config.get('acquisition_weight_lindec', True)
        model_update_interval= optim_config.get('model_update_interval', 1)
        hp_update_interval= optim_config.get('hp_update_interval', 1)
        hp_update_first= optim_config.get('hp_update_first', True)
        num_cores = optim_config.get('num_cores', 1)
        max_iters = optim_config.get('max_iters', 1000) # used when updating the hyper-parameters
        optimize_restarts = optim_config.get('optimize_restarts',5) # for the hyperparameters fitting
        self.hp_constrains = optim_config.get('hp_constrains', None)
        switch_to_gauss = optim_config.get('switch_to_gauss', None)
        # distinction between n_meas and n_meas_total in some cases
        n_meas = self.n_meas if hasattr(self, 'n_meas') else 1
        n_meas_total = self.n_meas_total if hasattr(self, 'n_meas_total') else n_meas
                
        # some redundancy here 
        aggregate = optim_config.get('aggregate', 'no')
        aggregate = 'fid' if(aggregate == True) else aggregate
        aggregate = 'no' if(aggregate == False) else aggregate
        
        if kernel_type == 'Matern52':
            kernel = GPy.kern.Matern52(self.n_params, variance=1., ARD=ARD)
        elif kernel_type == 'Matern32':
            kernel = GPy.kern.Matern32(self.n_params, variance=1., ARD=ARD)
        elif kernel_type == 'RBF':
            kernel = GPy.kern.RBF(self.n_params, variance=1., ARD=ARD)
        elif kernel_type == 'Exponential':
            kernel = GPy.kern.Exponential(self.n_params, variance=1., ARD=ARD)
        else:
            kernel = None

        
        if (switch_to_gauss is not None) and (n_meas > switch_to_gauss):
            #overwride parameters
            type_lik = None
            #aggregate = 'fid'
            #is_acq_target = False
            type_acq = type_acq.split('_')[0]
            mo=None
            logger.warning('nb shots: {}, Swith to Gaussian likelihood with fidelity aggregated output'.format(n_meas))

        if(optim_config.get('type_optim', 'BO') == 'BO_NOOPTIM'):
            nb_init_bo = nb_init + nb_iter
            nb_iter_bo = 0
            max_time_bo = optim_config.get('max_time', 23.5*3600)
        else:
            nb_init_bo = nb_init
            nb_iter_bo = nb_iter
            max_time_bo = optim_config.get('max_time', 23.5*3600)
        

        if(type_lik=='binomial'):
            f_BO = lambda x, **kw : (n_meas_total * self.f(x, **kw)).astype(int)
            #f_fact = n_meas 
        else:
            f_BO = self.f            
            #f_fact = 1
        
        #if is_acq_target:
        #    f_wrap = lambda x: self.f(x) 
        #    self.warp_f = lambda x:x
        #else: 
        #   f_wrap = lambda x: 1-self.f(x)
        #self.warp_f = lambda x:1-x
        #f_BO = lambda x: f_fact * f_wrap(x)
        #f_BO = self.f
        self.warp_f = lambda x:x
        
        if optim_config.get('X_init', None) is not None:
            X_init = optim_config['X_init']            
        else:
            X_init = np.transpose([rdm.uniform(*d, nb_init_bo) for d in self.domain]) 

        if optim_config.get('Y_init', None) is not None:
            Y_init = optim_config['Y_init']
        else:
            Y_init = f_BO(X_init)
        bounds_bo = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(self.domain)]

        if 'polish' not in optim_config:
            nb_polish = 0
            nb_to_keep = 0
            nb_more = 0
            hp_restart = False
            nb_iter_polish = 0
            kernel_list = None
            acq_list = None
        else:
            polish_dico = optim_config['polish']
            nb_polish = polish_dico.get('nb_polish',0)
            nb_to_keep = make_iter_if_not_iter(polish_dico.get('nb_to_keep', nb_init), nb_polish)
            nb_more = make_iter_if_not_iter(polish_dico.get('nb_more', 1), nb_polish)
            kernel_list = make_list_if_not_list(polish_dico.get('kernel_list', kernel_type), nb_polish)
            acq_list = make_list_if_not_list(polish_dico.get('acq_list', acq_weight), nb_polish)
            nb_iter_polish = make_iter_if_not_iter(polish_dico.get('nb_iter', nb_iter_bo), nb_polish)
            hp_restart = polish_dico.get('hp_restart', False)
        nb_exploit = optim_config.get('exploitation_steps',0)
        
        bo_args = {'model_update_interval':model_update_interval, 'X':X_init,
                'Y':Y_init, 'domain': bounds_bo, 'optim_num_anchor':nb_anchors, 
                'optim_num_samples':optim_num_samples, 'num_cores':num_cores, 
                'max_iters':max_iters, 'optimize_restarts':optimize_restarts,
                'hp_update_interval':hp_update_interval, 'hp_update_first':hp_update_first, 
                'nb_iter_bo':nb_iter_bo,'max_time_bo':max_time_bo, 'nb_polish':nb_polish, 
                'nb_to_keep':nb_to_keep,'nb_more':nb_more, 'nb_exploit':nb_exploit, 
                'hp_restart':hp_restart, 'nb_iter_polish':nb_iter_polish,'ARD':ARD, 
                'kernel':kernel,'kernel_list':kernel_list, 'acq_list':acq_list}
        
        if type_acq == 'EI':
            bo_args.update({'acquisition_type':'EI'})
        elif type_acq in ['LCB', 'LCB_pspace']:
            bo_args.update({'acquisition_type':type_acq, 'acquisition_weight':acq_weight, 
                            'acquisition_weight_lindec':acquisition_weight_lindec})
        
        elif type_acq == 'EI_target':
            bo_args.update({'acquisition_type':'EI_target', 'acquisition_ftarget': self.p_tgt})
        
        elif type_acq == 'LCB_target_oneq':
            acq_nbq = optim_config['acq_nbq']
            bo_args.update({'acquisition_type':'LCB_oneq', 'acquisition_ftarget': self.p_tgt,
                            'acquisition_weight':acq_weight,'acquisition_weight_lindec':acquisition_weight_lindec,'acq_nbq':acq_nbq})
        
        elif type_acq == 'LCB_target':
            bo_args.update({'acquisition_type':'LCB_target','acquisition_weight':acq_weight, 
                            'acquisition_weight_lindec':acquisition_weight_lindec, 'acquisition_ftarget': self.p_tgt})
        else:
            logger.error('type_acq {} not recognized'.format(type_acq))
  
        if type_lik == 'binomial':
            logger.info('FTARGET is used by BO')
            bo_args.update({ 'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                'likelihood':'Binomial_' + str(n_meas_total), 'normalize_Y':False})
            if type_acq != 'LCB_target_oneq':
                bo_args.update({'acquisition_ftarget':self.f_tgt})
        
        elif type_lik == 'bernoulli':
            logger.info('FTARGET is used by BO')
            bo_args.update({ 'model_type':'GP_CUSTOM_LIK', 'inf_method':'EP', 
                'likelihood':'Bernoulli_1', 'normalize_Y':False})
            if type_acq != 'LCB_target_oneq':
                bo_args.update({'acquisition_ftarget':self.f_tgt})
        # multioutput
        if mo is not None:
            bo_args.update({'mo':mo})
            self.nb_output = mo['output_dim']
        else:
            self.nb_output = 1
        self.type_likelihood = type_lik
        
        return bo_args, f_BO
    
    def get_info_BO(self, i_beg = None, i_end = None, tag='', save_full=False):
        """ Extract information about the optimization performed so far:
            best according to the model, etc.""" 
        res = {}
        (x_seen, _), (x_exp,_) = self.BO.get_best()
        x = self.BO.X
        y = self.BO.Y
        if i_beg is not None:
            x, y = x[i_beg:], y[i_beg:]
        if i_end is not None:
            x, y = x[:-i_end], y[:-i_end]
        res[tag + 'bo_x'] = x
        res[tag + 'bo_y'] = y
        bo_acq, bo_acq_std = self.BO.acquisition._compute_acq_splitted(x)
        if self.BO.normalize_Y:
            norm_args = self.BO.norm_args
            #normalization_type = self.normalization_type
            norm_mean = norm_args['mean']
            norm_std = norm_args['std']
            bo_acq = norm_std * bo_acq + norm_mean
            bo_acq_std = norm_std * bo_acq_std
        else:
            norm_mean = 0.
            norm_std = 1.
        bo_acq = self.warp_f(bo_acq)
        res[tag + 'norm_mean'] = norm_mean
        res[tag + 'norm_std'] = norm_std
        res[tag + 'bo_acq'] = bo_acq
        res[tag + 'bo_acq_std'] = bo_acq_std
        if(save_full):
            res[tag + 'bo_tgt'] = self.f_test(x)
        else:
            res[tag + 'bo_tgt'] = None
        res[tag + 'bo_args'] = np.copy(self.BO.model.model.param_array), 
        res[tag + 'bo_args_names'] = self.BO.model.model.parameter_names()
        res[tag + 'domain'] = np.array([d['domain'] for d in self.BO.domain])
        res[tag + 'x_exp'] = x_exp
        res[tag + 'x_seen'] = x_seen
        res[tag + 'test'] = self.f_test(x_exp)
        res[tag + 'nb_s'] = self.n_meas
        res[tag + 'call_f'] = self.call_f
        res[tag + 'call_f_single'] = self.call_f_single
        try:
            res[tag + 'alpha'] = np.sqrt(np.average(bo_acq, 0) * (1 - np.average(bo_acq, 0))/self.n_meas) 
        except:
            res[tag + 'alpha'] = 0.5
        
        return res
    
    def constrain_hp(self, constrains_dico):
        for k, v in constrains_dico.items():
            str_param = '.*'+k
            if np.any([k in name for name in self.BO.model.model.parameter_names()]):
                if v == 'positive':
                    self.BO.model.model[str_param].constrain_positive()
                elif (hasattr(v, '__iter__')) and (v[:5] == 'fixed'):
                    val = float(v[5:])
                    self.BO.model.model[str_param] = val
                    self.BO.model.model[str_param].fix()
                    
                elif (hasattr(v, '__iter__')) and (len(v) == 2):
#                    v_min, v_max = v[0], v[1]
#                    if('variance' in k) and self.BO.normalize_Y:
#                        scale = np.square(self.BO.args_norm['std'])                        
#                    
                    self.BO.model.model[str_param].constrain_bounded(v[0], v[1], warning = False)
                else:
                    self.BO.model.model[str_param] = v
                
    def save_hp_values(self):
        self.hp_vals = {name:[val] for name, val in zip(self.BO.model.model.parameter_names(),
                        self.BO.model.model.param_array)}
        
    def restore_hp_values(self):
        if(hasattr(self, 'hp_vals')):
            self.constrain_hp(self.hp_vals)
        else:
            logger.warning("no hp_vals to restore")
            
def filter_X(X, Y,  x_best, nb):
    d_to_best = dist_to(X, x_best)
    return X[np.argsort(d_to_best)[:nb]], Y[np.argsort(d_to_best)[:nb]]

#=======================================#
# HELPER FUNCTIONS
#=======================================#
class Warp():
    """
    warb by a function f(t) = K (1- exp^-a t)
    """
    def __init__(self, alpha = 1, T = 1):
        self.alpha = alpha
        self.T = T
        if self.alpha == 0:
            self.K = 1
        else:
            self.K = 1/(1-np.exp(-self.alpha*self.T))
        
    def __call__(self, t):
        if self.alpha == 0:
            res = t
        else:
            res = self.K * (1 - np.exp(-self.alpha * t))
        return res

def get_proba_after_evol(U, phi_0, list_e, n_ro):
     phi_f = U * phi_0
     final_expect = [real_with_test(e.matrix_element(phi_f, phi_f)) for e in list_e]
     proba = np.array([(1 + e)/2 for e in final_expect])
     proba = corrupt_readout_qubit(proba, n_ro)
     assert np.any(proba < 1 + 1e-5), "proba > 1: {}".format(proba)
     assert np.any(proba > -1e-5), "proba > 1: {}".format(proba)
     proba = np.clip(proba, 0, 1)
     return proba
 
def get_measurement_from_proba(proba, N, aggregate, p_tgt=None):
    if (N == np.inf): 
        res = proba 
        n_call = 1
    else:
        res = rdm.binomial(N, proba) / N
        n_call = len(res)*N
    if aggregate == 'fid':
        res = 1/8 * (1 + np.dot(2*p_tgt-1,2*res-1))
    elif aggregate == 'close':
        res = np.average(np.abs(p_tgt-res))
    return res, n_call

def make_iter_if_not_iter(x, nb_elements):
    if  hasattr(x, '__iter__'):
        assert len(x) == nb_elements, "Mismatch of len = {} vs {} ".format(len(x), nb_elements)
        x_iter = x
    else:
        x_iter = [x] * nb_elements
    return x_iter

def make_list_if_not_list(x, nb_elements):
    if  type(x) == list:
        assert len(x) == nb_elements, "Mismatch of len = {} vs {} ".format(len(x), nb_elements)
        x_iter = x
    else:
        x_iter = [x] * nb_elements
    return x_iter

def real_with_test(x):
    assert np.allclose(np.imag(x), 0.), ""
    return np.real(x)

def _get_some_res(field, list_res, criterion = np.max):
    """ Pick some res amongst a list of res according to some criterion"""
    field_values = np.array([np.atleast_1d(res.get(field))[0] for res in list_res])
    mask_none = np.array([f is not None for f in field_values])
    f_mask = field_values[mask_none]
    if (len(f_mask) > 0):
        field_best = criterion(f_mask)
        res = [list_res[n] for n, f in enumerate(f_mask) if f == field_best]
    else:
        res = [np.nan]
    return res

def _stats_one_field(field, list_res, dico_output = False, index=0):
    #TODO: Think better about the case when there are arrays of results
    field_values = np.array([res.get(field) for res in list_res])
    mask_none = np.array([f is not None for f in field_values])
    f = field_values[mask_none]
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

def probit(p):
    return np.clip(np.sqrt(2) * erfinv(2 * p -1), -2.3263478740408408, 2.3263478740408408)

def closest_to(X, x_tgt):
    return np.argsort(dist_to(X, x_tgt))

def dist_to(X, x_tgt):
    return np.linalg.norm(X-x_tgt, axis = 1) / np.shape(X)[1]

def corrupt_readout_qubit(proba, n_ro=0):
    """ classical readout noise with noise[i] = 1-p(i|i) with i in {0,1}"""
    proba_corrupted = np.array(proba)
    if n_ro>0:
        proba_corrupted = proba_corrupted * (1-n_ro) + (1-proba_corrupted) * n_ro
    return proba_corrupted 

def workaroundQObj(string):
    i_start = 0
    res = ''
    counter = 0
    while (string.find("Quantumobject", i_start)>0) and (counter < 100):
        counter += 1
        beg = string.find("Quantumobject", i_start)
        end = string.find("Qobjdata=", beg)
        res += string[i_start:beg]
        beg_data = end + 9
        end_data = string.find(']]', beg_data)+2
        res += 'np.array('
        res += string[beg_data:end_data].replace("][", "],[")
        res += ')'
        i_start = end_data 
    res += string[i_start:]
    res = res.replace('ObsAr','np.array')
    res = res.replace('nan', 'np.nan')
    return res



def read_string_rv(string, function = False, size = None):
    """ Generate a random variable according to info contain in the string
    if function = True return a function generating the rv rather than a single 
    instance of the rv
    """
    bits = string.split("_")
    if (bits[0] == 'fixed'):
        value = bits[1]
        if size is None:
            func = lambda : value
        else:
            func = lambda : np.ones(size=size) * value
        
    elif (bits[0] == 'uniform'):
        low, high = float(bits[1]), float(bits[2])
        func = lambda : np.random.uniform(low = low, high=high, size = size)
    elif (bits[0] == 'normal'):
        loc, scale = float(bits[1]), float(bits[2])
        func = lambda : np.random.normal(loc = loc, scale=scale, size = size)
    else:
        raise NotImplementedError()
    if function:
        res = func
    else:
        res = func()
    return res

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
            BatchSFMI.parse_and_save_meta_config(
                    file_input, output_folder = output_f, update_rules = True)

        elif(type_task == "run_one_config"):
            batch = BatchSFMI(file_input)
            batch.run_procedures(save_freq = 1)

        elif(type_task == "run_meta_config"):
            update_rules = str(sys.argv[3]) if(len(sys.argv) == 4) else True
            batch = BatchSFMI.from_meta_config(file_input, update_rules = update_rules)
            batch.run_procedures(save_freq = 1)

        else:
            logger.error("first argument not recognized")
   

        



