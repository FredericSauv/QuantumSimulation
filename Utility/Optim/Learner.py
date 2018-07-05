#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:03:37 2017

@author: fs
"""
import scipy.optimize as sco
import numpy as np
import pdb as pdb
import importlib as ilib
import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
import GPy
import GPyOpt
import copy
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim.RandomGenerator import  RandomGenerator as rdm
    from QuantumSimulation.Utility.Optim.MP import MPCapability as mp
    from QuantumSimulation.Utility.BayesOptim import BayesianOptimization

else:
    from ..BayesOptim import BayesianOptimization 
    from .. import Helper as ut
    from ..Optim.RandomGenerator import RandomGenerator as rdm
    from ..Optim.MP import MPCapability as mp
ilib.reload(ut)


class learner_base:
    """ Learner base class: follow (loosely the way scipy optimizer are implemented):
    a learner is build then when called for a model 

    learner ---  params_next  ---> model
    learner <---  fom  --- model
    TODO: BO2 (based on GPyOpt)
    TODO: Implement RandomSearch // BFGS // GRID
    """
    
    # mandatory params to initialise the learner
    _LIST_PARAMS_LEARNER = {}
    _LIST_PARAMS_LEARNER['algo'] =  "Type of optimization performed - DE' differential evolution, 'NM' nedler mead, 'BH' basin hopping 'NoOPtim' (still can be some testing)"
    
    # optional params     
    _LIST_PARAMS_LEARNER_OPT = {}
    _LIST_PARAMS_LEARNER_OPT['rdm_obj'] = ("random_object", None) 
    _LIST_PARAMS_LEARNER_OPT['mp_obj'] = ("multiprocess0ing_object", None) 


    def __init__(self, **params_learner):
        self.rdm_gen = params_learner.get('rdm_obj')
        self.mp = params_learner.get('mp_obj')
        self._setup_learner_options(**params_learner)
        
    def _setup_learner_options(self, **params_learner):
        """ To be implemented in the subclasses"""
        raise NotImplementedError()
        
          
    def __call__(self, **args_call):
        """ effectively run the optimization. Rely on a <model> which is an object which should
        at least have a method __call__(params) and may have attributes _info_simulations.
        Run the adequate optimization, save the results"""
        raise NotImplementedError()

    def Save(self, path_file = None, fun_string_gen = ut.custom_repr):
        """Save results as a text file """
        string = fun_string_gen(self._res_optim)
        ut.save_str(path_file, string)

    @property
    def rdm_gen(self):
        return self._rdmgen 
    
    @rdm_gen.setter
    def rdm_gen(self, rdm_obj):
        self._rdmgen = rdm.init_random_generator(rdm_obj)

    @property
    def mp(self):
        return self._mp 
    
    @mp.setter
    def mp(self, mp_obj):
        self._mp = mp.init_mp(mp_obj)


    @property
    def options_learner(self):
        """ optimizer hyperparameters e.g. tolerance, n_iter_max, etc...
        To add new options: has to add entries in _ALGO_INFOS and make sure
        these are used in _RUN_ALGO, to change them from the default value
        a new value should be provided when initializing the learner in **args_learner
        """
        return self._params_learner

    @options_learner.setter
    def options_learner(self, options):
        """ optimizer hyperparameters e.g. tolerance, n_iter_max, etc...
        To add new options: has to add entries in _ALGO_INFOS and make sure
        these are used in _RUN_ALGO, to change them from the default valuemeasurement
        a new value should be provided when initializing the learner in **args_learner
        """
        self._params_learner = options


class learner_Opt(learner_base):
    # mandatory params to initialise the learner
    _LIST_PARAMS_LEARNER = learner_base._LIST_PARAMS_LEARNER
    _LIST_PARAMS_LEARNER['nb_params'] =  "Nb of parameters to be optim"
    _LIST_PARAMS_LEARNER['model'] =  "a model which is at least callable"
    
    # optional params     
    _LIST_PARAMS_LEARNER_OPT = learner_base._LIST_PARAMS_LEARNER_OPT
    _LIST_PARAMS_LEARNER_OPT['params_init'] =  ("How to initialize the parameters - Mandatory if NM / Simple / BH algos are used", None)
    _LIST_PARAMS_LEARNER_OPT['params_bounds'] =  ("enforce parameters bounds.. if not assessed from the model object", None)


    def __init__(self, model, **params_learner):
        # <dico>{<str>'name_algo': (<dico> default_hyperparameters, <method> method)> 
        self._ALGO_INFOS ={
        'NM':({'disp':True, 'maxiter':200, 'ftol':1e-6, 'maxfev':200, 'adaptative':False}, self._run_NM),        
        'NOOPTIM':({}, self._run_NOOPTIM), 
        'BO2':({'disp':True, 'acq':'EI', 'maxiter':50,'verbose':False, 
               'kernel':'matern2.5', 'flag_MP':False, 'num_cores':1, 'custom_model': False,
               'acq_opt_type':'lbfgs', 'initial_design_type':'random'}, self._run_BO2),
        'BO':({'disp':True, 'acq':'ei', 'kappa':5.0, 'maxiter':50,'verbose':False, 'kernel':'matern2.5', 
               'whiteNoise':0.1, 'scalingKer':0.1, 'flag_MP':False, 'acq_iter':50, 'n_warmup':10000}, self._run_BO),
        'DE':({'disp':True, 'maxiter':50, 'popsize':10, 'tol':0.01}, self._run_DE)}
        
        learner_base.__init__(self, model=model, **params_learner)
        



    def _setup_learner_options(self, model, **params_learner):
        """ save main characteristics, fetch default parameters depending on the algo
        generate init and bounds of the parameters (most tricky part)
        """
        self._backup_initparams = params_learner # store initial params just in case
        default = self._ALGO_INFOS[params_learner['algo']][0] # default hyper parameters       
        opt_l = ut.merge_dico(default, params_learner, update_type = 4)     
        opt_l.update({'model': model})
        opt_l.update({'algo': params_learner['algo']})
        opt_l.update({'nb_params':model.n_params})
        opt_l.update({'bounds_obj':params_learner.get('bounds_obj')})
        opt_l.update({'bounds_params':self._gen_boundaries_params(**opt_l)})
        opt_l.update({'init_obj':params_learner.get('init_obj')})
        opt_l.update({'init_params':self._gen_init_params(**opt_l)})
        opt_l['rdm_gen'] = self.rdm_gen
        opt_l['mp_obj'] = self.mp
        self.options_learner = opt_l

    def _gen_boundaries_params(self, **args_optim):
        """ get boundaries from different ways: if model"""
        nb_params = args_optim['nb_params']
        bounds_obj = args_optim.get('bounds_obj')
        model = args_optim['model']
        if(bounds_obj is None):
            if(hasattr (model, 'params_bounds')):
                bounds = model.params_bounds
            else:
                None
        elif(isinstance(bounds_obj, tuple)):
            bounds = [bounds_obj for _ in range(nb_params)]
        else:
            bounds = np.array(model.params_bounds)    
            expected_dim = (nb_params, 2)
            real_dim = np.shape(bounds)
            assert (real_dim == expected_dim), "len boundaries = {1}, nb_params={2}".format(real_dim, expected_dim)
        return bounds


    def _gen_init_params(self, **args_optim):            
        """ Different algotithms, require different type of init.. that's why they are 
        splitted here..
        TODO: try to unify everything""" 
        algo = args_optim['algo']
        if(algo in ['BO', 'BO2']):
            init = self._init_params_BO(**args_optim)
        elif(algo == 'NM'):
            init = self._init_params_NM(**args_optim)
        elif(algo == 'DE'):
            init = self._init_params_DE(**args_optim)
        else:
            raise NotImplementedError("Should implement adhoc _init_params_{}".format(algo))
        return init


    def __call__(self, **args_call):
        """ effectively run the optimization. Rely on a <model> which is an object which should
        at least have a method __call__(params) and may have attributes _info_simulations.
        Run the adequate optimization, save the results"""
        options = self.options_learner
        algo_run = self._ALGO_INFOS[options['algo']][1]
        res_optim_raw = algo_run(options, **args_call)

        res_optim_duplicate = copy.copy(res_optim_raw)
        ## populate res
        res = {'name_algo': options['algo']}
        res['params'] = res_optim_duplicate.pop('x')
        res['fun'] = res_optim_duplicate.pop('fun')                    
        res['nfev'] = res_optim_duplicate.pop('nfev', None) 
        res['nit'] = res_optim_duplicate.pop('nit', None)
        res['success'] = res_optim_duplicate.pop('success', 'None')
        if(options['algo'][:2] == 'BO'):
            res['opt_more'] = res_optim_duplicate
        res['init'] = self.options_learner['init_params']
        res['bounds'] = self.options_learner['bounds_params']
        model = options['model']
        if(hasattr(model, '_track_calls')):
            res.update({'extra_'+k: v for k,v in model._track_calls.items()})

        self._res_optim_raw = res_optim_raw
        self.mp.close_mp()
        return res


    # ------------------------------------------------------
    # optimization procedures
    # ------------------------------------------------------
    def _run_NM(self, options, **args_call):
        """ Implementation of a Nedler Meat optimization  
        
        scipy.optimize.minimize(fun, x0, args=(), method='Nelder-Mead', tol=None, 
        callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 
        'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 
        'fatol': 0.0001, 'adaptive': False})"""
        init = options['init_params']
        model = options['model']
        if(len(np.shape(init)) == 2):
            options['initial_simplex'] = init
            init = init[0,:]
            
        cost = lambda x:model(x, **args_call)
        resultOptim = sco.minimize(cost, x0 =init, args = () , method='Nelder-Mead', options = options)
        return resultOptim

    def _run_DE(self, options, **args_call):
        """ Implementation of a differential evolution optimizer (need bounds)

        scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin',
        maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, 
        callback=None, disp=False, polish=True, init='latinhypercube')
        """
        options_DE = {k:options[k] for k in self._ALGO_INFOS['DE'][0]}
        options_DE['seed'] = self.rdm_gen
        model = options['model']
        bounds = options['bounds_params']
        def cost(x):
            res = model(x, **args_call)
            return res
        resultOptim = sco.differential_evolution(cost, bounds, **options_DE)
        return resultOptim

    def _run_NOOPTIM(self, options, **args_call):
        """ IS IT USEFULL """
 
        init = options['init_params']
        model = options['model']
        simRun = model(init, **args_call)
        resultTest = {'x': init, 'fun': simRun, 'nfev':1, 'nit':1, 'sucess':True}
        return resultTest

    def _run_BO(self, options, **args_call):
        """ Run a bayesian optimization using the library Bayesian Optimization
            (https://github.com/fmfn/BayesianOptimization) built on 
        """        
        #Init BO
        model = options['model']
        nb_params = options['nb_params']
        name_params = [str(i) for i in range(nb_params)]
        bounds_bo = {name_params[i]: options['bounds_params'][i] for i in range(nb_params)}
        options['random_state'] = self.rdm_gen
        options['name_params'] = name_params
        cost = ut.ArrayToDicoInputWrapper(model)
        bo = BayesianOptimization(cost, bounds_bo, **options)
        acq = options['acq']
        
        #Initialization phase
        init = options['init_params']
        if(ut.is_int(init)):
            nb_points_init = init
            bo.maximize(init_points = nb_points_init, n_iter=0, acq=acq, kappa=0) 
        else:
            pdb.set_trace()
            nb_points_init = len(init)
            to_explore = {name_params[i]: [init[j][i] for j in range(nb_points_init)] for i in range(nb_params)}
            bo.explore(to_explore)

        # Exploration-Exploitation phase
        kappa = self._process_kappa(options)        
        bo_args = {'n_iter':options['maxiter'], 'acq': options['acq'], 'kappa':kappa}    
        bo.maximize(init_points=0, **bo_args)

        # Exploitation phase
        bo_args_Maximize = {'n_iter':15, 'acq': 'ucb', 'kappa':0.0001} 
        bo.maximize(init_points=0, **bo_args_Maximize)

        # generate results      
        index_max = np.argmax(bo.Y)
        optim_params = bo.X[index_max]
        optim_value = model(optim_params)
        niter = bo_args['n_iter']
        nfev = niter + nb_points_init
        resultTest = {'x': optim_params, 'fun': optim_value, 'nfev':nfev, 'nit':niter, 'sucess':True}
        resultTest['gp_kernel_init'] = str(bo.gp.kernel)
        resultTest['gp_kernel_optim'] = str(bo.gp.kernel_)
        resultTest['gp'] = bo.gp.kernel_
        resultTest['nb_processes'] = bo.mp.n_workers
        resultTest['nb_cpus'] = bo.mp.n_cpus
        
        # Close pool of processors used (if it exists)
        return resultTest

    def _run_BO2(self, options, **args_call):
        """ Bayesian optimization using GPYOpt 
        def __init__(self, f, domain = None, constraints = None, cost_withGradients = None, 
        model_type = 'GP', X = None, Y = None, initial_design_numdata = 5, 
        initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1,
        evaluator_type = 'sequential',batch_size = 1, num_cores = 1, verbosity=False, 
        verbosity_model = False, maximize=False, de_duplication=False, **kwargs)
        
        
        def run_optimization(self, maxiter = 0, max_time = np.inf, eps = 1e-8, 
        context = None, verbosity=False, save_models_parameters= True, 
        report_file = None, evaluations_file = None, models_file=None):
        ## NEW PARAMS
        acq_optim_type
        initial_design_type='random'
        
        """
        #Init BO
        model = options['model']

        nb_params = options['nb_params']
        name_params = [str(i) for i in range(nb_params)]
        options['name_params'] = name_params        
        bounds_bo = [{'name': name_params[i], 'type': 'continuous', 
                   'domain': options['bounds_params'][i]} for i in range(nb_params)]
        
        acq = options['acq'] # EI MPI LCB
        ker = options['kernel'] # matern52 /matern32
        acq_opt_type = options['acq_opt_type'] #lbfgs DIRECT CMA
        num_cores = options['num_cores'] # 1-N
        custom_model = options['custom_model']
        
        if(custom_model):
            if(ker == 'matern52'):
                
                kernel = eval('GPy.kern.Matern52(self.input_dim = nb_params)')
            elif(ker == 'matern32'):
                kernel = eval('GPy.kern.Matern32(self.input_dim = nb_params)')
            else:
                SystemError("Please implement the kernel here")                      
            model_GP = GPyOpt.models.GPModel(kernel = kernel)
        else:
            model_GP = None # Def matern52
        
        #Initialization phase
        init = options['init_params']
        initial_design_type = options['initial_design_type']
        def cost(params):
            return model(np.squeeze(params), **args_call)
        
        if(ut.is_int(init)):
            bo = GPyOpt.methods.BayesianOptimization(cost,bounds_bo, model = model_GP, 
                acquisition_type=acq, acquisition_optimizer_type = acq_opt_type, 
                num_cores = num_cores, initial_design_numdata = init, 
                initial_design_type =initial_design_type)
        
        else:
            X = init
            print('acquisition of {} points'.format(X.shape[0]))
            Y = np.array([cost(x) for x in X])
            bo = GPyOpt.methods.BayesianOptimization(cost,bounds_bo, model = model_GP, 
                acquisition_type=acq, acquisition_optimizer_type = acq_opt_type, 
                num_cores = num_cores, X = X, Y =Y)


        # Exploration-Exploitation phase
        maxiter = options['maxiter']
        bo.run_optimization(maxiter)
        

        # Exploitation phase
        #### TODO:
        bo.acquisition_type = 'User defined acquisition used.'
        bo_new = GPyOpt.methods.BayesianOptimization(cost, bounds_bo, model = model_GP, 
                acquisition_type = 'LCB', acquisition_optimizer_type = acq_opt_type, 
                num_cores = num_cores, X = bo.X, Y =bo.Y, exploration_weight =0.000001)
        bo_new.run_optimization(15)

        # generate results      
        optim_params = bo_new.x_opt
        optim_value_cputed = model(optim_params)
        optim_value = bo_new.fx_opt
        nfev = len(bo_new.X)
        resultTest = {'x': optim_params, 'fun': optim_value, 'fun_ver': optim_value_cputed, 'nfev':nfev, 'nit':nfev, 'sucess':True}
        resultTest['gp_kernel_optim_names'] = bo_new.model.model.parameter_names()
        resultTest['gp_kernel_optim_vals'] = bo_new.model.model.param_array
        resultTest['nb_processes'] = bo_new.num_cores
        resultTest['nb_cpus'] = self.mp.n_cpus
        resultTest['X_evol'] = bo_new.X
        resultTest['Y_evol'] = bo_new.Y
        resultTest['Y_best'] = bo_new.Y_best
        
        # Close pool of processors used (if it exists)
        return resultTest

        raise NotImplementedError()

    def _process_kappa(self, options_GP):
        """ static or dynamic kappa"""
        kappa = options_GP['kappa']
        niter = options_GP['maxiter']
        if(isinstance(kappa, float)):
            kappa = np.repeat(kappa, niter)
        else:
            bits = ut.splitString(kappa)
            if(bits[0] == 'linear'):
                kappa = float(bits[1]) * (1 - np.arange(niter) / (niter - 1))
            else:
                raise NotImplementedError()
        return kappa
        
    # ------------------------------------------------------
    # init and boundaries... a lot of custom cases ...
    # ------------------------------------------------------

    def _init_params_DE(self, **args_optim):
        """ just pass through """
        init_obj = args_optim['init_obj']
        return init_obj

    def _init_params_NM(self, **args_optim):
        """Methods:
            + 'zero' for each params 0 <(nb_params) np.array>
            + 'uniform', 'normal', 'lhs' <(n_bparams+1, n_params ) np.array>
            + 'nmguess_guess_step0_stepNon0': <(n_params+1, n_params ) np.array>
            Init produce N+1 vertices arround a guess (adapted fm Matlab routine)
            guess[0] = guess, guess[i] = guess + e * n_i, with e= step0/stepNon0 
            (dep on value of guess for a particular element) and n_i is one basis unit vector)
            e.g. nmguess_[3,0,1]_0.5_1 >> [[3,0,1], [4, 0, 1], [3,0.5,1], [3,0,2]]
        """ 
        init_obj = args_optim['init_obj']
        nb_params = args_optim['nb_params']            
        if(ut.is_str(init_obj)):
            args = init_obj.split("_")
            if(args[0] == 'zero'):
                init_args = np.zeros_like(nb_params)
            
            elif(args[0] in ['uniform','normal', 'lhs']):  
                dim = [nb_params + 1, nb_params]
                init_args = self.rdm_gen.gen_rdmnb_from_string(init_obj, dim)                    

            elif(args[0] == 'nmguess'):
                assert (len(args)==4), 'nmguess, not the right format'
                guess, step0, stepNon0 = args[1], args[2], args[3]
                init_args = np.zeros_like([nb_params + 1, nb_params])
                init_args[0, :] = guess
                for i in range(nb_params):
                    perturb = np.zeros(nb_params)
                    if(guess[i] == 0):
                        perturb[i] = step0
                    else:
                        perturb[i] = stepNon0 
                    init_args[(i+1), :] = init_args[0,:] + perturb
                return init_args            
                print(init_args)
        elif init_obj is None:
            print('params set up to zero-vect')
            init_args = np.zeros(nb_params)
        else:
            init_args = np.array(init_obj)
            
        return init_args

    def _init_params_BO(self, **args_optim):
        """ Provides different ways to initialize bo depending of the input type:
        <None> / <int>: returns an integer (nb of points to be eval)
            e.g. None >> 2 * nb_params 
        <str>: random_init based on a string. returns a <P x N np.array>
            P is the population size N number of parameters  
            e.g. '40_lhs' >> 40 points drawn by latin hypercube sampling
                 '50_uniform' >> 50 points drawn uniformly
                  range of each params is infered from bounds
        <P * N array>: passs it though 
        """
        init_obj = args_optim['init_obj']
        nb_params = args_optim['nb_params']
        bounds = args_optim['bounds_params']

        if (init_obj is None):
            init_args = nb_params *2
            
        elif ut.is_int(init_obj):
            init_args = init_obj
            
        elif ut.is_string(init_obj):
            bits = init_obj.split("_",1)
            nb_points_init = int(bits[0])
            if(bits[1] == 'lhs'):
                size_pop = [nb_points_init, nb_params]
                limits = np.array(bounds).T
                init_args = self.rdm_gen.init_population_lhs(size_pop, limits)
            else:
                distrib_one_param = [bits[1]+'_'+ bounds[i][0] + '_' + bounds[i][1] for i in range(nb_params)]
                init_matrix = [self.rdm_gen.gen_rdmfunc_from_string(d, dim = nb_points_init) for d in distrib_one_param]
                init_args = np.array(init_matrix).T

        else:
            init_args = init_obj
        return init_args

    #TODO: tofill  
    def print_default_hyperparams(self):
        raise NotImplementedError

    def help(self):
        raise NotImplementedError








#==============================================================================
# Custom_optim_func
#==============================================================================

from GPyOpt.acquisitions import AcquisitionLCB
class AcquisitionExploitation(AcquisitionLCB):
    """ LCB with kappa = 0"""
    def __init__(self, model, space, optimizer=None, cost_withGradients=None):
        AcquisitionLCB.__init__(self, model, space, optimizer, cost_withGradients,exploration_weight=0)


#==============================================================================
# Some testing
# 6 humps camel function minimal example
# x1 ∈ [-3, 3], x2 ∈ [-2, 2]
# Opt: array([ 0.08984198, -0.71265648]), 'fun': -1.0316284534898246
#==============================================================================
if __name__ == '__main__':
    test_camel = True
    test_paramFunc = False
    if(test_camel):
        class Camel_model():
            def __init__(self):
                self.n_params = 2
                self.params_bounds = [(-3, 3), (-2, 2)]
            
            def __call__(self, params, **args_call):
                
                assert(len(params) == 2), "bad dim: expected 2 instead {0}".format(len(params))
                x1 = params[0]
                x2 = params[1]
                res = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2
                print(res)
                return res
            
            
    
        camel = Camel_model()
        
        optim_args = {'algo': 'BO2', 'maxiter':25}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)
        
        
        optim_args = {'algo': 'DE', 'init_obj':[0,0]}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)
        
        optim_args = {'algo': 'BO', 'maxiter':25}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)


    if(test_paramFunc):
        from QuantumSimulation.Utility.Optim import pFunc_base as pf
        
        class Param_model():
            def __init__(self, fun):
                self.fun = fun
                self.params_bounds = fun.theta_bounds
                self.n_params = fun.n_theta
            
            def __call__(self, params):
                self.fun.theta = params
                res = self.fun(1.82)
                print(res)
                return res

        func = pf.FourierFunc(A = [0.0, 0.0], B =[0.0, 0.0], phi = [0.0, 0.0], c0 = 0.0,
                               Om =[np.pi, 2* np.pi], A_bounds = (-1.0, 1.0))
    
        p_model = Param_model(func)
                
        optim_args = {'algo': 'BO2', 'maxiter':25}
        optim = learner_Opt(model = p_model, **optim_args)
        resBO2 = optim()
        print(resBO2)
        
        optim_args = {'algo': 'DE'}
        optim = learner_Opt(model = p_model, **optim_args)
        resDE = optim()
        print(resDE)
        
        optim_args = {'algo': 'BO', 'maxiter':25}
        optim = learner_Opt(model = p_model, **optim_args)
        resBO = optim()
        print(resBO)
        

        