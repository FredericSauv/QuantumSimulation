#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:03:37 2017

@author: fs
"""
from .. import Helper as ut
from . import RandomGenerator as rdmg

import scipy.optimize as optim
import numpy as np
import pdb as pdb
import importlib as ilib


if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim.RandomGenerator import  RandomGenerator as rdm
    from QuantumSimulation.Utility.Optim.MP import MPCapability as mp

else:
    from ..Utility import Helper as ut
    from ..Utility.Optim.RandomGenerator import RandomGenerator as rdm
    from ..Utility.Optim.MP import MPCapability as mp
ilib.reload(ut)


class learner_base:
    """ Learner base class: follow (loosely the way scipy optimizer are implemented):
    a learner is build then when called for a model 

    learner ---  params_next  ---> model
    learner <---  fom  --- model
    # TODO: Implement RandomSearch // BFGS // GRID
    """
    
    # mandatory params to initialise the learner
    _LIST_PARAMS_LEARNER = {}
    _LIST_PARAMS_LEARNER['algo'] =  "Type of optimization performed - "
               "DE' differential evolution, 'NM' nedler mead, 'BH' basin hopping"
               "'NoOPtim' (still can be some testing)"
    _LIST_PARAMS_LEARNER['nb_params'] =  "Nb of parameters to be optim"
    
    # optional params     
    _LIST_PARAMS_LEARNER_OPT = {}
    _LIST_PARAMS_LEARNER_OPT['params_init'] =  ("How to initialize the parameters - "
        "Mandatory if NM / Simple / BH algos are used", None)
    _LIST_PARAMS_LEARNER_OPT['params_bounds'] =  ("enforce parameters bounds.. if not" 
        "assessed from the model object", None)
    _LIST_PARAMS_LEARNER_OPT['rdm_obj'] = ("random_object", None) 
    _LIST_PARAMS_LEARNER_OPT['rdm_obj'] = ("multiprocessing_object", None) 

    # <dico>{<str>'name_algo': (<dico> default_hyperparameters, <method> method)> 
    _ALGO_INFOS =
    {
        'NM':({'disp':True, 'maxiter':200, 'ftol':1e-6, 'maxfev':200, 'adaptative':False}
            , self._run_NM),
        
        'NOOPTIM':({}, self._run_NOOPTIM),
        
        'GP' = ({'disp':True, 'acq':'ei', 'kappa':5.0, 'maxiter':150,
                'verbose':False, 'kernel':'matern2.5', 
               'whiteNoise':0.1, 'scalingKer':0.1, 'flag_MP':False, 
               'acq_iter':250, 'n_warmup':100000}, self._RUN_GP),
        
        'DE' = ({'disp':True, 'maxiter':50, 'popsize':10, 'tol':0.01}, self._RUN_DE)     
    }
 

    def __init__(self, model, init_obj, bounds_obj = None, **params_learner):
        self.rdm_gen = params_learner.get('rdm_obj')
        self.mp = args_model.get('mp_obj')
        self._setup_learner_options(model, init_obj, bounds_obj, **params_learning)


    def _setup_learner_options(self, init_obj, bounds_obj, **params_learner):
        """ save main characteristics, create the random_generator used, 
        fetch default parameters depending on the algo
        """
        self._backup_initparams = params_learner # store initial params just in case  
        default = self._DEF_HYPERPARAMS[params_learner['algo']][0] # default hyper parameters       
        opt_l = ut.merge_dico(default, params_learner, update_type = 4)     
        opt_l.update({'algo': params_learner['algo']})
        opt_l.update({'nb_params':self.get_nb_params(model, params_learner.get('params_nb'))})
        opt_l.update({'bounds_params':self.get_boundaries_params(model, bounds_obj, opt_l)})
        opt_l.update({'init_params':self.get_init_params(model, init_obj, **opt_l)})
        self._params_learner = opt_l

    def _get_nb_params(self, model, nb_params):
        """ get boundaries from different ways so far either by passing a number
        if not get it from the model.. """
        if nb_params is  None:
            nb_params = model.n_params
        return nb_params


    def _get_boundaries_params(self, model, bds_obj, option):
        """ get boundaries from different ways: if model"""
        nb_params = options['nb_params']
        if(bds_obj is None):
            if(hasattr (model, 'params_bounds')):
                pdb.set_trace()
                bounds = model.params_bounds
            else:
                None

        elif(isinstance(bds_obj, tuple)):
            bounds = [bds_obj for range(n_params)]

        else:
            bounds = np.array(model.params_bounds)    
            expected_dim = (n_params, 2)
            assert (dim(bounds) == expected_dim), "len boundaries = {1}, nb_params"
                    " ={2}".format(dim(bounds), expected_dim)
        return bounds


    def _gen_init_params(self, init_obj, **args_extra):
            
        """ Different algotithms, require different type of init.. that's why they are 
        splitted here..
        TODO: try to unify everything""" 
        if init_obj is None:
            return None

        algo = options['algo']
        if(algo == 'BO'):
            init = self._init_params_BO(init_obj, **args_extra)
        elif(algo == 'NM'):
            init = self._init_params_NM(init_obj, **args_extra)
        elif(algo == 'DE'):
            init = self._init_params_DE(init_obj, **args_extra)
        else:
            raise NotImplementedError("Should implement adhoc _init_params_{}".format(algo))
        return init
          

    def __call__(self, **args_call):
        """ effectively run the optimization. Rely on a <model> which is an object which should
        at least have a method __call__(params) and may have attributes _info_simulations.
        Run the adequate optimization, save the results"""
        options = self.options_learner
        algo2run = self.ALGO_AVAILABLE[options['algo']]
        res_optim_raw = algo2run(model, options, **args_call)
        
        ## populate res
        res = {'name_algo': self._NAME_ALGO}
        res['params'] = res_optim_raw['x']
        res['fun'] = res_optim_raw['fun']                    
        res['nfev'] = resOptim.get('nfev') 
        res['nit'] = resOptim.get('nit')
        res['sucess'] = resOptim.get('sucess')
        res['init'] = init
        res['bounds'] = bounds        
        if(hasattr(model, '_track_calls')):
            res.update({'extra_'+k: v for k,v in model._info_simulations.items()})

        self._res_optim_raw = res_optim_raw
        self.mp.close_mp
        return res

    def Save(self, path_file = None, fun_string_gen = ut.custom_repr):
        """Save results as a text file """
        string = fun_string_gen(self._res_optim)
        ut.save_str(path_file, string)



    @property
    def rdm_gen(self):
        return self._rdmgen 
    
    @property.setter
    def rdm_gen(self, rdm_obj)
        self._rdmgen = rdm.init_random_generator(rdm_obj)

    @property
    def mp(self):
        return self._mp 
    
    @property.setter
    def mp(self, mp_obj)
        self._mp = mp.init_mp(rdm_obj)


    @property
    def options_learner(self):
        """ optimizer hyperparameters e.g. tolerance, n_iter_max, etc...
        To add new options: has to add entries in _ALGO_INFOS and make sure
        these are used in _RUN_ALGO, to change them from the default value
        a new value should be provided when initializing the learner in **args_learner
        """
        return self._params_learner



    # ------------------------------------------------------
    # optimization procedures
    # ------------------------------------------------------
    def _run_NM(self, model, options, **args_call):
        """ Implementation of a Nedler Meat optimization  
        
        scipy.optimize.minimize(fun, x0, args=(), method='Nelder-Mead', tol=None, 
        callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 
        'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 
        'fatol': 0.0001, 'adaptive': False})"""
        if(len(np.shape(init)) == 2):
            options['initial_simplex'] = init
            init = init[0,:]
        resultOptim = optim.minimize(model, args = args_call, init, method='Nelder-Mead', options = options)
        return resultOptim

    def _run_DE(self, model, options, **args_call):
        """ Implementation of a differential evolution optimizer (need bounds)

        scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin',
        maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, 
        callback=None, disp=False, polish=True, init='latinhypercube')
        """
        options['seed'] = self.rdm_gen
        cost = lambda x : model(x, **args_call)
        resultOptim = sco.differential_evolution(cost, **options)
        return resultOptim

    def _run_NOOPTIM(self, model, options, **args_call):
        """ Simple run of the simulator on a set of parameters (no Optim involved,
            one call of the simulator)
        """
        #TODO: will probably evolved to contain more details abt the optim 
        simRun = model(options['init_params'], **args_call)
        resultTest = {'x': params, 'fun': simRun, 'nfev':1, 'nit':1, 'sucess':True}
        return resultTest

    def _run_BO(self, model, options, **args_call):
        """ Run a bayesian optimization using the library Bayesian Optimization
            (https://github.com/fmfn/BayesianOptimization) built on 
        """        
        #Init BO
        nb_params = options['n_params']
        name_params = [str(i) for i in range(nb_params)]
        bounds_bo = {name_params[i]: options['bounds_params'][i] for i in range(nb_params)}
        options['random_state'] = self.rdm_gen
        options['mp_obj'] = self.mp_obj
        options['name_params'] = name_params
        cost = lambda x : model(x, **args_call)
        bo = BayesianOptimization(cost, bounds_bo, **options)

        #Initialization phase
        init = options['init_params']
        if(ut.is_int(init)):
            nb_points_init = init
            bo.maximize(init_points = nb_points_init, n_iter=0, acq=acq, kappa=0) 
        else:
            nb_points_init = len(init)
            to_explore = {name_params[i]: [init_matrix[j][i] for j in range(nb_points_init)] for i in range(self._nb_params)}
            bo.explore(to_explore)

        
        # Exploration-Exploitation phase
        kappa = self._process_kappa(options['kappa'])        
        bo_args = {'n_iter':options['maxiter'], 'acq': options['acq'], 'kappa':kappa}    
        bo.maximize(init_points=0, **bo_args)


        # Exploitation phase
        bo_args_Maximize = {'n_iter':15, 'acq': 'ucb', 'kappa':0.0001} 
        bo.maximize(init_points=0, **bo_args_Maximize)

        
        # Close pool of processors used (if it exists)
        nb_workers_used = bo._nb_workers
        nb_cpus_seen = bo._nb_cpus
        bo.close_mp_pool()
       
        # generate results      
        index_max = np.argmax(bo.Y)
        optim_params = bo.X[index_max]
        optim_value = cost(optim_params)
        niter = bo_args['n_iter']
        nfev = niter + nb_points_init
        resultTest = {'x': optim_params, 'fun': optim_value, 'nfev':nfev, 'nit':niter, 'sucess':True}
        resultTest['gp_kernel_init'] = str(bo.gp.kernel)
        resultTest['gp_kernel_optim'] = str(bo.gp.kernel_)
        resultTest['gp'] = bo.gp.kernel_
        resultTest['nb_processes'] = nb_workers_used
        resultTest['nb_cpus'] = nb_cpus_seen
        return resultTest


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

    def _init_params_DE(self, init_obj, **args_init):
        """ just pass through """
        return init_obj

    def _init_params_NM(self, init_obj, **args_init):
        """Methods:
            + 'zero' for each params 0 <(n_params) np.array>
            + 'uniform', 'normal', 'lhs' <(n_params+1, n_params ) np.array>
            + 'nmguess_guess_step0_stepNon0': <(n_params+1, n_params ) np.array>
            Init produce N+1 vertices arround a guess (adapted fm Matlab routine)
            guess[0] = guess, guess[i] = guess + e * n_i, with e= step0/stepNon0 
            (dep on value of guess for a particular element) and n_i is one basis unit vector)
            e.g. nmguess_[3,0,1]_0.5_1 >> [[3,0,1], [4, 0, 1], [3,0.5,1], [3,0,2]]
        """ 
        n_params = args_init['nb_params']            
        if(ut.is_str(init_ob))
            args = init_obj.split("_")
            if(args[0] == 'zero'):
                init_args = np.zeros_like(n_params)
            
            elif(args[0] in ['uniform','normal', 'lhs']):  
                dim = [n_params + 1, n_params]
                init_args = self.rdm_gen.gen_rdmnb_from_string(method, dim)                    

            elif(args[0] == 'nmguess'):
                assert (len(args)==4), 'nmguess, not the right format'
                gues, step0, stepNon0 = args[1], args[2], args[3]
                init_args = np.zeros_like([nb_params + 1, nb_params])
                init_args[0, :] = guess_vector
                for i in range(nb_params):
                    perturb = np.zeros(nb_params)
                    if(guess[i] == 0):
                        perturb[i] = step0
                    else:
                        perturb[i] = stepNon0 
                    init_args[(i+1), :] = init_args[0,:] + perturb
                return init_args            
                print(init_args)
        else:
            init_args = np.array(init_obj)
            
        return init_args

    def _init_params_BO(self, init_obj, **args_init):
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
        nb_params = args_init['nb_params']
        bounds = args_init['bounds_params']

        if (init_obj is None):
            init_args = nb_params *2
            
        elif ut.is_int(init_obj):
            init_args = init_type
            
        elif ut.is_string(init_obj):
            bits = init_obj.split("_",1)
            nb_points_init = int(bits[0])
            if(bits[1] == 'lhs'):
                size_pop = [nb_points_init, nb_params]
                limits = np.array(bounds).T
                init_args = self.rdm_gen.init_population_lhs(size_pop, limits)
            else:
                distrib_one_param = [bits[1]+'_'+ bounds[i][0] + '_' + bounds[i][1] for i in range(nb_params)]
                init_matrix = [self.rdm_gen.gen_rdmfunc_from_string(d, dim = np_points_init) for d in distrib_one_param]
                init_args = np.array(init_matrix).T

        else ut.is_string(init_obj):
            init_args = init_obj
        return init_args



    
    def print_default_hyperparams(self):
        raise NotImplementedError

    #TODO: tofill  
    def help(self):
        raise NotImplementedError






