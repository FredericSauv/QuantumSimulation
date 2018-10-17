#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:03:37 2017

@author: fs
"""
import logging
logger = logging.getLogger(__name__)
import scipy.optimize as sco
import numpy as np
import sys, copy, time, pdb
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
sys.path.insert(0, '/Users/frederic/Desktop/GPyOpt') #Laptop's loc
import GPyOpt


if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Misc.RandomGenerator import  RandomGenerator as rdm
    from QuantumSimulation.Utility.Misc.MP import MPCapability as mp
    from QuantumSimulation.Utility.BayesOptim import BayesianOptimization

else:
    from ..BayesOptim import BayesianOptimization 
    from .. import Helper as ut
    from ..Misc.RandomGenerator import RandomGenerator as rdm
    from ..Misc.MP import MPCapability as mp


class learner_base:
    """ Learner base class: follow (loosely the way scipy optimizer are implemented):
    a learner is build then when called for a model (instead of a function)

    learner ---  params_next  ---> model
    learner <---  fom  --- model
    TODO: Implement RandomSearch // BFGS // GRID
    TODO: CLEAN
    TODO: rethink structure of the res / rethink the use creation of default arguments, 
    arguments used for optim put everything as tuple??
    
    """
    
    # mandatory params to initialise the learner
    _LIST_PARAMS_LEARNER = {}
    _LIST_PARAMS_LEARNER['algo'] =  "Type of optimization performed - 'DE' differential evolution, 'NM' nedler mead, 'BO2' Bayesian Optim, 'BO' depreciated"
    
    # optional params     
    _LIST_PARAMS_LEARNER_OPT = {}
    _LIST_PARAMS_LEARNER_OPT['rdm_obj'] = ("random_object", None) 
    _LIST_PARAMS_LEARNER_OPT['mp_obj'] = ("multiprocessing_object", None) 


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
        # used in such a way that only parameters appearing here can be used
        self._ALGO_INFOS ={
        'NM':({'disp':True, 
               'maxiter':200, 
               'ftol':1e-6, 
               'maxfev':200, 
               'adaptative':False, 
               'max_time':None},
              self._run_NM),     
               
        'NOOPTIM':({}, 
              self._run_NOOPTIM), 
        
        'BO2':({'disp':True, 
                'acq':'EI',                         # Acquisition fonction type
                'maxiter':50,                       # Number of iterations for the BO
                'verbose':False,                    # print extra information
                'model_type':'GP',                  # Type of Guassian process GP, GP_
                'kernel':None,                      # Type of Kernel for the GP
                'flag_MP':False,                    # Use of Multi processing
                'num_cores':1,                      # Number of core to use
                'acq_opt_type':'lbfgs',             # Optimizer to use when maximizing the acqu function
                'initial_design_type':'latin',      # Choice of the random 
                'optim_num_anchor':50,              # Used when minimizing the acq function
                'optim_num_samples':100000,         # Used when minimizing the acq function
                'acquisition_jitter':0.001, 
                'max_time':np.inf,
                'acquisition_weight':2, 
                'exploit_steps':15, 
                'batch_method':'sequential',
                'batch_size':1, 
                'num_inducing':10,
                'ARD':False,
                'to_transfer':None,
                'acquisition_weight_lindec':False,
                'constraints':None,                 # Constraints on the parameter space
                'likelihood':None},                 # Likelihood to use (by default 'Gaussian', could be 'Bernouilli', 'Binomial_10')
               self._run_BO2),
                
        'BO':({'disp':True, 'acq':'ei', 'kappa':5.0, 'maxiter':50,'verbose':False, 'kernel':'matern2.5', 
               'whiteNoise':0.1, 'scalingKer':0.1, 'flag_MP':False, 'gp_acq_iter':50, 'gp_n_warmup':10000},
                self._run_BO),
        
        'DE':({'disp':True, 
               'maxiter':500, 
               'popsize':10, 
               'tol':0.01, 
               'max_time':None}, 
              self._run_DE)}
        
        learner_base.__init__(self, model=model, **params_learner)
        



    def _setup_learner_options(self, model, **params_learner):
        """ save main characteristics, fetch default parameters depending on the algo
        generate init and bounds of the parameters (most tricky part)
        """
        self._backup_initparams = params_learner # store initial params just in case
        default = self._ALGO_INFOS[params_learner['algo']][0] # default hyper parameters       
        opt_l = ut.merge_dico(default, params_learner, update_type = 4)     
        opt_l.update({'model': model,
                      'algo': params_learner['algo'], 
                      'nb_params':model.n_params,
                      'bounds_obj':params_learner.get('bounds_obj'),
                      'init_obj':params_learner.get('init_obj'),
                      'rdm_gen':self.rdm_gen, 'mp_obj':self.mp})
        opt_l.update({'bounds_params':self._gen_boundaries_params(**opt_l)})
        opt_l.update({'init_params':self._gen_init_params(**opt_l)})
            
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
        if (args_call.pop('debug', False)):
            pdb.set_trace()
        options = self.options_learner
        algo_run = self._ALGO_INFOS[options['algo']][1]
        res_optim_raw = algo_run(options, **args_call)

        res_optim_duplicate = copy.copy(res_optim_raw)
        ## populate res
        res = {'name_algo': options['algo']}
        res['params'] = res_optim_duplicate.pop('x')
        res['params_exp'] = res_optim_duplicate.pop('x_exp', None)
        res['fun'] = res_optim_duplicate.pop('fun')                    
        res['nfev'] = res_optim_duplicate.pop('nfev', None) 
        res['nit'] = res_optim_duplicate.pop('nit', None)
        res['success'] = res_optim_duplicate.pop('success', 'None')
        if(options['algo'][:2] == 'BO'):
            res['opt_more'] = res_optim_duplicate
        res['init'] = self.options_learner['init_params']
        res['bounds'] = self.options_learner['bounds_params']
        res['maxtime'] = res_optim_raw.get('maxtime', False)
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
        
        # workaround doesn't work here
        if(options.get('max_time', None) is not None):
            _ = options.pop('max_time')
            logger.warning('NM: max_time is not used')
            callback = None
        else:
            callback = None
        resultOptim = sco.minimize(cost, x0 =init, args = () , method='Nelder-Mead', 
                                   options = options, callback = callback)
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
        
        #workaround to terminate optims when a certain time has been reached
        # use a callback with two mock arguments 
        if(options_DE.get('max_time', None) is not None):
            time_limit = time.time() + int(options_DE.pop('max_time'))
            options_DE['callback'] = lambda x, convergence: time.time() > time_limit
        
        resultOptim = sco.differential_evolution(cost, bounds, **options_DE)
        if(options_DE.get('callback') is not None):
            resultOptim['maxtime'] = options_DE.get('callback')(None, None)

        return resultOptim

    def _run_NOOPTIM(self, options, **args_call):
        """ IS IT USEFULL?? """
        init = options['init_params']
        model = options['model']
        simRun = model(init, **args_call)
        resultTest = {'x': init, 'fun': simRun, 'nfev':1, 'nit':1, 'sucess':True}
        return resultTest

    def _run_BO(self, options, **args_call):
        """ DEPRECIATED
        Run a bayesian optimization using the library Bayesian Optimization
            (https://github.com/fmfn/BayesianOptimization) built on 
        """        
        #Init BO
        model = options['model']
        nb_params = options['nb_params']
        name_params = [str(i) for i in range(nb_params)]
        bounds_bo = {name_params[i]: options['bounds_params'][i] for i in range(nb_params)}
        options['random_state'] = self.rdm_gen
        options['name_params'] = name_params
        cost = ut.ArrayToDicoInputWrapper(model, **args_call)
        bo = BayesianOptimization(cost, bounds_bo, **options)
        acq = options['acq']
        
        #Initialization phase
        init = options['init_params']
        if(ut.is_int(init)):
            nb_points_init = init
            bo.maximize(init_points = nb_points_init, n_iter=0, acq=acq, kappa=0) 
        else:
            # pdb.set_trace()
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
        parameters
        ----------
        options.constraints: <str> or None
            encode the type of constraints which can be used. This string is converted
            to be passed to the BayesianOptimizer constructor with the following structure  
            [{'name':'name1', 'constraint':obj_constraint1}, .., {'name':'nameN', 'constraint':obj_constraintN}]
            where obj_constraintX can be a function or a string which can be exec to generate a function
            a constraint is of the form c(x) <= 0 (i.e. a parameter is accepted if c(x) <= 0) 
            e.g. 'step_a' difference between two consecutive parameters should be less than a
            e.g. 'step_a_a0_aN' same as above plus the first (last) parameter is also compared to a0 (aN)
            e.g. 'smooth_s' smoothness calculated should be <=s

        
        """
        #Init BO
        model = options['model']

        nb_params = options['nb_params']
        name_params = [str(i) for i in range(nb_params)]
        options['name_params'] = name_params        
        bounds_bo = [{'name': name_params[i], 'type': 'continuous', 
                   'domain': options['bounds_params'][i]} for i in range(nb_params)]
        constraints = options.get('constraints')
        cst = self._build_constraints(constraints, **options)        

        args_BO = {'acquisition_type':options['acq'], 
                   'acquisition_optimizer_type':options['acq_opt_type'], 
                   'num_cores':options['num_cores'], 
                   'domain': bounds_bo,
                   'optim_num_anchor':options['optim_num_anchor'], 
                   'optim_num_samples':options['optim_num_samples'],
                   'acquisition_jitter':options['acquisition_jitter'], 
                   'acquisition_weight':options['acquisition_weight'],
                   'batch_size':options['batch_size'], 
                   'evaluator_type':options['batch_method'],
                   'num_inducing':options['num_inducing'], 
                   'model_type':options['model_type'], 
                   'ARD':options['ARD'],
                   'acquisition_weight_lindec':options['acquisition_weight_lindec'], 
                   'constraints':cst}    

        # definition of the cost function
        def cost(params):
            return model(np.squeeze(params), **args_call)

        #V0.1 NON DEFAULT LIKELIHOOD
        if(args_BO['model_type'] == 'GP_CUSTOM_LIK'):
            logger.info('Use of GP_CUSTOM_LIK: enforce **normalize_Y** as False ')
            logger.info('Use of GP_CUSTOM_LIK: enforce **inf_method** as Laplace')
            args_BO['normalize_Y'] = False
            args_BO['inf_method'] = 'Laplace'
            args_BO['likelihood'] = options.get('likelihood', 'Binomial_10') 



        #V0.1 transfer learning
        to_transfer = options['to_transfer']
        if((to_transfer is not None) and (args_BO['model_type'] == 'GP_STACKED')):
            args_BO_transfer = copy.copy(args_BO)
            args_BO_transfer['model_type'] = 'GP'
            args_BO_transfer['X'] = to_transfer['X']
            args_BO_transfer['Y'] = to_transfer['Y']
            regressor_to_transfer = GPyOpt.methods.BayesianOptimization(cost, **args_BO)
            regressor_to_transfer.model._create_model(args_BO_transfer['X'], args_BO_transfer['Y'])
            args_BO['prev'] = regressor_to_transfer.model

        init = options['init_params']
        if(init is None):
            args_BO['initial_design_numdata'] = int(3 * nb_params)
            args_BO['initial_design_type'] = options['initial_design_type']
        elif(ut.is_int(init)):
            args_BO['initial_design_numdata'] = init
            args_BO['initial_design_type'] = options['initial_design_type']
        else:
            args_BO['X'] = init
            logger.info('Init of the GP: acquisition of {} points'.format(init.shape[0]))
            args_BO['Y'] = np.array([cost(x) for x in init])
        
        ker = options['kernel'] # matern52 /matern32
        if(ker is not None):
            ard = options['ARD']
            if(ker == 'matern52'):
                args_BO['ker'] = eval("GPy.kern.Matern52(self.input_dim = nb_params, ARD={})".format(ard))
            elif(ker == 'matern32'):
                args_BO['ker'] = eval("GPy.kern.Matern32(self.input_dim = nb_params, ARD={})".format(ard))
            else:
                logger.warning("{} not a valid kernel".format(ker))
        
        bo = GPyOpt.methods.BayesianOptimization(cost, **args_BO)
        #Initialization phase


        # Exploration-Exploitation phase
        max_time = options['max_time']
        bo.run_optimization(max_iter = options['maxiter'], max_time = max_time)
        time_left = max_time - bo.cum_time
        max_reached = (time_left < 0)
        
        # Exploitation phase
        # should it be forced eben if max_time is reached ? No so far 
        exploit_steps = options['exploit_steps']
        if(exploit_steps > 0): 
            if(not(max_reached)):
                bo.acquisition_type = 'LCB'
                bo.acquisition_weight = 0.000001
                bo.kwargs['acquisition_weight'] = 0.000001
                bo.acquisition = bo._acquisition_chooser()
                #maybe to add new logic when Batch/Sparse
                if(bo.batch_size > 1):
                    bo.batch_size = 1
                bo.evaluator = bo._evaluator_chooser()
                                     
                logger.info('Exploitation (i.e. ucb with k=0) for {}'.format(exploit_steps))
                logger.info('Still {} s left'.format(time_left))
                bo.run_optimization(exploit_steps, max_time = time_left)
                time_left -= bo.cum_time
                max_reached = (time_left < 0)

        if((_still_potentially_better(bo)) and (exploit_steps <= 50) and not(max_reached)):
            logger.info('2nd round of exploitation for {}'.format(exploit_steps))
            logger.info('Still {} s left'.format(time_left))
            bo.run_optimization(exploit_steps, max_time = time_left)

        # generate results
        optim_params = bo.x_opt
        optim_value = bo.fx_opt
        optim_params_exp = _get_best_exp_from_BO(bo)
        
        optim_value_cputed = model(optim_params)

        nfev = len(bo.X)
        resultTest = {'x': optim_params, 'x_exp':optim_params_exp, 'fun': optim_value, 
                      'fun_ver': optim_value_cputed, 'nfev':nfev, 'nit':nfev, 'sucess':True}
        resultTest['gp_kernel_optim_names'] = bo.model.model.parameter_names()
        resultTest['gp_kernel_optim_vals'] = bo.model.model.param_array
        resultTest['nb_processes'] = bo.num_cores
        resultTest['nb_cpus'] = self.mp.n_cpus
        resultTest['X_evol'] = bo.X
        resultTest['Y_evol'] = bo.Y
        resultTest['Y_best'] = bo.Y_best
        resultTest['still_potentially_better'] = _still_potentially_better(bo)
        resultTest['maxtime'] = max_reached
        
        # Close pool of processors used (if needed)
        self.mp.close_mp()
        return resultTest

    def _build_constraints(self, constraint_obj, **xargs):
        # Creation of constraints
        if constraint_obj is not None:
            if ut.is_str(constraint_obj):
                components = constraint_obj.split('_')
                if(components[0] == 'step'):
                    limit = float(components[1])
                    logger.info("use of constraints: step with step value {0}".format(limit))
                    nb_params = xargs['nb_params']
                    cst = [{'name':str(i), 'constraint': 'abs(x[:,{0}]-x[:,{1}])-{2}'.format(i-1, i, limit)} for i in range(1, nb_params)]
                    if(len(components) >= 3):
                        xref = float(components[2])
                        cst += [{'name':'0', 'constraint':'abs(x[:,0]-{0})-{1}'.format(xref, limit)}]
                    if(len(components) >= 4):
                        xref = float(components[3])
                        cst += [{'name':str(nb_params), 'constraint':'abs(x[:,-1]-{0})-{1}'.format(xref, limit)}]

                elif(components[0] == 'smooth'):
                    # quite costly to compute
                    model = xargs['model']
                    func = model.control_fun[0].clone()
                    if model.n_controls > 1:
                        logger.warning('smoothness constraints have just been computed on the first control function')
                    limit = float(components[1])
                    time = model.t_array.copy()
                    time[0] = time[0] - 1e-6
                    time[-1] = time[-1] + 1e-6
                    logger.info("use of constraints: smoothness is capped to {0}".format(limit))
                    
                    def compute_smoothness(x, func, limit):
                        func.theta = x
                        return func.smoothness(time) - limit
                    
                    def smooth_constr(x):
                        return np.array([compute_smoothness(xx, func, limit) for xx in x])
                    
                    cst = [{'name':'smooth', 'constraint':smooth_constr}]
                    
                elif(components[0] == 'smoothlin'):
                    limit = float(components[1])
                    if(len(components) == 4):
                        xinit = float(components[2])
                        xfinal = float(components[3])
                        def smoothlin_constr(x):
                            res = np.sum(np.square(np.diff(x)), axis =1) + np.square(x[:,0] -xinit) + np.square(x[:,-1] -xfinal)
                            return res / (x.shape[1] + 1) -limit
                    else:
                        def smoothlin_constr(x):
                            return np.sum(np.square(np.diff(x)), axis = 1)  / (x.shape[1] -1) -limit
                    logger.info("use of constraints: smoothness linear is capped to {0}".format(limit))

                    
                    cst = [{'name':'smooth', 'constraint':smoothlin_constr}]

                else:
                    logger.error("Constraints for BO {} is not recognized".format(components[0]))
            else:
                 logger.error("Constraints for BO {} is not recognized".format(constraint_obj))
        else:
            cst = None
        return cst

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




def _still_potentially_better(bo, nb_last = 5, nb_thr = 2, accept_thr = 1e-7):
    """ """
    last_Y = np.squeeze(bo.Y[-(nb_last+1):])
    nb_better = np.sum(np.diff(last_Y)<-accept_thr)
    return nb_better >= nb_thr
    
def _get_best_seen_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y"""
    return bo.X[np.argmin(bo.Y)]
    
def _get_best_exp_from_BO(bo):
    """ From BO optimization extract X giving the best predicted Y (only amongst 
    the tested values)"""
    Y_pred = bo.model.predict(bo.X)
    return bo.X[np.argmin(Y_pred[0])]


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
    test_camel = False
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

        optim_args = {'algo': 'DE', 'max_time':0}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)

        optim_args = {'algo': 'NM', 'max_time':0}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)
        
        optim_args = {'algo': 'BO2', 'maxiter':25, 'optim_num_samples':100000, 
                      'optim_num_anchor':15, 'initial_design_type':'latin',
                      'acquisition_jitter':0.0001, 'max_time':20}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        print(resOptim)
        
        #optim_args = {'algo': 'DE', 'init_obj':[0,0]}
        #optim = learner_Opt(model = camel, **optim_args)
        #resOptim = optim()
        #print(resOptim)
        optim_args = {'algo': 'BO2', 'maxiter':25, 'acq':'LCB',
                      'acquisition_weight_lindec':True, 'optim_num_samples':10000, 
                      'optim_num_anchor':15, 'initial_design_type':'latin',
                      'acquisition_jitter':0.0001}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        # pdb.run("resOptim = optim()")
        
        ##Batch
        optim_args = {'algo': 'BO2', 'maxiter':10, 'acq':'LCB',
                      'acquisition_weight_lindec':True, 'optim_num_samples':10000, 
                      'optim_num_anchor':15, 'initial_design_type':'latin',
                      'acquisition_jitter':0.0001, 'batch_size':4, 'batch_method':'local_penalization', 'num_cores':4}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        
        ##Sparse
        optim_args = {'algo': 'BO2', 'maxiter':10, 'acq':'LCB',
                      'acquisition_weight_lindec':True, 'optim_num_samples':10000, 
                      'optim_num_anchor':15, 'initial_design_type':'latin',
                      'acquisition_jitter':0.0001, 'model_type':'sparseGP'}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        
        ##Lineardecay
        optim_args = {'algo': 'BO2', 'maxiter':40, 'acq':'EI',
                      'acquisition_weight_lindec':True, 'optim_num_samples':10000, 
                      'optim_num_anchor':15, 'initial_design_type':'latin',
                      'acquisition_jitter':0.0001, 'acquisition_weight':4, 'acquisition_weight_lindec':True}
        optim = learner_Opt(model = camel, **optim_args)
        resOptim = optim()
        
        optim_args = {'algo': 'BO', 'maxiter':25, 'gp_acq_iter':50,'acq':'ei'}
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
                
        optim_args = {'algo': 'BO2', 'maxiter':25, 'acquisition_type':'LCB','acquisition_weight_lindec':True}
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
        

        