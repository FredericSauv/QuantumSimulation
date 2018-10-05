#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import sys, copy, pdb, time, logging
from functools import partial
import operator as op, itertools
import numpy as np
logger = logging.getLogger(__name__)
import matplotlib.pylab as plt
plt.rc('text', usetex=True) 

if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Misc.RandomGenerator import  RandomGenerator as rdm
    from QuantumSimulation.Utility.Misc.MP import MPCapability as mp
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    from QuantumSimulation.Utility.Optim import pFunc_zoo as pfzoo

else:
    from ..Utility import Helper as ut
    from ..Utility.Misc.RandomGenerator import RandomGenerator as rdm
    from ..Utility.Misc.MP import MPCapability as mp
    from ..Utility.Optim import pFunc_base as pf
    from ..Utility.Optim import pFunc_zoo as pfzoo



class model_base:
    """Base class for (quantum) models i.e. it defines what a model (simulation) is.
    A model is comprised of an Hamiltonian (self._H), an underlying state space
    (self._ss), time attributes (i.e. self.T, self.t_simulation, self.t_simul), 
    and an initial state (self.state_init)

    Evolve: evolve the initial state according to the Hamiltonian up to final time (self.T) 
    while recording intermediate state if 'flag_intermediate' is True. 

    Simulate: evolve the system and computing some figure of merit (FoM).

    Notes
    -----
    * Dimension
        d: dimension of the Hilbert space, n_t number of time points, e: number of emsembles
        A state has
        A state over time (i.e. state_t) has dim d x n_t
        
    
    """

    # {<str:name_mandatory_param> : <str: info_opt_param>}
    _LIST_ARGS = {'T':'<float> Horizon of the simul', 'dt':'<float> time step', 
                 'flag_intermediate':'<bool> allows to store intermediate evolution', 
                 'state_init':'<string/np.array> Initial state'}
    
    # {<str:name_opt_param> : (<str: info_opt_param>, <var:default_value>)}
    _LIST_ARGS_OPT = {'rdm_obj':('<int/randomstate/None> Use to generate/pass the random generator', None),
                     'mp_obj':('<int/bool/mp> Use to generate/pass the random generator', None),
                     'noise':('<dict<string>> to build noise in the model ',None), 
                     'fom':('fom <str> or <list<str>>, has', None),
                     'state_init':('<string/np.array> Initial state',None),
                     'fom_print':('<bool> should the fom be printed',False)
                     }
    
    def __init__(self, **args_model):
        """Intializes the `model` object.
        States: state_init, state_tgt, state
        self.pop: population i.e. the probability of a state (given in the reference basis)

        Notes
        -----
        * DIMENSIONALITY:
            States which don't depend on time are of dim d while states over time have dim d x t
        
        * ENSEMBLE BEHAVIOR: self._ensemble_simulation flag the use of this behavior
            Ensemble simulations means that the different noisy realizations of 
            the dynamics are simulated. This noise could be introduced either 
            on the Hamiltonians (e.g. energy scales are noisy) in this case 
            self._H_ensemble is used or on the initial state (e.g. thermal state, 
            or imprecision on the state), in this case self._init_ensemble will
            be used


        Parameters
        -----------
        T : float
            Horizon (i.e. t max) 
            
        dt : float
            time step

        flag_intermediate: boolean
            Should we store state of the system at intermediate time (i.e. at each time steps) 

        state_init : string, np.array
            Should be of the right size 
        
        rdm_obj : np.random.RandomState, optional
            a random state 

        mp_obj : MultiProcessing object, optional
            
        noise : dict, optional
            XXXXXXXXXXX

        fom: list of str, optional
            Encode which figure of merit should be used
        
        measures: list of str, optional
            Things to measure. Similar logic as fom except that 'fom' is 
            the main figure of interest while 'measures' refers extra stuff computed
            
    

        TO DO
        ------
        state should be [t][hilbert space][(optional) nb parallel simuls] 
        noise when multidim

        """
    
        self._ss = None # underlying state space

        self._H = None # Hamiltonian
        self._rdmgen = None #random generator
        self._fom_func = {} # functions available to compute FOM
        self._t_array = None #
        self.pop_t = None #
        self.t_simul = None # time array of the simulation
        self.T = None #Horizon of the simulation
        
        # ensemble behavior
        self._ensemble_simulation = False # Should we simulate
        self._nb_H_ensemble = 1 # How many H in the ensemble
        self._nb_init_ensemble = 1 # How many state_init in the ensemble
        self._H_ensemble = None # list with len = _nb_H_ensemble
        self._init_ensemble = None # list with len = _nb_init_ensemble

        self._setup_time(**args_model)
        self.rdm_gen = args_model.get('rdm_obj')
        self._setup_fom_basic_bricks()
        self.mp = args_model.get('mp_obj')
        self.noise = args_model.get('noise')
        self.fom = args_model.get('fom') 
        self.fom_ensemble = args_model.get('fom_ensemble', 'avgens') #For ensemble
        self.measures = args_model.get('measures') 
        self.measured = None
        self._fom_print = args_model.get('fom_print', False)
        


    def _setup_fom_basic_bricks(self):
        """ populate the dictionary self._fom_func of functions which can be used
            to compute FoM """
        self._fom_func['std'] = np.std
        self._fom_func['max'] = np.max
        self._fom_func['min'] = np.min
        self._fom_func['avg'] = np.average
        self._fom_func['sqrt'] = np.sqrt
        self._fom_func['square'] = np.square
        self._fom_func['neg'] = lambda x: 1- x
        self._fom_func['last'] = model_base._get_last
        self._fom_func['clip01'] = lambda x: np.clip(0, 1, x)

        #To add randomness on top of the FoM
        self._fom_func['rdmplus'] = (lambda x: x + self._noise_func['fom']())
        self._fom_func['rdmtime'] = (lambda x: x * (1 + self._noise_func['fom']()))

        #For ensemble computation: how do we gather results
        self._fom_func['stdens'] = lambda x: np.std(x, axis = 0)
        self._fom_func['maxens'] = lambda x: np.nax(x, axis = 0)
        self._fom_func['minens'] = lambda x: np.min(x, axis = 0)
        self._fom_func['avgens'] = lambda x: np.average(x, axis = 0)

    @classmethod
    def info(cls):
        logger.info('Mandatory params for the model')
        logger.info(cls._LIST_ARGS)
        logger.info('Optional params for the model')
        logger.info(cls._LIST_ARGS_OPT)

    @property
    def state_init(self):
        return self._state_init 
    
    @state_init.setter
    def state_init(self, init):
        self._state_init = self.get_state(init)

    @property
    def state_tgt(self):
        return self._state_tgt 
    
    @state_tgt.setter
    def state_tgt(self, tgt):
        self._state_tgt = self.get_state(tgt)

    @property
    def state(self):
        if(hasattr(self, '_state_t')):
            return self._state_t 
        else:
            return None

    @state.setter
    def state(self, st):
        self._state_t = st
        self._pop_t = self._state_to_pop(st)

    @property
    def pop(self):
        if(hasattr(self, '_pop_t')):
            return self._pop_t
        else:
            return None

    @property
    def pop_adiab(self):
        if(hasattr(self, '_pop_adiab')):
            return self._pop_adiab
        else:
            return None

    @pop_adiab.setter
    def pop_adiab(self, pop):
        self._pop_adiab = pop
    
    
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
        self._mp = mp.init_mp(mp_object = mp_obj)

    @property
    def noise(self):
        return self._noise_func
    
    @noise.setter
    def noise(self, noise=None):
        """ allow generation of noise based on a noise object

        Notes
        -----
            * self._noise_func will contain callable which generates noise
            * self._ensemble_XXX

        Parameters
        -----------
        noise: dict
            * keys = '_ensemble_H' values = int (default = 1)
              if >1: will build an ensemble of several H
              based on random draws
            
            * keys = '_ensemble_state_init' values = int (default = 1)
              if >1: will build an ensemble of several state_init
              based on random draws

            * keys values: <callable> or string 

        RETURNS
        ------
        _noise_func: dict 
            * dict({}) : no noise will be implemented
            * {name_noise_1: callable, ...}
               These functions can be used later on when building the
               Hamiltonian, state_init or computing the FoM

        Example
        -------
        noise = {'Ex':'normal_0_0.05','Ez':'normal_0_0.05'}
        >> self._noise_func = {'Ex':lambda : np.random.normal(0,1), 
                               'Ez'lambda : np.random.normal(0,1)}
        """
        if noise is None:
            self._noise_func = dict({})

        elif(ut.is_dico(noise)):
            n_dico = copy.copy(noise)
            self._nb_H_ensemble = n_dico.pop('nb_H_ensemble', 1) 
            self._nb_init_ensemble = n_dico.pop('nb_init_ensemble', 1)
            self._noise_func = {k:self._gen_noise_function(v) for 
                                k, v in n_dico.items()}
        else:
            raise NotImplementedError('Noise should be passed as a dict or None')

    @staticmethod
    def _get_last(state):
        """ Return a state at its final time: states over time have dim d x t """
        if(np.ndim(state) == 1):
            return state
        else:
            return state[:, -1]

    def _gen_noise_function(self, noise_obj):
        """ allow generation of noise function based on a noise_obj
        Parameters
        -----------
        noise_obj: string or callable
                * if callable simply return it
                * if string will try to build a callable based on rdm_gen 
                method (this way enforce the use of the random state 
                associated to the model) if it fails try to eval the string
        
        RETURNS
        -------
        res: a callable 

        """
        if(ut.is_str(noise_obj)):
            try:
                res = self.rdm_gen.gen_rdmfunc_from_string(noise_obj)
            except:
                res = eval(noise_obj)
        elif(ut.is_callable(noise_obj)):
            res = noise_obj
        else:
            raise NotImplementedError
        return res
        
    def _setup_time(self, **args_model):
        """ Generate time attributes of the model """
        self.T = args_model['T']
        self.dt = args_model['dt']
        self._flag_intermediate = args_model['flag_intermediate']
        
        self.t_array = np.concatenate((np.arange(0, self.T, self.dt), [self.T]))
        if(self._flag_intermediate):
            self.t_simul = self.t_array
        else:
            self.t_simul = self.T
           
     
    #LOGIC FOR COMPUTING THE FOM
    @ut.extend_dim_method(0, True)
    def _compute_fom(self, fom = None, st = None, func_assemblate = np.sum):
        """Compute a (potentially composed) FoM (or list of FOM)
        Parameters
        ----------
        fom: string (seeparator=':' and '_')
            the description of the FoM
        
        st: np.array
            state as an array an be n-D

        Notes
        -----
        Extended to list of fom using a wrapper


        Returns
        --------
        value of FoM: float or list
        """    
        fom = fom if fom is not None else self.fom
        st = st if fom is not None else self.state
        components = ut.splitString(fom) # default separator = '_'
        res = func_assemblate([self._compute_atom_fom(c, st) for c in components])
        return res       


    def _compute_atom_fom(self, fom, st):        
        """Compute an atomic (simplest brick) FoM 
        Parameters
        ----------
        fom: string (seeparator=':')
            the description of the FoM
        
        st: np.array
            state as an array an be n-D


        Returns
        --------
        value of FoM: float
        
        Examples
        ---------
        >>> x =  self._compute_atom_fom('lst:f2t:neg:0.3', st)
        corresponds to :math:`B = 0.3*(1 - |2<Psi(T)|Psi(target)>|)`
        take the last part of the state (i.e. state at time T), compute the fidelity 
        to target return the neagtive version (neg(x)=1-x) and times it by 0.3  
        """                
        f2apply = [self._get_fom_func(k) for k in ut.splitString(fom, ":")]
        res = ut.compoFunctions(f2apply, st, order = 0) # compo from left to right
        return res 

    def _get_fom_func(self, fom_str):
        """ find the fom_func associated to a string, if it can't find it
        treats it a multiplying coeff"""
        f = self._fom_func.get(fom_str)
        f = f if f is not None else partial(op.mul, float(fom_str))
        return f  


    ## TO BE IMPLEMENTED IN THE CHILD CLASSES
    def get_state(self, state):
        """ Return a state based on some string/array/other formats"""
        raise NotImplementedError()      
            
    def _state_to_pop(self, st):
        """ compute population from state """
        raise NotImplementedError()

    def Simulate(self, **args):
        """ Main entry point to simulate the system: evolves the init state and 
        performs some other actions (typically computing a figure of Merit or other 
        statistics)
        """
        raise NotImplementedError()            

    def Evolution(self, **args):
        """ Evolve the init_state according to the relevant method and store if
            required, output is a state either a single one or one over time """
        raise NotImplementedError()      


class cModel_base(model_base):
    """ Base class for controlled models i.e. the hamiltonian depends on some
    control function(s) """
    _LIST_ARGS = model_base._LIST_ARGS
    _LIST_ARGS['control_obj'] = "<callable> or <dict> or <str>"
    
    def __init__(self, **args_model):
        """ init model and control functions:
        
        Notes
        -----
        self.control_fun: list of callables
        self._aggregated_nb_call should keep track of how many times the fun 
        has been called, not in use right now 
        """
        model_base.__init__(self, **args_model)
        self.control_fun = args_model['control_obj']
        self._setup_fom_controlfun_bricks()
        self._aggregated_nb_call = 0 

    #-----------------------------------------------------------------------------#
    # Management of the contro_fun (a list)
    #-----------------------------------------------------------------------------#
    @property
    def control_fun(self):
        return self._control_fun

    @property
    def n_controls(self):
        return len(self.control_fun)

    @control_fun.setter
    def control_fun(self, control):
        list_fun = self._process_control_function(control)
        self._control_fun = list_fun if ut.is_iter(list_fun) else [list_fun]
        self._update_model_after_change_of_control_fun()
        
    def _update_model_after_change_of_control_fun(self):
        """ Extra stuff to do when the control_fun is replaced by another one"""
        pass


    @ut.extend_dim_method()
    def _process_control_function(self, control):
        """ control_fun should be a (list of) callable(s)"""
        if(ut.is_callable(control)):
            return control
        else:
            raise SystemError("couldn't use type {0} to create the control_functions "
                "in class {1}").format(type(control), self.__class__) 

    #-----------------------------------------------------------------------------#
    # New FoM functions 
    #-----------------------------------------------------------------------------#
    def _setup_fom_controlfun_bricks(self):
        """ New self._fom_func relating to the controlfun: fluence and smoothness
        
        Notes
        -----
        lambda functions with a mock parameter
        One of a rare case where t_array is used and not t_simul
        """
        self._fom_func['fluence'] =  (lambda x: 
            self._get_info_control('fluence', self.t_array, None, None))
        self._fom_func['smooth'] = (lambda x: 
            self._get_info_control('smoothness', self.t_array, None, None))

    def _get_info_control(self, info_type = 'fluence', time = None, index = None, split = False):
        """ compute information (smoothness, fluence) relating  to the control function  
        
        PARAMETERS
        ----------
        time: list of float
            Used to compute the smoothness and fluence
        index: int or None
            which control_function are we looking at, if None: all of them
        split: boolean
            if False sum over all the control_fun required else return a list
        """
        time = self.t_array if time is None else time
        dico_info_type = {'fluence':self.__class__._fluence, 'smoothness':self.__class__._smoothness}
        info_func = dico_info_type[info_type]
        if(index is None):
            index = np.arange(self.n_controls)
        elif(not(ut.is_iter(index))):
            index = [index]
        res = [info_func(f, time) for n,f in enumerate(self.control_fun) if n in index]
        res = res if split else np.sum(res)
        return res
    

    @staticmethod
    def _smoothness(func, time):
        """ Smoothness as avg <([f(x) - f(x + dx)]/dx) ** 2> """
        time_copy = time.copy()
        time_copy[0] = time_copy[0]-1e-6 # to ensure first change is taken into account
        time_copy[-1] = time_copy[-1]+1e-6
        step = np.diff(time_copy)
        diff_val_square = np.square(np.diff(func(time_copy)))
        res = np.sum(np.array(diff_val_square / step))
        res = res/(time_copy[-1] - time_copy[0])
        return res
    
    @staticmethod
    def _fluence(func, time):
        """ Fluence as sum_x f(x)**2 dx / (x_max - xmin) """
        step = np.diff(time)
        val_square = np.square(np.abs(func(time)))
        res = np.sum(np.array(val_square[:-1] * step))
        res = res/(time[-1] - time[0])
        return res
    
    
class pcModel_base(cModel_base):
    """Base class for parametrized control models i.e. control_func[i] is a pFunc_base
    obj. The class implements __call__ making it take some paramaters as arguments,
    update the parameters of the control_func and simulate the system returning 
    some FoM

    NOTE
    ----
        * pFunc_base obj are callable objects with extra attributes and methods: 
          nb_params, params_bounds

    """

    # Available type of parametrized control functions
    _LIST_CUSTOM_FUNC = pfzoo.pFunc_factory._LIST_CUSTOM_FUNC

    def __init__(self, **args_model):
        cModel_base.__init__(self, **args_model)
        # flag indicating if we  track the calls to the control_fun
        self._flag_track_calls_void = True 

    @property
    def n_params(self):
        return self.control_fun.n_theta

    @property
    def params_bounds(self):
        return self.control_fun.theta_bounds
    
    #TODO: Use pFunc_collec instead of list
    def _process_control_function(self, control):
        """ New implementation of the setter
        
        NOTE
        ----
        * the function passed is cloned  
        """
        
        if(isinstance(control, pf.pFunc_List)):
            res = control.clone()
            
        elif(isinstance(control, pf.pFunc_base)):
            res = pf.pFunc_List(list_func = [control]).clone()

        elif ut.is_list(control):
            res = [self._process_atom_control_function(c) for c in control]
            res = pf.pFunc_List(res)
        else:
            raise SystemError('control should be provided as a list.. even if only one elem')
        
        return res
            
    def _process_atom_control_function(self, control, as_collection = True):
        """ if it is a pFuncBase fine, else should be created by pFuncZooFactory """
        
        if(isinstance(control, pf.pFunc_base)):
            res = control.clone()
        else: 
            res = pfzoo.pFunc_factory.build_custom_func(control, rdm_gen = self.rdm_gen) 
        return res

    def update_control_parameters(self, params, index_control = None, update_H = False,  **args_update):
        """ Update the parameters of the control function(s) i.e. the thetas (free params)
        of the functions """
        if(args_update.get('debug')):
            pdb.set_trace()

        if(index_control is None):
            self.control_fun.theta = params
        else:
            self.control_fun[index_control].theta = params

        if(update_H):
            self._update_model_after_change_of_control_fun()


    def __call__(self, params, trunc_res = True, **args_call):
        """ take an array of parameters update the parametrized control functions
        and return the (possibly truncated) result of the simulation (self.Simulate)

        Params
        ------
        * trunc_res : boolean
            Simulate may return a list of values (e.g. if fom is a list) if True return 
            the first value
        
        (extra_args) should they be part of the model?
        * track_learning : boolean
            Tracking of the calls to the simulator
        
        * update_H: boolean
            Should we rebuild the Hamiltonians each time we call the model
            (it makes only sense for noisy setups when we want to simulate that noise 
            change for one simulation to the other)

        Note
        ----
        * This is used for optimization purposes
        * Could add a gradient if available

        """
        self._aggregated_nb_call += 1
        args_call_dupl = copy.copy(args_call)
        track = args_call_dupl.pop('track_learning', False)
        update = args_call_dupl.pop('update_H', None)
        if(update is None):
            if(len(self._noise_func) > 0):
                update = True
            else:
                update = False
        self.update_control_parameters(params, update_H = update,  **args_call_dupl)
        res_tmp = self.Simulate(**args_call_dupl)
        if(self._fom_print):
            logger.info(res_tmp)
        res = res_tmp[0] if(trunc_res and ut.is_iter(res_tmp)) else res_tmp

        # 
        if track:
            if self._flag_track_calls_void:
                if(not(hasattr(self, '_timer'))):
                    self._timer = time
                self._track_time_init = self._timer.time()
                self._track_fom = None
                self._flag_track_calls_void = False
                
                self._track_calls = {'history_nev_fun':[], 'history_time_fun':[],
                    'history_func_fun':[], 'best_fun':None, 'best_fun_full':None, 
                    'history_nev_params':[], 'history_nev_time':[], 'history_measured':[]}
                                
            best = self._track_calls['best_fun']
            if (best is None) or (best > res):
                time_elapsed = self._timer.time() - self._track_time_init
                self._track_calls['best_fun'] = res
                self._track_calls['best_fun_full'] = res_tmp
                self._track_calls['history_nev_fun'].append([self._aggregated_nb_call, res])
                self._track_calls['history_time_fun'].append([time_elapsed, res])
                self._track_calls['history_func_fun'].append([repr(self.control_fun), res])
                self._track_calls['history_nev_params'].append([self._aggregated_nb_call, params])
                if(self.measured is not None):
                    self._track_calls['history_nev_measured'].append([self._aggregated_nb_call, self.measured])

        return res


    
class pcModel_qspin(pcModel_base):

    """ Models based on the QuSpin package. inherit from the parametrized control
    function base class. 
    
    New features:
    * Building the state space (equiv to the basis objects in QuSpin)
    * Build the Hamiltonian
    * Define new building blocks for computing the fom
    * provide helper functions to act on QuSpin objects self.h_XXX
      e.g. self.h_ip for the inner product
    """
    def __init__(self, **args_model):
        """ """
        pcModel_base.__init__(self, **args_model)
        self._setup_basis(**args_model)
        self._setup_H(**args_model)
        self._setup_fom_qspin_bricks()

    def _update_model_after_change_of_control_fun(self):
        """ Regenerate H each time control_fun """
        if(hasattr(self, '_H') and (self._H is not None)):
            self._setup_H()
    
    # --------------------------------------------------------------------------- #
    #   SIMULATIONS 
    # --------------------------------------------------------------------------- #
    def Simulate(self, time = None, H = None, state_init = None, fom = None, store = False, method = 'se', **extra_args):
        """ Main entry point to simulate the system. If fom is not None, it will 
        return it, if not return the state_t of the system.

        Note
        ----
            * Behavior when ensemble has been spin-off
            * ensemble based on different state_init is not yet fully implemented
        """
        if extra_args.pop('debug', None):
            pdb.set_trace()
        time = self.t_simul if time is None else time
        fom = self.fom if fom is None else fom
        measures = self.measures
        
        if(self._ensemble_simulation and (H is None) and (state_init is None)):
            if(self._ensemble_simulation):
                H_ensemble = self._H_ensemble
            else:
                H_ensemble = [self._H]
            if(self._nb_init_ensemble > 1):
                init_ensemble = self._init_ensemble
            else:
                init_ensemble = [self.state_init]
            fom_ensemble = self.fom_ensemble
            res = self.Simulate_Ensemble(time, H_ensemble, init_ensemble, fom, store, fom_ensemble, **extra_args)

        else:
            state_init =  self.state_init  if state_init is None else state_init
            H =  self._H  if H is None else H
            res = self.Evolution(time = time, H = H, state_init = state_init, method = method, store = store, **extra_args)
            if (fom not in [None, '']):
                res = self._compute_fom(fom, res)
            if(measures not in [None, '']):
                self.measured = self._compute_fom(measures, res)
        if(extra_args.get('print')):
            logger.info("FOM="+str(res))
        return res

    def Simulate_Ensemble(self, time, H, state_init, fom = None, store = False, fom_ensemble = 'avgens', **extra_args):
        """Simulate the system for an ensemble of models (either because there is an ensemble of H or state init)

        Arguments
        ---------
            


        Note
        ----
        * Should implement here the multiprocessing capability
        * 
        
        """
        configs = itertools.product(H, state_init, [time])
        configs_index = list(itertools.product(np.arange(len(H)), np.arange(len(state_init))))

        #MP should be here mp.map()
        #res = [c[0].evolve(c[1], t0=0, times = time, **extra_args) for c in configs]

        res = self.mp.map_custom(evolve, configs)
        
        if (fom not in [None, '']):
            fom_values = [self._compute_fom(fom, r) for r in res]
            res = self._compute_fom(fom_ensemble, fom_values)

        if (self.measures not in [None, '']):
            measures = [self._compute_fom(self.measures, r) for r in res]
            self.measured = self._compute_fom(fom_ensemble, measures)

        if(store):
            self._state_t_ensemble = res
            self._indices_ensemble = configs_index
            self._fom_values_ensemble = fom_values

        return res


    def Evolution(self, time, H, state_init, method, store, **extra_args):
        """ Evolve the state_init according to the relevant method and store if
            required
            state_t has dim (from quspin) [dim_H, dim_t]
        """
        if(method == 'se'):
            state_t = self.EvolutionSE(time, H, state_init, **extra_args)
        elif(method == 'pop_adiab'):
            state_t = self.EvolutionPopAdiab(time, H, state_init, **extra_args)
        else:
            raise NotImplementedError()
        if store:
            self.state_t = state_t
        return state_t

    # --------------------------------------------------------------------------- #
    #   Custom evolutions
    # --------------------------------------------------------------------------- #
    def EvolutionSE(self, time = None, H = None, state_init = None, iterable = False, **args_evolve):
        """  Evolve the state_init under the Schrodinger equatiohn
        Wrap the QuSpin evolve method: quspin.hamiltonian.evolve(state_init, t0 = 0,times = T, 
        eom="SE",solver_name="dop853",stack_state=False, verbose=False,iterate=False,imag_time=False,
        **solver_args)
        
        Notes
        -----
        * only deal with one H one state_init (i.e. no ensemble computation)
        * state_t has dim (from quspin) [dim_H, dim_t]
        """
        time = self.t_simul if time is None else time 
        if H is None:
            if self._ensemble_simulation:
                logger.warning("Evolution_SE: {0} of H in the ensemble, H[0] has beeen used".format(self._nb_H))
            H = self._H

        if state_init is None:
            if(self._nb_init_ensemble >1):
                logger.warning("Evolution_SE: {0} of state_init in the ensemble, state_init[0] has been used".format(self._nb_state_init))
            state_init = self.state_init
            
        state_t = H.evolve(state_init, t0 = 0, times = time, iterate=iterable, **args_evolve)
        return state_t


    def EvolutionPopAdiab(self, time = None, H = None, state_init = None, nb_ev = 2, **args_evolve):
        """ Evolve the state according to SE and project it on the instantaneous 
        eigen vectors of the hamiltonian. 
        
        Note
        ----
        * Pop_adiab is stored by default and state_t is returned 
        """
        time = self.t_array if time is None else time 
        patch_degen= args_evolve.pop('patch_degen', False) # correction when there is degen in the spectrum
        state_t = self.EvolutionSE(time = time, H = H, state_init = state_init, **args_evolve)
        ev, EV = self._h_get_eigensystem(time = time, H = H, nb_ev = nb_ev)
        
        n_t = len(time)
        if  state_t.shape[1] != n_t:
            logger.exception("EvolutionPopAdiab: state_t has dim {0}".format(state_t.shape))
        
        if(patch_degen):
            for t, e in enumerate(ev):
                list_degen = self._which_degen(e)
                if(t > 0 and len(list_degen)>0):
                    #Should we use graham schmidt first??
                    E_bf = EV[t-1]
                    E = EV[t]
                    E_new = self._patch_degen(list_degen, E, E_bf)
                    EV[t] = E_new

        proj_ev = [[pcModel_qspin._h_ip(state_t[:, t], EV[t][:, n]) for n in range(nb_ev)] for t in range(n_t)]
        self.adiab_pop = np.square(np.abs(np.array(proj_ev)))
        self.adiab_cf = np.array([self.control_fun(t) for t in time]) 
        self.adiab_en = np.array(ev) # both energies and control field
        self.adiab_evect = np.array(EV)
        self.adiab_t = time
        return state_t
    
    def FindMinDelta(self, time = None, H = None, level_min = 0, level_max=1):
        """ Find the minimal energy gap over time """
        ev = self._h_get_lowest_energies(time, H, level_max+1)
        return np.min(ev[level_max,:] - ev[level_min])
    
    def _patch_degen(self, list_degen, E, E_bf):
        """ if degen pick eigenvectors s.t. they are the closest to previous ones
        """
        E_new = np.copy(E)
        for deg_index in list_degen:
            #ss_bf = E_bf[:,deg_index]
            ss = E[:,deg_index]
            ss_new = []
            for n in deg_index:
                E_ref = E_bf[:, n]
                E_ref_plan = self._proj_SS(E_ref, ss)
                ss_new.append(self._proj_ortho_SS(E_ref_plan, np.transpose(ss_new)))
            E_new[:, deg_index] = np.transpose(ss_new)
        return E_new
                
                
    def _proj_SS(self, vec, SS, normalize=True):
        """project and normalize a vector on a Subspace (SS)
        
        Notes
        -----
        Assumption that SS is provided as list of ortho vectors
        """
        proj = np.zeros_like(vec)
        for st in np.transpose(SS):
            proj += pcModel_qspin._h_ip(st, vec) * st
        if normalize:
            proj = proj / pcModel_qspin._h_norm(proj)
        return proj
    
    def _proj_ortho_SS(self, vec, SS):
        """project and normalize a vector on the orthogonal on a ubspace
        Notes
        -----
        Assumption that SS is provided as list of ortho vectors
        """
        proj = np.copy(vec)
        for st in np.transpose(SS):
            proj -= pcModel_qspin._h_ip(st, vec) * st
        proj = proj / pcModel_qspin._h_norm(proj)
        return proj
    
    

    def _which_degen(self, e, dec = 8):
        """ return which energies are degenerate """
        e_rounded = np.round(e, dec)
        range_index = np.arange(len(e_rounded))
        val_uniques, val_nb = np.unique(e_rounded, return_counts=True)
        list_degen = [range_index[e_rounded == val_uniques[n]] for n, v in enumerate(val_nb) if v > 1]        
        return list_degen
    
    #-----------------------------------------------------------------------------#
    # plotting capabilities
    #-----------------------------------------------------------------------------#
    def plot_pop_adiab(self, **args_pop_adiab):
        """ Plot pop adiab where each population_t is dispatched on one of the 
        three subplot
        #TODO: add min gap
        """
        col_list = ['b', 'g', 'r', 'c', 'm', 'k','C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] * 50
        if(hasattr(self,'pop_adiab')):
            limit_legend = args_pop_adiab.get('lim_legend', 15)
            limit_enlevels = args_pop_adiab.get('lim_enlevels', np.inf)
            plot_gap = args_pop_adiab.get('plot_gap', False)
            pop_adiab = self.adiab_pop #txn
            t = self.adiab_t 
            en = self.adiab_en #txn
            cf = self.adiab_cf # txcf
            nb_levels = min(pop_adiab.shape[1], limit_enlevels)    
            #[0,0] control function
            f, axarr = plt.subplots(2,2, sharex=True)
            axarr[0,0].plot(t, cf, label = r"$\Gamma(t)$")
            second_pop_populated = False
            for i in range(nb_levels):
                col = col_list[i]
                pop_tmp = pop_adiab[:, i]
                max_tmp = np.max(pop_tmp)
                if(i<=limit_legend):
                    lbl_tmp = str(i)
                else:
                    lbl_tmp = None
                if(max_tmp > 0.1):
                    axarr[0,1].plot(t, pop_tmp, label = lbl_tmp, color = col)
                elif(max_tmp > 0.01):
                    second_pop_populated = True
                    axarr[1,1].plot(t, pop_tmp, label = lbl_tmp, color = col)
                if(i<10):
                    axarr[1,0].plot(t, en[:, i] - en[:, 0], label = lbl_tmp, color = col)
            
            ax_tmp = axarr[0,1]
            ax_tmp.legend(fontsize = 'small')
            ax_tmp.set_title('Population', fontsize = 8)
           
            
            ax_tmp = axarr[1,1]
            ax_tmp.set(xlabel='t')
            if(second_pop_populated):
                ax_tmp.legend(fontsize = 'small')
            
            ax_tmp = axarr[0,0]
            ax_tmp.legend()
            
            ax_tmp = axarr[1,0]
            ax_tmp.set(xlabel='t', ylabel=r"$E_i - E_0$")
            if(plot_gap):
                try:                    
                    diff_01 = en[:, 1] - en[:, 0]
                    index_min = np.argmin(diff_01)
                    t_min = t[index_min]
                    y_min= diff_01[index_min]
                    ax_tmp.arrow(t_min, 0, 0, y_min)
                    plot_gap_dx = args_pop_adiab.get('plot_gap_dx', -2.2)
                    ax_tmp.text(t_min - plot_gap_dx , (y_min/2), r"$\Delta = %.2f$"%(y_min), fontsize=8)
                except:
                    pass
                    
            save_fig = args_pop_adiab.get('save_fig')
            if(ut.is_str(save_fig)):
                f.savefig(save_fig, bbox_inches='tight', transparent=True, pad_inches=0)

        else:
            logger.warning("pcModel_qspin.plot_pop_adiab: no pop_adiab found.. Generate it first")

    #def plot_bloch_sphere(self):
        
    #def plot_evol_on_bloch_sphere(self, states= None):
        
    #def _plot_one_state_bs(self, state):
    
    
    #-----------------------------------------------------------------------------#
    # Setup functions
    #-----------------------------------------------------------------------------#
    def _setup_basis(self, **args_model):
        """ build the basis (self._ss) and potentially other basis"""
        raise NotImplementedError()
        
    def _setup_H(self, **args_model):
        """ build the Hamiltonian governing the dynamics of the system (self.H)"""
        raise NotImplementedError()
        
    def _setup_fom_qspin_bricks(self):
        """ Set up fom based of quspin states """
        #fidelity to the target state
        self._fom_func['f2t'] =  (lambda x: self._h_fid_tgt(x))
        self._fom_func['f2t2'] =  (lambda x: self._h_fid2_tgt(x))
        self._fom_func['projSS'] = (lambda x: self._h_projSS_tgt(x))

        # measurement projection on the target_state
        self._fom_func['energy'] = (lambda x: self._h_expect_energy(st = x, time= self.T, H = None, normalize=True ))
        self._fom_func['energyinf'] = (lambda x: self._h_expect_energy(st = x, time= np.inf, H = None, normalize=True ))
        self._fom_func['proj5'] = (lambda x: self._h_n_measures_tgt(x, nb =5))
        self._fom_func['proj10'] = (lambda x: self._h_n_measures_tgt(x, nb =10))
        self._fom_func['proj100'] = (lambda x: self._h_n_measures_tgt(x, nb =100))
        self._fom_func['proj1000'] = (lambda x: self._h_n_measures_tgt(x, nb =1000))

    def get_state(self, state_obj = None):
        """ Generate quantum states from state_obj <str> or <array/list<num>>
        """
        basis = self._ss
        if(state_obj is None):
            state_res = None

        elif(isinstance(state_obj, str)):
            if(state_obj == 'GS_i'):
                # GS at t=0
                _, state_res = self._H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
            elif(state_obj == 'GS_f'):
                #GS at t = T
                _, state_res = self._H.eigsh(time = self.T, k=1, which='SA',maxiter=1E10) 
            elif(state_obj == 'GS_inf'):
                #GS at t = 5*T (to use with owritten functions)
                _, state_res = self._H.eigsh(time = 5 * self.T, k=1, which='SA',maxiter=1E10) 
            elif(state_obj[:3] == 'EES'):
                # Energy EigenState e.g. 'EES_0_0.0' is the first Energy Eigen 
                # State at t = 0
                extra = state_obj[4:]
                n_ev, t = extra.split('_')
                n_ev = int(n_ev)
                t = float(t)
                
                _, state_res = self._h_get_eigensystem(time=[t], nb_ev=n_ev+1)
                state_res = state_res[0]
                state_res = state_res[:, n_ev] if n_ev > 0 else state_res
            
            elif(state_obj[:3] == 'ESS'):
                # Energy SubSPace e.g. 'EES_0_0.0' is the first Energy Eigen 
                # State at t = 0
                n_ev, t = state_obj[4:].split('_')
                n_ev = int(n_ev)
                t = float(t)
                
                _, state_res = self._h_get_eigensystem(time=t)
                state_res = state_res[min(n_ev, len(state_res) - 1)]
            
            elif(state_obj == 'uniform'):
                #GS at t = T
                state_res = np.random.uniform(-1, 1, size=basis.Ns)
                state_res = state_res / self.pcModel_qspin._h_norm(state_res)
            else:
                i_res = basis.index(state_obj)
                state_res = np.zeros(basis.Ns, dtype=np.float64)
                state_res[i_res] =1.0
        else:
            assert(len(state_obj) == basis.Ns), 'wrong size'
            state_res = np.array(state_obj)

        return np.squeeze(state_res)

    def _state_to_pop(self, st):
        """ compute pop from state"""
        return pcModel_qspin._h_state2proba(st)
    #-----------------------------------------------------------------------------#
    # Helper functions: functions to manipulate QuSpin objects (states, operators, etc..) 
    #-----------------------------------------------------------------------------#
    @staticmethod
    def _h_ip(V1, V2):
        """ compute an inner product between V1 and V2"""
        return np.dot(np.conj(np.squeeze(V1)), np.squeeze(V2))
    
    @staticmethod
    def _h_norm(V1):
        """ compute an inner product between V1 and V2"""
        return np.sqrt(np.abs(pcModel_qspin._h_ip(V1, V1)))
    
    @staticmethod
    def _h_fid2(V1, V2):
        """ compute fidelity(square conv) between V1 and V2"""
        return np.square(np.abs(pcModel_qspin._h_ip(V1, V2)))
    
    @staticmethod
    def _h_fid(V1, V2):
        """ compute fidelity between V1 and V2"""
        return np.abs(pcModel_qspin._h_ip(V1, V2))

    def _h_fid2_tgt(self,V1):
        """ compute fidelity(square conv) between V1 and V2"""
        return pcModel_qspin._h_fid2(self.state_tgt, V1)
    
    def _h_projSS_tgt(self,V1):
        """ compute """
        residual = np.copy(V1)
        tgt = self.state_tgt if(np.ndim(self.state_tgt) > 1) else [self.state_tgt]
        for st in tgt:
            residual -= pcModel_qspin._h_ip(st, V1) * st
        res = 1 - np.square(pcModel_qspin._h_norm(residual))
        return res

    def _h_fid_tgt(self,V1):
        """ compute fidelity(square conv) between V1 and V2"""
        return pcModel_qspin._h_fid(self.state_tgt, V1)
        
    @staticmethod
    def _h_state2proba(ket1):
        """ Gen proba distrib from a quantum state"""
        return np.square(np.abs(ket1))
        
    @staticmethod              
    def _h_variance(O, V):
        """ variance of a QuSpin operator O wrt a state V"""
        OV = O.dot(V)
        VOOV = np.asscalar(O.matrix_ele(V, OV))
        VOV2 = O.expt_value(V) ** 2
        var = VOOV -VOV2
        assert (np.imag(var) < 1e-8), 'Imaginary part not neglectible.. pb'
        return np.abs(var)
    
    @staticmethod
    def _h_variance2(O, V, **args):
        """ Another flavor based on quant_fluct (sign inversion atm)"""
        res = -O.quant_fluct(V, **args)
        return res
    

    @staticmethod
    def _h_n_measures(ket1, nb = 1, measur_basis = None, num_precis = 1e-6):        
        """ Projective measurements of the state(s) <ket1> in the basis <measur_basis>  
        repeated <nb> times 
        

        Arguments
        ---------
        * ket1: 1D-2D np.array
                state(s) to measure 
        * nb: int
              number of measurements 
        * measur_basis:  np.array dim = N x D 
              with N the dim of the basis, D dim of the entire H-space
              it should be an ORTHOGONAL basis      
              by default it is the computational basis of the full H-space 
              i.e. the basis in which ket1 is represented  

        Output
        -----
        * frequencies an 1D np.array dim = N
            frequencies[i] nb of time it was measured on the state measur_bais[i]

        """
        dim_ket = len(ket1)
        single_value = False

        if(measur_basis is None):
            measur_basis = np.eye(dim_ket) #Computational basis
        elif (len(measur_basis.shape) == 1):
            assert(len(measur_basis) == dim_ket), "problem with the dimension"
            measur_basis = [measur_basis] # just reshape it
            single_value = True
        else:
            assert(measur_basis.shape[1] == dim_ket)
            
        index_measure = np.arange(len(measur_basis))
        proj = [pcModel_qspin._h_ip(basis, ket1) for basis in measur_basis]
        proba = pcModel_qspin._h_state2proba(proj)
        assert(np.sum(proba)<=(1.0 + num_precis)), "not a valid proba distrib" #should it be relaxed 
        proba_cum = np.concatenate(([0],np.cumsum(proba)))
        samples = np.random.sample(nb)
        observations = [np.sum((samples > proba_cum[i]) * (samples <= proba_cum[i+1])) for i in index_measure]
        frequencies = np.array(observations)/nb
        if(single_value):
            assert(len(frequencies) == 1), "pb"
            frequencies = frequencies[0]
        return frequencies    
    
    def _h_n_measures_tgt(self, st, nb = 1, num_precis = 1e-6):  
        return pcModel_qspin._h_n_measures(st, nb , self.state_tgt, num_precis)

    
    def _h_expect_energy(self, st, time, H=None, normalize=True):
        """ Get <H> for a state"""
        if H is None:
            if(self._ensemble_simulation):
                res = [self._h_expect_energy(st, time, H, normalize) for H in self._H_ensemble]
                return res
            else:
                H = self._H
        en = H.expt_value(st, time=time)
        if(np.abs(np.imag(en))>1e-10):
            logger.error('expected energy shouldnt be have a imaginary part: {0}'.format(en))
        en = np.real(en)
        if(normalize):
            ev = self._h_get_lowest_energies(time, H=H, nb = H.Ns)
            en = (en - ev[0])/(ev[-1]-ev[0])    
        return en

    @ut.extend_dim_method(0, True)
    def _h_get_lowest_energies(self, time=None, H=None, nb = 2):
        """ Get the nb lowest energies of the Hamiltonian at time(s) t
        
        Output
        ------
        * ev with dimensionality (n_time x nb)

        """
        ev, _ = self._h_get_eigensystem(time, H, nb)
        return ev 
    

    def _h_get_eigensystem(self, time, H=None, nb_ev=5):
        """ get instantaneous eigenvalues and eigenvectors 

        Output
        -----
        ev sorted (ascending) eigenvalues DIM = [t, n]
        EV eigen vectors dim = [t, d, n] 
        with t the time index, d refers to the Hilbert space, and n is the index of the ev/EV
        i.e. at time t H(t) EV[t,:,i] = ev[t,i] EV[t, :, i]
        
                
        
        Notes
        -----
        * When several eigenvectors/values are required it may be more exact
          to use eigh (rather eigsh)
        * even when ensemble 
        """        
        if H is None:
            if self._ensemble_simulation:
                logger.warning("Evolution_SE: {0} of H in the ensemble, H[0] has beeen used".format(self._nb_H))
            H = self._H
        if(time is None):
            time = self.t_array if time is None else time
        time = [time] if(np.ndim(time)==0) else time

        if(nb_ev < H.Ns): 
            eigen_t = [H.eigsh(time=t, k=nb_ev, which='SA',maxiter=1E10) for t in time]
        else:
            eigen_t = [H.eigh(time=t) for t in time]
        ev = [ev for ev, _ in eigen_t]
        EV = [EV for _, EV in eigen_t]
        
        #Sorting in energy ascending order 
        idx = [np.argsort(e) for e in ev]
        EV = np.array([E[:,idx[n]] for n, E in enumerate(EV)])
        ev = np.array([e[idx[n]] for n, e in enumerate(ev)])
        res = (np.squeeze(ev), np.squeeze(EV))            
        return res
    
    

def evolve(conf):
    """ Evolve 
    conf: (quspin.Hamiltonian, state, time)
    """
    return conf[0].evolve(conf[1], t0=0, times = conf[2])