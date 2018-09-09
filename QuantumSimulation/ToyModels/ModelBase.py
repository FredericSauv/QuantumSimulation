#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import sys, copy, pdb, time, logging
from functools import partial
import operator as op
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
    """Base class for (quantum) models i.e. define what a model / simulation is.
    A model is comprised of an Hamiltonian (self._H), an underlying state space
    (self._ss), time characteristics (i.e. horizon self.T, ), and computing some
    figure of merits (fom)
    
    Evolving the system: from an initial state (self.state_init) make it evolve according 
    to an Hamiltonian (self._H) up to final time (horizon self.T) while recording intermediate
    state if 'flag_intermediate' is True. 
    
    When states are over time they are given as H X T 

    Simulating the system is understood as evolving the system and computing some figure of merit
    (a choice has be made to make this computation of fom part of the model_base).

    Optionally one can provide a random_obj and noise_obj

    TODO: implement mprocessing capabilities the same way as random
    """
    # {<str:name_mandatory_param> : <str: info_opt_param>}
    _LIST_ARGS = {'T':'<float> Horizon of the simul', 'dt':'<float> time step', 
                 'flag_intermediate':'<bool> allows the use of intermediate times', 
                 'state_init':'<string/np.array> Initial state'}
    
    # {<str:name_opt_param> : (<str: info_opt_param>, <var:default_value>)}
    _LIST_ARGS_OPT = {'rdm_obj':('<int/randomstate/None> Use to generate/pass the random generator', None),
                     'mp_obj':('<int/bool/mp> Use to generate/pass the random generator', None),
                     'noise':('<dict<string>> to build noise in the model ',None), 
                     'fom':('fom <str> or <list<str>>, has', None),
                     'state_init':('<string/np.array> Initial state',None)}
    
    def __init__(self, **args_model):
        """ Attributes:
            
        -- noise -- 
            noise input {'name_noise':<callable> or <string> <list<string>>,..}
            where a string can be used to generate a random value i.e. uniform_0_1
            cf. RandomGenerator.gen_rdmfunc_from_string
            """
        self._ss = None # underlying state space
        self._H = None # Hamiltonian
        self._rdmgen = None #random generator
        self._fom_func = {} # functions available to compute FOM
        self._t_array = None #
        self.pop_t = None
        self.t_simul = None # time array of the simulation
        self.T = None #Horizon of the simulation
               
        self._setup_time(**args_model)
        self.rdm_gen = args_model.get('rdm_obj')
        self._setup_fom_basic_bricks()
        self.mp = args_model.get('mp_obj')
        self.noise = args_model.get('noise')
        self.fom = args_model.get('fom') 
        self._fom_print = args_model.get('fom_print', False)
        
        # self.state_init = args_model['state_init']
        # self.state_tgt = args_model('state_tgt')
        # in s



    def _setup_fom_basic_bricks(self):
        """ populate the dictionary self._fom_func of functions which can be used
        to compose fom measures 
        e.g. fom = 'last:f2t:square:rdmplus' stands for taking the last state ('last')
         compute fidelity to target ('f2t' not implemented yet), square it ('square') 
         and finaly add some randomness ('rdmplus' whih require that some _noise_func['fom'] 
         exists).
        """
        def last(V):
            if(len(np.shape(V)) == 2):
                return V[:, -1]
            elif(len(np.shape(V)) == 1):
                return V
            else:
                raise NotImplementedError

        def neg(x):
            return 1 - x

        self._fom_func['max'] = np.max
        self._fom_func['min'] = np.min
        self._fom_func['avg'] = np.average
        self._fom_func['sqrt'] = np.sqrt
        self._fom_func['square'] = np.square
        self._fom_func['last'] = last
        self._fom_func['neg'] = neg

        #To add randomness
        self._fom_func['rdmplus'] = (lambda x: x + self._noise_func['fom']())
        self._fom_func['rdmtime'] = (lambda x: x * (1 + self._noise_func['fom']()))

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
        self._mp = mp.init_mp(mp_obj)

    @property
    def noise(self):
        return self._noise_func
    
    @noise.setter
    def noise(self, noise):
        if noise is None:
            self._noise_func = None
        elif(ut.is_dico(noise)):
            self._noise_func = {k:self._gen_noise_function(v) for 
                                k, v in noise.items()}
        else:
            raise NotImplementedError()

    def _gen_noise_function(self, noise_obj):
        """ allow generation of function based on different type of inputs"""
        if(ut.is_str(noise_obj)):
            try:
                res = self.rdm_gen.gen_rdmfunc_from_string(noise_obj)
            except:
                res = eval()
        elif(ut.is_callable(noise_obj)):
            res = noise_obj
        else:
            raise NotImplementedError
        return res
        
    def _setup_time(self, **args_model):
        """ Generate time attributes of the model."""
        self.T = args_model['T']
        self.dt = args_model['dt']
        self._flag_intermediate = args_model['flag_intermediate']
        
        self.t_array = np.concatenate((np.arange(0, self.T, self.dt), [self.T]))
        if(self._flag_intermediate):
            self.t_simul = self.t_array
        else:
            self.t_simul = self.T
           
     
    ### LOGIC FOR COMPUTING THE FOM
    @ut.extend_dim_method(0, True)
    def _compute_fom(self, fom = None, st = None):
        """Compute a potentially composed FOM (or list of FOM)
            i.e. fom = ['lst:f2t:neg:0.3, last:f2t2'] """    
        if fom is None:
            fom = self.fom
        if st is None:
            st = self.state
        components = ut.splitString(fom)
        
        res = np.sum([self._compute_atom_fom(c, st) for c in components])
        return res       


    def _compute_atom_fom(self, fom, st):        
        """Compute a fom by composing functions
            e.g. ComputeFOMAtom(state, 'lst:f2t:neg:0.3') >> 0.3 * neg(f2t(lst))
            i.e. 0.3*(1 - f2t(st[-1])) i.e. inverse of the fidelity 
            computed on the last state.
            AWFULL NO??
        """                
        #f2apply = [self._fom_func.get(k, lambda x: float(k)*x) for k in ut.splitString(fom, ":")]
        f2apply = []
        for k in ut.splitString(fom, ":"):
            f_tmp = self._fom_func.get(k)
            if(f_tmp is None):
                f_tmp = partial(op.mul, float(k))
            f2apply.append(f_tmp)
    
        res = ut.compoFunctions(f2apply, st, order = 0)
        return res 

    
    ###TO BE IMPLEMENTED IN THE CHILD CLASSES
    def get_state(self, state):
        """ Return a state based on some string/array/other format"""
        raise NotImplementedError()      
            
    def _state_to_pop(self, st):
        """ compute pop from state"""
        raise NotImplementedError()

    def Simulate(self, **args):
        """ Main entry point to simulate the system: evolves the init state and 
        performs some other actions (typically computing a figure of Merit or other 
        statistic)
        """
        raise NotImplementedError()            

    def Evolution(self, **args):
        """ Evolve the init_state according to the relevant method and store if
            required
        Output:
            state_t, 
        """
        raise NotImplementedError()      


class cModel_base(model_base):
    """ Base class for controlled models i.e. simulations depending of a
    (some) time-dependent control function(s)"""
    _LIST_ARGS = model_base._LIST_ARGS
    _LIST_ARGS['control_obj'] = "<callable> or <dict> or <str>"
    
    def __init__(self, **args_model):
        """ init model and control functions (stored as a list) """
        model_base.__init__(self, **args_model)
        self.control_fun = args_model['control_obj']
        self._setup_fom_controlfun_bricks()
        self._aggregated_nb_call = 0

    def _setup_fom_controlfun_bricks(self):
        """ populate the dictionary self._fom_func of functions relating to the controlfun 
        cf. doc self._setup_fom_basic_bricks
        workaround: lambda function with one input x which is not used
        remark: rare case where t_array is used and not t_simul
        """
        self._fom_func['fluence'] =  (lambda x: 
            self._get_info_control('fluence', self.t_array, None, None))
        self._fom_func['smooth'] = (lambda x: 
            self._get_info_control('smoothness', self.t_array, None, None))

    #-----------------------------------------------------------------------------#
    # Management of the contro_fun (a list)
    #-----------------------------------------------------------------------------#
    @property
    def control_fun(self):
        return self._control_fun

    @control_fun.setter
    def control_fun(self, control):
        list_fun = self._process_control_function(control)
        if(not(ut.is_iter(list_fun))):
            list_fun = [list_fun]
        self._control_fun = list_fun
        self._update_model_after_change_of_control_fun()
        
    def _update_model_after_change_of_control_fun(self):
        """ Extra stuff to do when the control_fun is replaced by another one"""
        pass

    @property
    def n_controls(self):
        return len(self.control_fun)
    
    def get_one_control_fun(self, index = 0):
        return self.control_fun[index]

    def _process_control_function(self, control):
        """ control_fun should be a (list of) callable(s)"""
        if(ut.is_iter(control)):
            return [self._process_control_function(c) for c in control]
        elif(ut.is_callable(control)):
            return control
        else:
            raise SystemError("couldn't use type {0} to create the control_functions "
                "in class {1}").format(type(control), self.__class__) 

    def _get_info_control(self, info_type = 'fluence', time = None, index = None, func_wrap = None):
        """ compute info on the controlfluenc of the control function (or of wrapped control functions) 
        Provide some flexibility on which function(s) to consider and on potential
        wrapping of the function (e.g. may be intereseted in the fluence of 1 - controlFun) """
        if(time is None):
            time = self.t_array
        if info_type == 'fluence':
            info_func = self.__class__._fluence
        elif info_type == 'smoothness':
            info_func = self.__class__._smoothness

        if(index is not None):
            func_tmp = self.control_fun[index]
            if(func_wrap is not None):
                func_tmp_wrapped = [lambda x: func_wrap(func_tmp(x))]
            else:
                func_tmp_wrapped = [func_tmp]        
        else:
            func_tmp = self.control_fun
            if(func_wrap is not None):
                if(ut.is_iter(func_wrap)):
                    func_tmp_wrapped = [lambda x: func_wrap[i](f(x)) for i, f in enumerate(func_tmp)]
                else:
                    func_tmp_wrapped = [lambda x: func_wrap(f(x)) for f in func_tmp]
            else:
                func_tmp_wrapped = func_tmp

        res = np.sum([info_func(f, time) for f in func_tmp_wrapped])
        return res
    
    def get_control_fun_t(self, t_array = None):
        if(t_array is None):
            t_array = self.t_array
            
        if(self.n_controls == 0):
            res = None
        elif(self.n_controls == 1):
            res = np.array([self.control_fun[0](t) for t in t_array])
        else:
            res = np.array([[cf(t) for cf in self.control_fun] for t in t_array])
        return res

    @staticmethod
    def _smoothness(func, time):
        """ Smoothness as avg <([f(x) - f(x + dx)]/dx) ** 2> """
        time_copy = time.copy()
        
        time_copy[0] = time_copy[0]-1e-6
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
    """ parametrized control models """

    # Available type of parametrized control functions
    _LIST_CUSTOM_FUNC = pfzoo.pFunc_factory._LIST_CUSTOM_FUNC

    def __init__(self, **args_model):
        cModel_base.__init__(self, **args_model)
        self._flag_track_calls_void = True


    @property
    def n_params(self):
        return self.control_fun.n_theta

    @property
    def params_bounds(self):
        return self.control_fun.theta_bounds
    
    #TODO: Use pFunc_collec instead of list
    def _process_control_function(self, control):
        """ clone the function passed  """
        
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
        """ delegate everything to the capability of pFunc_zoo.pFunc_factory 
        store the dico (as a list of dicos allowing to rebuild the function) """
        
        if(isinstance(control, pf.pFunc_base)):
            res = control.clone()
        else: 
            res = pfzoo.pFunc_factory.build_custom_func(control, rdm_gen = self.rdm_gen) 
        return res

    def update_control_parameters(self, params, index_control = None,  **args_update):
        """ Update the parameters of the control function(s) i.e. the thetas (free params)
        of the functions """
        if(args_update.get('debug')):
            pdb.set_trace()

        if(index_control is None):
            self.control_fun.theta = params
        else:
            self.control_fun[index_control].theta = params


    def __call__(self, params, trunc_res = True, **args_call):
        """ model(params) >> fom (just one value)
        Should be used at some point but first need to implement logs"""
        self._aggregated_nb_call += 1
        args_call_dupl = copy.copy(args_call)
        track = args_call_dupl.pop('track_learning', False)
        self.update_control_parameters(params, **args_call_dupl)
        res_tmp = self.Simulate(**args_call_dupl)
        if(self._fom_print):
            logger.info(res_tmp)
        if(trunc_res and ut.is_iter(res_tmp)):
            res = res_tmp[0]
        else:
            res = res_tmp

        if track:
            if self._flag_track_calls_void:
                if(not(hasattr(self, '_timer'))):
                    self._timer = time
                self._track_time_init = self._timer.time()
                self._track_fom = None
                self._flag_track_calls_void = False
                
                self._track_calls = {'history_nev_fun':[], 'history_time_fun':[],
                    'history_func_fun':[], 'best_fun':None, 'best_fun_full':None, 
                    'history_nev_params':[], 'history_nev_time':[]}
                                
            best = self._track_calls['best_fun']
            if (best is None) or (best > res):
                time_elapsed = self._timer.time() - self._track_time_init
                self._track_calls['best_fun'] = res
                self._track_calls['best_fun_full'] = res_tmp
                self._track_calls['history_nev_fun'].append([self._aggregated_nb_call, res])
                self._track_calls['history_time_fun'].append([time_elapsed, res])
                self._track_calls['history_func_fun'].append([repr(self.control_fun), res])
                self._track_calls['history_nev_params'].append([self._aggregated_nb_call, params])

        return res


    
class pcModel_qspin(pcModel_base):
    """ Models based on the QuSpin package. inherit from the parametrized control
    function base class. On top of the that implement:
    + Building the state space (equiv to the basis objects in QuSpin)
    + Build the Hamiltonian
    + Define new building blocks for computing the fom
    + provide helper functions to act on QuSpin objects self.h_XXX
      e.g. self.h_ip for the inner product
    """
    def __init__(self, **args_model):
        """ """
        pcModel_base.__init__(self, **args_model)
        self._setup_basis(**args_model)
        self._setup_H(**args_model)
        self._setup_fom_qspin_bricks()

    def _update_model_after_change_of_control_fun(self):
        """ Extra stuff to do when the control_fun is replaced by another one:
            Regenerate H"""
        if(hasattr(self, '_H') and (self._H is not None)):
            self._setup_H()
    
    # --------------------------------------------------------------------------- #
    #   SIMULATIONS 
    #   TODO: Maybe put it in quSpin models
    # --------------------------------------------------------------------------- #
    def Simulate(self, time = None, state_init = None, fom = None, store = None, method = None, **extra_args):
        """ Main entry point to simulate the system. If fom is not None, it will 
        return it, if not return the state_t of the system.
        """
        if extra_args.pop('debug', None):
            pdb.set_trace()
        if time is None:
            time = self.t_simul
        if state_init is None:
            state_init = self.state_init
        if fom is None:    
            fom = self.fom
        if store is None:
            store = False
        if method is None:
            method = 'se'
        
        res = self.Evolution(time = time, state_init = state_init, method = method, store = store, **extra_args)

        if (fom not in [None, '']):
            res = self._compute_fom(fom, res)
        if(extra_args.get('print')):
            logger.info("FOM="+str(res))
        return res

    def Evolution(self, time , state_init, method, store, **extra_args):
        """  Evolve the state_init according to the relevant method and store if
            required
        state_t has dim (from quspin) [dim_H, dim_t]
        """
        if(method == 'se'):
            state_t = self.EvolutionSE(time, state_init, **extra_args)

        elif(method == 'pop_adiab'):
            state_t = self.EvolutionPopAdiab(time, state_init, **extra_args)

        else:
            raise NotImplementedError()

        if store:
            if(method == 'pop_adiab'):
                logger.warning("adiabatic population has been stored as state_t")
            self.state_t = state_t
        
        return state_t

    # --------------------------------------------------------------------------- #
    #   Custom evolutions
    # --------------------------------------------------------------------------- #
    def EvolutionSE(self, time = None, state_init = None, iterable = False, **args_evolve):
        """  Wrap evolve from QuSpin only expose some of teh arguments
        hamiltonian.evolve(state_init, t0 = 0,times = T, eom="SE",solver_name="dop853",stack_state=False,
        verbose=False,iterate=False,imag_time=False,**solver_args)
        
        state_t has dim (from quspin) [dim_H, dim_t]
        """
        if time is None:
            time = self.t_simul
        if state_init is None:
            state_init = self.state_init

        state_t = self._H.evolve(state_init, t0 = 0, times = time, iterate=iterable, **args_evolve)
        return state_t


    def EvolutionPopAdiab(self, time = None, state_init = None, nb_ev = 2, **args_evolve):
        """ Evolve the state according to SE and project it on the instantaneous 
        eigen vectors of the hamiltonian. This pop_adiab is stored by default and
        state_t is returned """
        if time is None:
            time = self.t_array
        if state_init is None:
            state_init = self.state_init
        n_t = len(time)
        state_t = self.EvolutionSE(time, state_init, **args_evolve)
        # Not optimal
        ev, EV = self._h_get_instantaneous_ev_EV(time=time, nb_ev=nb_ev)


        try:
            assert state_t.shape[1] == n_t
        except AssertionError as err:
            logger.exception("pb dim ")
            raise err

        proj_ev = [[pcModel_qspin._h_ip(state_t[:, t], EV[t][:, n]) for n in range(nb_ev)] for t in range(n_t)]
        self.adiab_pop = np.square(np.abs(np.array(proj_ev)))
        self.adiab_cf = np.array([self.control_fun(t) for t in time]) 
        self.adiab_en = np.array(ev) # both energies and control field
        self.adiab_evect = np.array(EV)
        self.adiab_t = time
        return state_t
    
    def FindMinDelta(self, time = None, state_init = None, level_min = 0, level_max=1, **args_evolve):
        """ find minimal gap over time where the gap is the difference between two 
        instantaneous energies """
        if time is None:
            time = self.t_array
        if state_init is None:
            state_init = self.state_init

        gap = np.Inf
        for t in time:
            if((level_max+1) < self._H.Ns): 
                e_tmp, _ = self._H.eigsh(time=t, k=level_max+1, which='SA',maxiter=1E10)
            else:
                e_tmp, _ = self._H.eigh(time=t)
            gap_tmp = e_tmp[level_max] - e_tmp[level_min]
            if(gap_tmp < gap):
                gap = gap_tmp


        return gap

    
    #-----------------------------------------------------------------------------#
    # plot capabilities
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
            ax_tmp.set(xlabel='t')
            
            ax_tmp = axarr[1,1]
            if(second_pop_populated):
                ax_tmp.legend(fontsize = 'small')
            
            ax_tmp = axarr[0,0]
            ax_tmp.legend()
            ax_tmp.set(xlabel='t')
            
            ax_tmp = axarr[1,0]
            if(plot_gap):
                try:
                    ax_tmp.set(xlabel='t', ylabel=r"$E_i - E_0$")
                    diff_01 = en[:, 1] - en[:, 0]
                    index_min = np.argmin(diff_01)
                    t_min = t[index_min]
                    y_min= diff_01[index_min]
                    ax_tmp.arrow(t_min, 0, 0, y_min)
                    ax_tmp.text(t_min - 2.2 , (y_min/2), r"$\Delta = %.2f$"%(y_min), fontsize=8)
                except:
                    pass
            
            

        
        
            save_fig = args_pop_adiab.get('save_fig')
            if(ut.is_str(save_fig)):
                f.savefig(save_fig)

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
                
                _, state_res = self._h_get_instantaneous_ev_EV(time=[t], nb_ev=n_ev+1)
                state_res = state_res[0]
                state_res = state_res[:, n_ev] if n_ev > 0 else state_res
            
            elif(state_obj[:3] == 'ESS'):
                # Energy SubSPace e.g. 'EES_0_0.0' is the first Energy Eigen 
                # State at t = 0
                n_ev, t = state_obj[4:].split('_')
                n_ev = int(n_ev)
                t = float(t)
                
                _, state_res = self._h_get_instantaneous_SS(time=t)
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
    def _h_last(V):
        """ return last element of a state_t """
        return V[:, -1]
    
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
    
    #@ut.extend_dim_function(1, True)
    @staticmethod
    def _h_n_measures(ket1, nb = 1, measur_basis = None, num_precis = 1e-6):        
        """Frequencies of <nb> projective measurements of the state <ket1>  
        in  <measur_basis> (by default the basis in which the ket is represented) 
        
        args = (ket1 <1D-2D np.array>, nb = 1 <int>, measur_basis = <N x D np.array> 
        N number of vectors, D dim of the H-space, num_precis = 1e-6)        
        #TODO: Could probably do better (concise) + ortho measurement only
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
    
    @ut.extend_dim_method(0, True)
    def _h_get_lowest_energies(self, time, nb = 2):
        """ Get the <nb> lowest energies of the Hamiltonian <H> at a time <time>
        args = (time <numeric> or <list>, nb=2 <integer>, H = self.H <QuSpin.Hamiltonian>)
        """
        if(nb < self._H.Ns):
            res, _ = self._H.eigsh(time = time, k=nb, which='SA',maxiter=1E10)
        else:
            res, _ = self._H.eigh(time = time)
        return res    
    
    
    def _h_get_instantaneous_SS(self, time=None):
        """ get instantaneous eigen values SubSpaces """
        if(time is None):
            time = self.t_array ## Should it t_simul
        ## Carefull this method can be inacurate if too many eigenvectors/vals are requested
        ## cf. quspin package comments
        ev, EV = self._H.eigh(time=time)
        idx = np.argsort(ev)
        EV = EV[:,idx]
        ev = ev[idx]
        
        ev_SS, EV_SS, EV_tmp = [], [], []
        ref = None
        
        for e, E in zip(ev, EV):
            if (ref is None):
                ref = e
                ev_SS.append(e)
                EV_tmp.append(E)
            
            elif (np.abs(ref - e) < 1e-8):
                EV_tmp.append(E)
                
            else:
                EV_SS.append(EV_tmp)
                EV_tmp = [E]
                ev_SS.append(e)
                ref = e
        if len(EV_tmp)>0:
            EV_SS.append(EV_tmp)
        return (ev_SS, EV_SS)
    
    def _h_get_instantaneous_ev_EV(self, time=None, nb_ev=5):
        """ get instantaneous eigen values and eigen vectors """
        if(time is None):
            time = self.t_array ## Should it t_simul
        ## Carefull this method can be inacurate if too many eigenvectors/vals are requested
        ## cf. quspin package comments
        if(nb_ev < self._H.Ns): 
            eigen_t = [self._H.eigsh(time=t, k=nb_ev, which='SA',maxiter=1E10) for t in time]
        else:
            eigen_t = [self._H.eigh(time=t) for t in time]
        ev = [ev for ev, _ in eigen_t]
        EV = [EV for _, EV in eigen_t]
        
        idx = [np.argsort(e) for e in ev]
        EV = [E[:,idx[n]] for n, E in enumerate(EV)]
        ev = [e[idx[n]] for n, e in enumerate(ev)]
        
        return (ev, EV)