#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import sys 
import numpy as np
import matplotlib.pylab as plt
import pdb
import importlib as ilib

if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import RandomGenerator as rdm
    from QuantumSimulation.Utility.Optim import pFunc_base as pfbase
    from QuantumSimulation.Utility.Optim import pFunc_zoo as pfzoo

else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import RandomGenerator as rdm
    from ..Utility.Optim import pFunc_base as pfbase
    from ..Utility.Optim import pFunc_zoo as pfzoo



class model_base:
    """Base class for (quantum) models i.e. define what a model / simulation is.
    A model is comprised of an Hamiltonian (self._H), an underlying state space
    (self._ss), time characteristics s.a horizon self.T, ). 
    
    Evolving the system: from an initial state (self.state_init) make it evolve according 
    to an Hamiltonian (self._H) up to final time (horizon self.T) while recording intermediate
    state if 'flag_intermediate' is True

    Simulating the system is understood as evolving the system and computing some figure of merit
    (a choice has be made to make this computation of fom part of the model_base).

    Optionally one can provide a random_obj and noise_obj

    TODO: implement mprocessing capabilities the same way as random
    """
    _LIST_ARGS = {'T':'<float> Horizon of the simul', 'dt':'<float> time step', 
                 'flagInterTime':'<bool> allows the use of intermediate times', 
                 'state_init':'<string/np.array> Initial state'}
    
    _LIST_ARGS_OPT = {'rdm_obj': '<int/randomstate/None> Use to generate/pass the random generator',
                     'noise':'<dict<string>> to build noise in the model ', 
                     'state_target': '<string/np.array> Target state'}
    
    def __init__(self, **args_model):
        """ 
        """
        self._ss = None # underlying state space
        self._H = None # Hamiltonian
        self._rdmgen = None #random generator
        self._fom_func = None # functions available to compute FOM
        self._t_array = None #
        self.pop_t = None
        self.t_simul = None # time array of the simulation
        self.T = None #Horizon of the simulation
               
        self._setup_time(**args_model)
        self.rdm_gen = args_model.get('rdm_obj')
        self.noise = args_model.get('noise')
        self.state_init = args_model['state_init']
        self.state_tgt = args_model.get('state_tgt')
        self.main_fom = args_model.get('fom') # functions available to compute FOM


    def set_up_fom_basic_bricks(self):
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
            elif(len(np.shape(V)) == 1)
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
        

    @property
    def state_init(self):
        retun self._state_init 
    
    @property.setter
    def state_init(self, init)
        self._state_init = self.get_state(init)

    @property
    def state_tgt(self):
        retun self._state_init 
    
    @property.setter
    def state_tgt(self, tgt)
        self._state_tgt = self.get_state(tgt)

    @property
    def state(self):
        if(hasattr(self, '_state_t')):
            return self._state_t 
        else:
            return None

    @property.setter
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
    def rdm_gen(self):
        return self._rdmgen 
    
    @property.setter
    def rdm_gen(self, rdm_obj)
        self._rdmgen = rdm.init_random_generator(rdm_obj)

    @property
    def noise(self):
        return self._noise_func
    
    @property.setter
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
                res = self.rdm_gen.gen_rdmfunc_from_string(v)
            except:
                res = eval()
        elif(ut.is_callable(noise_obj)):
            res = noise_obj
        else:
            raise NotImplementedError
     
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
    @ut.extend_dim_method
    def _compute_fom(self, fom, st=self.state):
        """Compute a potentially composed FOM (or list of FOM)
            i.e. fom = ['lst:f2t:neg:0.3, last:f2t2'] """        
        components = ut.splitString(fom)
        res = np.sum([self._compute_atom_fom(c, st) for c in components])
        return res       


    def _compute_atom_fom(self, fom, st=self.state):        
        """Compute a fom by composing functions
            e.g. ComputeFOMAtom(state, 'lst:f2t:neg:0.3') >> 0.3 * neg(f2t(lst))
            i.e. 0.3*(1 - f2t(st[-1])) i.e. inverse of the fidelity 
            computed on the last state.
        """                
        f2apply = [self._fom_func.get(k, lambda x: float(k)*x) for k in ut.splitString(fom, ":")]
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
    _LIST_ARGS = Models._LIST_ARGS
    _LIST_ARGS['control_obj'] = "<callable> or <dict> or <str>"
    
    def __init__(self, controlFun, **args_model):
        """ init model and control functions (stored as a list) """
        Models.__init__(self, **args_model)
        self.control_fun = args_model('control_fun')
        self._setup_fom_controlfun_bricks()

       def _setup_fom_controlfun_bricks(self):
        """ populate the dictionary self._fom_func of functions relating to the controlfun 
        cf. doc self._setup_fom_basic_bricks
        workaround: lambda function with one input x which is not used
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

    @property.setter
    def control_fun(self, control):
        list_fun = self._process_control_function(control)
        if(not(ut.is_iter(funs))):
            list_fun = [list_fun]
        self._control_fun = list_fun

    @property
    def n_controls(self):
        return len(self.control_fun)
    
    def get_one_control_fun(self, index = 0):
        return self.control_fun[index]

    def _process_control_function(self, control):
        """ control_fun should be a (list of) callable(s)"""
        if(ut.is_callable(control)):
            return control
        if(ut.is_iter(contol)):
            return [self._process_control_function(c) for c in control]

    def _get_info_control(self, info_type = 'fluence', t_array = self.t_array, index = None, func_wrap = None):
        """ compute info on the controlfluenc of the control function (or of wrapped control functions) 
        Provide some flexibility on which function(s) to consider and on potential
        wrapping of the function (e.g. may be intereseted in the fluence of 1 - controlFun) """
        if info_type == 'fluence':
            info_func = self._fluence
        elif info_type == 'smoothness':
            info_func = self._smoothnes

        if((index is not None) or (self._nb_control_func <= 1)):
            func_tmp = self.get_one_control_fun(index)
            if(func_wrap is not None):
                func_tmp_wrapped = [lambda x: func_wrap(func_tmp(x))]
            else:
                func_tmp_wrapped = [func_tmp]        
            
        else:
            func_tmp = self.controlFun
            if(func_wrap is not None):
                if(ut.is_iter(func_wrap)):
                    func_tmp_wrapped = [lambda x: func_wrap[i](f(x)) for f in func_tmp]
                else:
                    func_tmp_wrapped = [lambda x: func_wrap(f(x)) for f in func_tmp]
            else:
                func_tmp_wrapped = func_tmp

        res = np.sum([info_func(f, t_array) for f in func_tmp_wrapped])
        return res
    
    def get_controlFun_t(self, t_array = None):
        if(t_array is None):
            t_array = self.t_array
            
        if(self._nb_control_func == 0):
            res = None
        elif(self._nb_control_func == 1):
            res = np.array([self.controlFun(t) for t in t_array])
        else:
            res = np.array([[cf(t) for cf in self.controlFun] for t in t_array])
        return res

    def _smoothness(self, ):
        """ could be impl here but will be in children classes"""
        raise NotImplementedError()

    def _fluence():
        """ could be impl here but will be in children classes """
        raise NotImplementedError
    

class pcModel_base(cModel_base):
    """ parametrized control models """

    # Available type of parametrized control functions
    _LIST_CUSTOM_FUNC = pfzoo.pFunc_factory._LIST_CUSTOM_FUNC

    def __init__(self, **args_model):
        cModel_base.__init__(self, **args_model)

    def _process_control_function(self, control):
        """ delegate everything to the capability of pFunc_zoo.pFunc_factory 
        store the dico (as a list of dicos allowing to rebuild the function)
        TODO: (probably) remove dico_control
        """
        control, dico_control = pf.pFunc_factory.build_function(control, True, self.rdm_gen)
        self._dico_control = dico_control
        return control
            

    def update_control_parameters(self, params, index_control = None,  **args_update):
        """ Update the parameters of the control function(s) i.e. the thetas (free params)
        of the functions 
        TODO: implement the use of index_control (i.e. choose a (a list) of which func to be updated)
        """
        if(args_update.get(debug)):
            pdb.set_trace()

        if(index_function is None):
            self.control_fun.thetas = params
        
        else:
            raise NotImplementedError


    def __call__(self, params, **args_call):
        """ model(params) >> fom
        Should be used at some point but first need to implement logs"""
        self.update_control_parameters(params, **args_call)
        res = self.Simulate(**args_call)
        if(ut.is_iter(res)):
            res = res[0]
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
        self._setup_helper_functions()
        self._setup_fom_qspin_bricks()

    
    #-----------------------------------------------------------------------------#
    # Setup functions
    #-----------------------------------------------------------------------------#
    def _setup_basis(self, **args_model):
        """ build the basis (self._ss) and potentially other basis"""
        raise NotImplementedError()
        
    def _setup_H(self, **args_model):
        """ build the Hamiltonian governing the dynamics of the system (self.H)"""
        raise NotImplementedError()

    def _setup_helper_functions(self):
        """ create an attribute helper which is a dico<'name_help_func':func> s.t.
        these helper functions are available through self.helper.h_XXXX 
        simply to make things more readable (isn't it??) """
        helper_dico={'ip': self._h_ip, 'norm':self._h_norm, 'fid':self._h_fid, 
        'fid2': self._h_fid2, 'last':self._h_last, 's_to_p':self._h_state2proba,
        'var':self._h_variance, 'var2': self._h_variance2, 'n_meas': self._h_n_measures,
        'le': self._h_get_lowest_energies, 'proj_instant': self._h_project_to_instant_evect
        'fid2_tgt': self._h_fid2, 'fid_tgt':self._h_fid_tgt}
        
        self.helper = helper_dico
        self.helper_infos = {k:v.__doc__ for k,v in helper_dico.items()}

        
    def _setup_fom_qspin_bricks(self):
        """ Set up fom based of quspin states """
        #fidelity to the target state
        self._fom_func['f2t'] =  (lambda x: self.helper.fid_tgt(x))
        self._fom_func['f2t2'] =  (lambda x: self.helper.fid2_tgt(x))

        # measurement projection on the target_state 
        self._fom_func['proj5'] = (lambda x: self.helper.n_meas_tgt(x, nb =5))
        self._fom_func['proj10'] = (lambda x: self.helper.n_meas_tgt(x, nb =10))
        self._fom_func['proj100'] = (lambda x: self.helper.n_meas_tgt(x, nb =100))
        self._fom_func['proj1000'] = (lambda x: self.helper.n_meas_tgt(x, nb =1000))

    def get_state(self, state_obj = None):
        """ Generate quantum states from state_obj <str> or <array/list<num>>"""
        basis = self._ss
        if(state is None):
            state_res = None

        elif(isinstance(state, str)):
            #pdb.set_trace()
            if(state == 'GS_i'):
                # GS at t=0
                _, state_res = self.H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
            elif(state == 'GS_f'):
                #GS at t = T
                _, state_res = self.H.eigsh(time = self.T, k=1, which='SA',maxiter=1E10) 
            elif(state == 'uniform'):
                #GS at t = T
                state_res = np.random.uniform(-1, 1, size=basis.Ns)
                state_res = state_res / self.helper.norm(state_res)
            else:
                i_res = basis.index(state)
                state_res = np.zeros(basis.Ns, dtype=np.float64)
                state_res[i_res] =1.0
        else:
            assert(len(state) == basis.Ns), 'wrong size'
            state_res = np.array(state)

        return np.squeeze(state_res)

    def _state_to_pop(self, st):
        """ compute pop from state"""
        self.helper.s_to_p(st)
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
    
    def _h_fid2_tgt(V1):
        """ compute fidelity (square convention) between V1 and state_tgt"""
        return pcModel_qspin._h_fid2(V1, self.state_tgt)

    def _h_fid_tgt(V1):
        """ compute fidelity between V1 and state_tgt"""
        return pcModel_qspin._h_fid(V1, self.state_tgt)

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
    
    @ut.extend_dim_method(n_dim=1, array_output = True)
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
        proba = pcModel_qspin._h_probaFromState(proj)
        assert(np.sum(proba)<=(1.0 + num_precis)), "not a valid proba distrib" #should it be relaxed 
        proba_cum = np.concatenate(([0],np.cumsum(proba)))
        samples = np.random.sample(nb)
        observations = [np.sum((samples > proba_cum[i]) * (samples <= proba_cum[i+1])) for i in index_measure]
        frequencies = np.array(observations)/nb
        if(single_value):
            assert(len(frequencies) == 1), "pb"
            frequencies = frequencies[0]
        return frequencies    
    
    @static
    def _h_n_measures_to_target(ket1, nb=1, num_precis=1e-6):
        """ _h_n_measures with measur_basis = self.state_tgt """
        return _h_n_measures_to_target(ket1, nb, self.state_tgt, num_precis)

    @ut.extend_dim_method()
    def _h_get_lowest_energies(self, time, nb = 2, H = self.H):
        """ Get the <nb> lowest energies of the Hamiltonian <H> at a time <time>
        args = (time <numeric> or <list>, nb=2 <integer>, H = self.H <QuSpin.Hamiltonian>)
        """
        res, _ = H.eigsh(time = time, k=nb, which='SA',maxiter=1E10)
        return res    
    
    @ut.extend_dim_method()
    def _h_project_to_instant_evect(self, time = self.t_simul, H = self.H, state_t = self.state_t, nb_ev = 5):
        """ Project a state <state_t> on the <nb_ev> first eigenvectors of <H> at <time>
        args = (time = self.t_simul <list<num>> or <num>, H = self.H <QuSpin.Hamiltonian>
                state_t = self.state_t <np.array>, nb_ev = 5 <int>)
        """
        en, ev = H.eigsh(time = time, k=nb_ev, which='SA',maxiter=1E10)
        pop_ev = np.square(np.abs(pcModel_qspin._h_ip(state_t, ev)))
        return np.array(pop_ev)
    

    #-----------------------------------------------------------------------------#
    # plot capabilities
    #-----------------------------------------------------------------------------#
    def plot_pop_adiab(self, **args_pop_adiab):
        """ Plot pop adiab where each population_t is dispatched on one of the 
        three subplot
        #TODO: better plots
        """
        if(hasattr(self,'pop_adiab')):
            pop_adiab = self.pop_adiab
            t_simul = self.t_simul
            energies = self.energies
            cf = self.get_control_t(t_simul)
            nb_levels = pop_adiab.shape[1]    
            f, axarr = plt.subplots(2,2, sharex=True)
            
            axarr[0,0].plot(t_simul, cf, label = 'f(t)')
            for i in range(nb_levels):
                pop_tmp = pop_adiab[:, i]
                max_tmp = np.max(pop_tmp)
                if(max_tmp > 0.1):
                    axarr[0,1].plot(t_simul, pop_tmp, label = str(i))
                    axarr[1,0].plot(t_simul, energies[:, i], label = str(i))
                elif(max_tmp > 0.01):
                    axarr[1,1].plot(t_simul, pop_tmp, label = str(i))
                    axarr[1,0].plot(t_simul, energies[:, i], label = str(i))
            
            axarr[0,1].legend()
            axarr[1,1].legend()
            axarr[1,0].legend()
            axarr[0,0].legend()

                        
        else:
            _ = self.EvolutionPopAdiab(**args_pop_adiab)
            self.plot_pop_adiab()
        

