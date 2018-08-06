#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""

import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import matplotlib.pylab as plt
import pdb
import importlib as ilib

if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import pFunc_base as pf
    from . import ModelBase as mod


ilib.reload(ut)
ilib.reload(mod)


class Qubits(mod.pcModel_qspin):
    """ Simulate a 1D - MH model. Implementation of the pcModel_qspin """
    _LIST_ARGS = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS['setup'] = '<str> string pointing to one of the setup'
    
    _LIST_ARGS_OPT = mod.pcModel_qspin._LIST_ARGS
    # _LIST_ARGS_OPT['sym'] = ('<dict> definition of the symmetries to use c.f. quspin doc', {}) 
    _LIST_ARGS_OPT['flag_store'] = ('<bool> Allow the storage of intermediate results', False)  

    def __init__(self, **args_model):
        """ Initialize the simulations of the driven BH  """
        mod.pcModel_qspin.__init__(self, **args_model)
        self._FLAG_STORE = args_model.get('flag_store', False)
        self.state_init = args_model['state_init']
        self.state_tgt = args_model.get('state_tgt')


    def _setup_basis(self, **args_model):
        """ Build and store:
        - the basis (self._ss) and the basis without use of symmetries()
        - config_system
        """
        self.setup = args_model['setup']
        self.L = int(self.setup[0])
        self._ss = spin_basis_1d(L=self.L)
   
    def _setup_H(self, **args_model):
        """  1 and two qubits hamioltonians are implemented here
        """                 
        if(self.setup[0]  == "1"):  
            ## 1 Qbit Hamiltonians     
            self.gen_energy_scaling() #Noise implemented here
            Ex = self.Ex
            Ez = self.Ez
            
            if(self.setup == '1Q0'):
                # Ref Bukov (with a - sign // verif )
                # f(t) in {-4, 4} or [-4, 4]
                # H(t)=  f(t) X + Z                
                assert (self.n_controls == 1), "number of control functions don't match"
                f = self.control_fun[0]
                args_f = []
                g = lambda t: 1 - f(t)
                args_g = []    
                dynamic = [['x', [[Ex, 0]], f, args_f]]
                static = [['z', [[Ez, 0]]]]
                self._H = hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64)
            
            elif(self.setup == '1Q1'):
                # H(t)=  f(t) X + (1-f(t)) Z
                assert (self.n_controls == 1), "number of control functions don't match"
                f = self.control_fun[0]
                g = lambda t: 1 - f(t)
                args_f = args_g = []    
                dynamic = [['x', [[Ex, 0]], f, args_f], ['z', [[Ez, 0]], g, args_g]]
                self._H = hamiltonian([], dynamic, basis=self._ss, dtype=np.float64)
            
            elif(self.setup == '1Q2'):
                # H(t)=  f(t) X + g(t) Z 
                # 1Q1 with relaxed constraints that g(t) = 1-f(t)
                assert (self.n_controls == 2), "number of control functions don't match"
                f = self.control_fun[0]
                g = self.control_fun[1]
                args_f = args_g = []                    
                dynamic = [['x', [[Ex, 0]], f, args_f], ['z', [[Ez, 0]], g, args_g]]
                self._H = hamiltonian([], dynamic, basis=self._ss, dtype=np.float64)
        else:
            # Only one setup implemeneted so far
            raise NotImplementedError()

    def gen_energy_scaling(self, randomized = True):
        """ generate energy scaling for the different setups (including randomness)
        if needed
        """
        if(self.setup[0] == '1'):
            self.Ex = 1
            self.Ez = 1
            if(randomized and hasattr(self, '_noise_func') and self._noise_func is not None):
                if( ('Ex' in self._noise_func) and self._noise_func['Ex'] is not None):                
                    self.Ex *= (1 + self._noise_func['Ex']()) 
                    
                if(('Ez' in self._noise_func) and self._noise_func['Ez'] is not None):                
                    self.Ez *= (1 + self._noise_func['Ez']())
        else:
            pass

    def get_state(self, state_obj = None):
        """ Generate quantum states from state_obj <str> or <array/list<num>>"""
        list_str_states = ['0', '1', 'm', 'p']
        if state_obj in list_str_states:
            res = self._get_custom_state(state_obj)
        else:
            res = mod.pcModel_qspin.get_state(self, state_obj)
        return res
            
    def _get_custom_state(self, string):
        """ (normalizazion constants are omitted in the description)
        |m> = |0> - |1> // |p> = |0> + |1>
        TODO: generalized with a little pit of parsing""" 
        if (string == '0'):
            psi = np.array([0,1], dtype=np.complex128)
        elif (string == '1'):
            psi = np.array([1,0], dtype=np.complex128)
        elif (string == 'p'):
            psi = 1/np.sqrt(2) * np.array([1,1], dtype=np.complex128)
        elif (string == 'm'):
            psi = 1/np.sqrt(2) * np.array([-1,1], dtype=np.complex128)
        else:
            raise SystemError("string {} not reco".format(string))
        return psi
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
            print("FOM="+str(res))
        return res

    def Evolution(self, time , state_init, method, store, **extra_args):
        """  Evolve the state_init according to the relevant method and store if
            required
        """
        if(method == 'se'):
            state_t = self.EvolutionSE(time, state_init, **extra_args)

        elif(method == 'pop_adiab'):
            state_t = self.EvolutionPopAdiab(time, state_init, **extra_args)

        else:
            raise NotImplementedError()

        if store:
            self.state_t = state_t
        
        return state_t

# --------------------------------------------------------------------------- #
#   Custom evolutions
# --------------------------------------------------------------------------- #
    def EvolutionSE(self, time = None, state_init = None, iterable = False, **args_evolve):
        """  Wrap evolve from QuSpin only expose some of teh arguments
        hamiltonian.evolve(state_init, t0 = 0,times = T, eom="SE",solver_name="dop853",stack_state=False,
        verbose=False,iterate=False,imag_time=False,**solver_args)
        """
        if time is None:
            time = self.t_simul
        if state_init is None:
            state_init = self.state_init

        state_t = self._H.evolve(state_init, t0 = 0, times = time, iterate=iterable, **args_evolve)
        return state_t


    def EvolutionPopAdiab(self, time = None, state_init = None, **args_evolve):
        """ Evolve the state according to SE and project it on the instantaneous 
        eigen vectors of the hamiltonian. This pop_adiab is stored by default and
        state_t is returned """
        if time is None:
            time = self.t_array
        if state_init is None:
            state_init = self.state_init

        nb_ev = args_evolve.pop('nb_ev', 2)
        state_t = self.EvolutionSE(time, state_init, **args_evolve)
        # Not optimal
        pop_adiab = self._h_project_to_instant_evect(time, state_t, nb_ev)
        energies = self.EvolutionInstEnergies(time, nb_ev)
        self.pop_adiab = pop_adiab
        self.energies = energies # both energies and field
        
        return state_t
        

    def EvolutionInstEnergies(self, time = None, nb = 2):
        """ Custom study of the gap """
        if(time is None):
            time = self.t_array
        energies = self._h_get_lowest_energies(time, nb)
        func_values = [self.control_fun(t) for t in time] 
        res = np.c_[(func_values, energies)]
        return res
        

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
            t = self.t_array
            en_tmp = self.energies
            n_cf = self.n_controls
            en = en_tmp[:, n_cf:]
            cf = en_tmp[:, :n_cf]
            nb_levels = pop_adiab.shape[1]    
            f, axarr = plt.subplots(2,2, sharex=True)
            
            axarr[0,0].plot(t, cf, label = 'f(t)')
            for i in range(nb_levels):
                pop_tmp = pop_adiab[:, i]
                max_tmp = np.max(pop_tmp)
                if(max_tmp > 0.1):
                    axarr[0,1].plot(t, pop_tmp, label = str(i))
                    axarr[1,0].plot(t, en[:, i], label = str(i))
                elif(max_tmp > 0.01):
                    axarr[1,1].plot(t, pop_tmp, label = str(i))
                    axarr[1,0].plot(t, en[:, i], label = str(i))
            
            axarr[0,1].legend()
            axarr[1,1].legend()
            axarr[1,0].legend()
            axarr[0,0].legend()
        
        else:
            print("pcModel_qspin.plot_pop_adiab: no pop_adiab found.. Generate it first")
        




### ======================= ###
# TESTING
### ======================= ###
if(__name__ == '__main__'): 
    Qubits.info()
    
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
    TQSL = np.pi/np.sqrt(2)
    T = TQSL*2

    linear_ramp = pf.LinearFunc(w=1/T , bias =0)
    fom_name = ['f2t2:neg_fluence:0.0001_smooth:0.0005', 'f2t2', 'fluence', 'smooth']
    fom_name_last = ['last:'+f for f in fom_name]

    dico_simul = {'control_obj':linear_ramp, 'T':T, 'dt':0.01, 'setup':'1Q1', 
                'flag_intermediate':False, 'state_init':'GS_i', 'state_tgt':'GS_f', 'fom':fom_name_last}

    simul = Qubits(**dico_simul)
    if(False):
        res_fom = simul.Simulate(fom = fom_name, store = True, debug = False)
    
    #study Evol of the population in the first energy levels
    if(False):    
        state_t = simul.EvolutionPopAdiab(nb_ev = 2)
        simul.plot_pop_adiab()
        
    #Try the exponential ram        
    if(False):        
        pass
    
    if(False):
        basis = spin_basis_1d(1)
        f = lambda t:0+1/T * t
        g = lambda t:1-f(t)
        args_f = args_g = []        
        dynamic = [['x', [[1, 0]], f, args_f]]
        static = [['z', [[1, 0]]]]
        H = hamiltonian(static, [], basis=basis, dtype=np.float64)
        H2 = hamiltonian([], dynamic, basis=basis, dtype=np.float64)
        dynamic = [['x', [[1, 0]], f, args_f], ['z', [[1, 0]], g, args_g]]
        H3 = hamiltonian([], dynamic, basis=basis, dtype=np.float64)

    