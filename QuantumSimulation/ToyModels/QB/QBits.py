#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""

#import sys
#sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
import logging, pdb
logger = logging.getLogger(__name__)
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions


if(__name__ == '__main__'):
    import sys
    sys.path.append("../../../")
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ...Utility.Optim import pFunc_base as pf
    from .. import ModelBase as mod



class Qubits(mod.pcModel_qspin):
    """ Simulate a qubit system. Implementation of the pcModel_qspin

    Notes
    -----
    * Ensemble behavior has been implemented for noise on Ex / Ez
    * Not implemented for state_init so far

    """
    _LIST_ARGS = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS['setup'] = '<str> string pointing to one of the setup'
    _LIST_ARGS_OPT = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS_OPT['flag_store'] = ('<bool> Allow the storage of intermediate results', False)  

    def __init__(self, **args_model):
        """ Initialize the system """
        mod.pcModel_qspin.__init__(self, **args_model)
        self._FLAG_STORE = args_model.get('flag_store', False)
        self.state_init = args_model['state_init']
        self.state_tgt = args_model.get('state_tgt')

    def _setup_basis(self, **args_model):
        """ Build and store the basis (self._ss) """
        self.setup = args_model['setup']
        self.L = int(self.setup[0])
        self._ss = spin_basis_1d(L=self.L)
   
    def _update_model_after_change_of_control_fun(self):
        """ Regenerate H each time the control_fun is changed"""
        if(hasattr(self, '_H') and (self._H is not None)):
            self._setup_H()
    
    def _setup_H(self, **args_model):
        """  Construct the Hamiltonian (or ensemble of Hamiltonians)
        Notes
        -----
        * 3 1-qubit hamiltonians have been implemented so far

        """
        map_hamiltonian = {'1Q0': self._gen_H_1Q0, 
                           '1Q1': self._gen_H_1Q1,
                           '1Q2': self._gen_H_1Q2}
        h_constructor = map_hamiltonian[self.setup]
        energies = self.get_system_energies() #Noise implemented here
        if(len(energies) > 1):
            # test Herniticity and Symmetry only for the first element of the
            # ensemble
            test_H = [True] + [False] * (len(energies)-1) 
            self._H_ensemble = [h_constructor(en, check_symm=t, check_herm=t) for en, t in zip(energies, test_H)]
            self._H = self._H_ensemble[0] # arbitrarly taken reference Hamiltonian
            self._energies = energies
            if(not(self._ensemble_simulation)):
                self._ensemble_simulation = True
                logger.warning("Ensemble simulations set to True")
        else:
            self._H_ensemble = None
            self._H = h_constructor(energies[0])            
            self._energies = energies[0]


    def get_system_energies(self, randomized = True):
        """ generate energies for the different setups 

        Output
        -----
        * a list of dict with energies entries
          length of the list corresponds to how many noise set-ups are requested
          if > 1 ensemble behavior (i.e. the model coreesponds to an ensemble of
          randomized noise draws)  

        """
        #1 qubit
        zero = lambda : 0 
        if(self.setup[0] == '1'):
            Ex, Ez = 1, 1
            n_func = self._noise_func
            if(randomized and len(n_func)>0):
                Ex_noise = n_func.get('Ex', zero)
                Ez_noise = n_func.get('Ez', zero)
                energies = [{'Ex': Ex * (1+Ex_noise()), 'Ez': Ez * (1+Ez_noise())} for i in np.arange(self._nb_H_ensemble)]
            else:
                energies = [{'Ex':Ex, 'Ez':Ez}]
        else:
            raise SystemError("More than one qubit models haven't beeen implemented yet")
        return energies

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

    #-------------------------------------------------------------------#
    # Implementations of the hamiltonians
    #-------------------------------------------------------------------#
    def _gen_H_1Q0(self, energies, **args):
        """ H(t)=  f(t) X + Z
        with f(t) in {-4, 4} or [-4, 4]
        Ref Bukov (with a - sign // verif )
        """
        Ex = energies['Ex']
        Ez = energies['Ez']                
        if (self.n_controls != 1):
            raise SystemError("_gen_H_1Q0: {0} control instead of 1".format(self.n_controls))
        f = self.control_fun[0]
        args_f = [] 
        dynamic = [['x', [[Ex, 0]], f, args_f]]
        static = [['z', [[Ez, 0]]]]
        return hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64, **args)

    def _gen_H_1Q1(self, energies, **args):
        """ H(t)=  f(t) X + (1-f(t)) Z  """                
        Ex = energies['Ex']
        Ez = energies['Ez']                
        if (self.n_controls != 1):
            raise SystemError("_gen_H_1Q0: {0} control instead of 1".format(self.n_controls))
        f = self.control_fun[0]
        g = lambda t: 1 - f(t)
        args_f = args_g = []    
        dynamic = [['x', [[Ex, 0]], f, args_f], ['z', [[Ez, 0]], g, args_g]]
        return hamiltonian([], dynamic, basis=self._ss, dtype=np.float64, **args)

    def _gen_H_1Q2(self, energies, **args):
        """ H(t)=  f(t) X + g(t) Z  """                
        Ex = energies['Ex']
        Ez = energies['Ez']                
        if (self.n_controls != 2):
            raise SystemError("_gen_H_1Q0: {0} control instead of 2".format(self.n_controls))
        f = self.control_fun[0]
        g = self.control_fun[1]
        args_f = args_g = []                    
        dynamic = [['x', [[Ex, 0]], f, args_f], ['z', [[Ez, 0]], g, args_g]]
        return hamiltonian([], dynamic, basis=self._ss, dtype=np.float64, **args)
        


### ======================= ###
# TESTING
### ======================= ###
if(__name__ == '__main__'): 
    # Qubits.info()
    
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
    TQSL = np.pi/np.sqrt(2)
    T = TQSL*0.5

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

    
