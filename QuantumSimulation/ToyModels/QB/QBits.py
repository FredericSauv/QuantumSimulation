#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""

#import sys
#sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import pdb


if(__name__ == '__main__'):
    import sys
    sys.path.append("../../../")
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ...Utility.Optim import pFunc_base as pf
    from .. import ModelBase as mod



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

    
