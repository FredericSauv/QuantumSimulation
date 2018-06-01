#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""

import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import matplotlib.pylab as plt
import pdb
import importlib as ilib


if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import ParametrizedFunctionFactory as fun
    from QuantumSimulation.ToyModels import ModelBase as mod
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import ParametrizedFunctionFactory as fun
    from . import ModelBase as mod


ilib.reload(ut)
ilib.reload(mod)
ilib.reload(fun)


class BH1D(mod.QuspinModels):
    """ Simulate a 1D - MH model
    """
    LIST_ARGS = mod.QuspinModels.LIST_ARGS
    LIST_ARGS['setup'] = '<str> string pointing to one of the setup'
    LIST_ARGS['L'] = 'Length of the chain'
    
    
    LIST_ARGS_OPT = mod.QuspinModels.LIST_ARGS
    LIST_ARGS_OPT['N'] = '<int>Number of particles (def = L)'
    LIST_ARGS_OPT['mu'] = '<float> Chemical potential (def = 0)'
    LIST_ARGS_OPT['sps'] = '<int> Number of particles (def = L)'
    LIST_ARGS_OPT['sym'] = '<dict> definition of the symmetries'  
    LIST_ARGS_OPT['flagStore'] = '<bool> Allow the storage of intermediate results'  

    def __init__(self, controlFun = (lambda t: 0), **args_model):
        """ Initialize the simulations of the driven BH
        """
        mod.QuspinModels.__init__(self, controlFun, **args_model)
        self.SetupExtra(**args_model)

    def SetupBasis(self, **args_model):
        """ Build and store the basis (state space)
        """
        self.L = args_model['L']
        L = self.L
        self.N = args_model.get('N', self.L) 
        self.mu = args_model.get('mu', 0)
        self.sps = args_model.get('sps', 3) 
        self.sym = args_model.get('sym', {})
        self._flag_store = args_model.get('flagStore', False)

        if(self.sym in [{}, None]):
            self._flag_symm = False
            self._ss = boson_basis_1d(L, Nb = self.N, sps = self.sps)
            self._ss_nosym = self._ss
            self._P = None #Id
            
        else:
            self._flag_symm = True
            self._ss = boson_basis_1d(self.L, Nb = self.N, sps = self.sps, **self.sym)
            self._ss_nosym = boson_basis_1d(self.L, Nb = self.N, sps = self.sps)
            self._P = self._ss.get_proj(dtype=np.float64,pcon=True)
        
    def SetupH(self, **args_model):
        """  
        """ 
        self.setup = args_model['setup']
        if(self.setup  == "1"):
            # BHM with L sites, cyclical boundary conditions 
            # U(t) = f(t) and J(t) = 1 - U(t)    
            assert (self._nb_control_func == 1), "number of control functions don't match"
            L = self.L
            U = self.controlFun
            args_U = []
            J = lambda t: 1 - U(t)
            args_J = []    
            hop = [[-1, i, (i+1)%L] for i in range(L)] # Cyclical boundaries
            dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]
            inter_nn = [[0.5, i, i] for i in range(L)]
            inter_n = [[-0.5, i] for i in range(L)]
            dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]
            dynamic = dynamic_inter + dynamic_hop
            pot_n =  [[self.mu, i] for i in range(L)]
            static = [['n', pot_n]]
            self.H = hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64)
            
        else:
            # Only one setup implemeneted so far
            raise NotImplementedError()

    def SetupExtra(self, **args_model):
        """ add new methods which can be used e.g. to define the FOM
        """                
        # New functions to be used to compute the FOM
        self._op_n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=self._ss_nosym, dtype=np.float64) for i in range(self.L)]
        if (self._flag_symm):
            def avg_var_occup(V_symm):
                V = self._P.dot(V_symm)
                n_var_sites = [mod.QuspinModels.h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
        else:
            def avg_var_occup(V):
                n_var_sites = [mod.QuspinModels.h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
        self._avg_var_occup = avg_var_occup
        self._fom_func['varN'] = self._avg_var_occup    


# --------------------------------------------------------------------------- #
#   SIMULATIONS 
# --------------------------------------------------------------------------- #
    def Simulate(self, time = None, state_init = None, fom = None, store = None, debug=False, **extra_args):
        """ Main entry point to simulate the system. If fom is not None, it will 
        return it, if not return the state_t of the system.
        """
        if debug:
            pdb.set_trace()
        if store is None:
            store = self._flag_store
        
        res = self.Evolution(time = time, state_init = state_init, method = 'se', store = store, **extra_args)

        if (fom is not None):
            res = self.ComputeFOM(res, fom)
            print("FOM="+str(res))
        
        return res

    def Evolution(self, time = None, state_init = None, method = 'se', store = True, **extra_args):
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
        if(state_init is None):
            state_init = self.state_init            

        if(time is None):
            time = self.t_simul
        else:
            self.t_simul = time

        state_t = self.H.evolve(state_init, t0 = 0, times = time, iterate=iterable,**args_evolve)
        return state_t


    def EvolutionPopAdiab(self, time = None, state_init = None, **args_evolve):
        """ Evolve the state according to SE and project it on the instantaneous 
        eigen vectors of the hamiltonian. This pop_adiab is stored by default and
        state_t is returned
        """
        nb_ev = args_evolve.pop('nb_ev', 5)
        state_t = self.EvolutionSE(time, state_init, **args_evolve)
        # Not optimal
        pop_adiab = self.project_to_instant_evect(time, self.H, state_t, nb_ev)
        energies = self.EvolutionInstEnergies(time, nb_ev)
        
        self.pop_adiab = pop_adiab
        self.energies = energies
        
        return state_t
        

    def EvolutionInstEnergies(self, time = None, nb = 2):
        """ Custom study of the gap 
        """
        #pdb.set_trace()
        if(time is None):
            time = self.t_array
        energies = self.get_lowest_energies(time, nb)
        func_values = [self.controlFun(t) for t in time] 
        res = np.c_[(func_values, energies)]
        return res
        

### ======================= ###
# TESTING
### ======================= ###

if(__name__ == '__main__'):
    def gen_linear_ramp(v, ymin=0, ymax=1):
        ramp = (lambda t: ymin + v * t)
        T = (ymax - ymin)/v
        return ramp, T

    print(BH1D.LIST_ARGS)
    print(BH1D.LIST_ARGS_OPT)
    
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
    v = 0.01
    U, T = gen_linear_ramp(v)
    dico_simul={'controlFun':U, 'L':6, 'N':6, 'mu':0, 'T':T, 'dt':0.01, 
                'flagInterTime':False, 'setup':'1', 'state_init':'GS_i', 
                'state_tgt':'GS_f'}

    simul = BH1D(**dico_simul)
    

    fom_name = ['f2t2:neg_fluenceNorm:0.0001_smooth:0.0005', 'f2t2', 'varN:sqrt', 'fluenceNorm', 'smooth']
    fom_name_last = ['last:'+f for f in fom_name]
    res_fom = simul.Simulate(fom = fom_name, store = True, debug = False)
    
    
    #study Evol of the population in the first energy levels
    if(False):    
        L = 6
        N = L
        v = 0.1
        U, T = gen_linear_ramp(v)
        sym = {'kblock':0, 'pblock':1}
        dico_simul={'controlFun': U, 'L':L, 'N':N, 'mu':0, 'T':T, 'dt':0.01, 
                    'flagInterTime':True, 'setup':'1', 'state_init':'GS_i', 
                    'state_tgt':'GS_f', 'sym':sym}
    
        simul = BH1D(**dico_simul)
        nb_ev = 10
        state_t = simul.EvolutionPopAdiab(nb_ev = nb_ev)
        simul.plot_pop_adiab()
        
    #Try the exponential ram        
    if(True):    
        L = 6
        N = L
        v = 0.1
        T = 10
        sym = {'kblock':0, 'pblock':1}
        controlFun = {'T':T, 'control':{'func':'expramp', 'ampl':1, 'T':T, 'l': -5}}
 
        dico_simul={'controlFun': controlFun, 'L':L, 'N':N, 'mu':0, 'T':T, 
                    'dt':0.01, 'flagInterTime':True, 'setup':'1', 
                    'state_init':'GS_i', 'state_tgt':'GS_f', 'sym':sym}
    
        simul = BH1D(**dico_simul)
        res_square = simul.Simulate(fom = fom_name_last, store = True, debug = False)
        nb_ev = 10
        state_t = simul.EvolutionPopAdiab(nb_ev = nb_ev)
        simul.plot_pop_adiab()
    
    if(False):
        #study Energy gap
        L = 8
        N = L
        v = 0.01
        U, T = gen_linear_ramp(v)
        sym = {'kblock':0, 'pblock':1}
        dico_simul={'L':L, 'N':N, 'mu':0, 'T':T, 'dt':0.1, 'flagInterTime':False,
                    'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_f', 'sym':sym}
    
        simul = BH1D(U, dico_simul)
        energy_evol = simul.EvolutionInstEnergies(nb = 5)
        
        
        gap_1 = energy_evol[:, 2] - energy_evol[:, 1]
        ind_min = np.argmin(gap_1)
        gap_min = gap_1[ind_min]
        U_min = energy_evol[ind_min, 0]
        ratio_min = U_min / (1-U_min)        
        plt.plot(energy_evol[:, 0], gap_1)
        print(ratio_min)
        print(gap_min)
        
        #plt.plot(energy_evol[:, 0], energy_evol[:, 3] - energy_evol[:, 2])
        #plt.plot(energy_evol[:, 0], energy_evol[:, 4] - energy_evol[:, 3])
        #plt.plot(energy_evol[:, 0], energy_evol[:, 5] - energy_evol[:, 4])

    
    