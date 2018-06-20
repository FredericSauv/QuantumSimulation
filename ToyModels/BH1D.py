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
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import pFunc_base as pf
    from . import ModelBase as mod


ilib.reload(ut)
ilib.reload(mod)


class BH1D(mod.pcModel_qspin):
    """ Simulate a 1D - MH model. Implementation of the pcModel_qspin """
    _LIST_ARGS = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS['setup'] = '<str> string pointing to one of the setup'
    _LIST_ARGS['L'] = '<int> Length of the chain'
    
    _LIST_ARGS_OPT = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS_OPT['Nb'] = ('<int>Number of particles', _LIST_ARGS['L'])
    _LIST_ARGS_OPT['mu'] = ('<float> chemical potential', 0)
    _LIST_ARGS_OPT['sps'] = ('<int> Number of particles max per site', 3)
    _LIST_ARGS_OPT['kblock'] =('<int/None> if int reduce the dynamics to the k-th momentum block (0 to L)', None)
    _LIST_ARGS_OPT['pblock'] =('<int/None> if int reduce the dynamics to the k-th parity block (-1 or 1)', None)
    # _LIST_ARGS_OPT['sym'] = ('<dict> definition of the symmetries to use c.f. quspin doc', {}) 
    _LIST_ARGS_OPT['flag_store'] = ('<bool> Allow the storage of intermediate results', False)  

    def __init__(self, **args_model):
        """ Initialize the simulations of the driven BH  """
        mod.pcModel_qspin.__init__(self, **args_model)
        self._setup_basis(**args_model)
        self._setup_H(**args_model)
        self._FLAG_STORE = args_model.get('flag_store', False)
        self._setup_fom_bh1d_bricks(**args_model)
        self.state_init = args_model['state_init']
        self.state_tgt = args_model.get('state_tgt')


    def _setup_basis(self, **args_model):
        """ Build and store:
        - the basis (self._ss) and the basis without use of symmetries()
        - config_system
        """
        config_bh1D ={}
         
        for p in ['L', 'Nb']:
            config_bh1D[p] = args_model[p]

        for p in ['sps', 'kblock', 'pblock']:
            config_bh1D[p] = args_model.get(p, self._LIST_ARGS_OPT[p][1])

        self._ss = boson_basis_1d(**config_bh1D)
        if np.any([config_bh1D[p] is not None for p in ['kblock', 'pblock']]):
            self._flag_basis_symm = True
            config_bh1D['pblock'] = None
            config_bh1D['kblock'] = None
            self._ss_nosym = boson_basis_1d(**config_bh1D)
            self._P = self._ss.get_proj(dtype=np.complex128, pcon=True)      
        else:
            self._flag_basis_symm = False
            self._ss_nosym = self._ss
            self._P = None #Id      
        
    def _setup_H(self, **args_model):
        """  Bose Hubbard Hamiltonians:
        H = - J(t) Sum_l (b_l b_{l+1}^t + h.c.) + U(t) Sum_l (n_l (n_{l}-1)) + mu Sum_l (n_l)   
        ++ setup 1: cyclical boundary conditions, U(t) = control_fun(t), J(t) = 1- U(t)

        """ 
        self.setup = args_model['setup']
        L = args_model['L']
        mu = args_model.get('mu', self._LIST_ARGS_OPT['mu'][1])
        if(self.setup  == "1"):  
            assert (self.n_controls == 1), "number of control functions don't match"
            U = self.control_fun[0]
            args_U = []
            J = lambda t: 1 - U(t)
            args_J = []    
            hop = [[-1, i, (i+1)%L] for i in range(L)] # Cyclical boundaries
            dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]
            inter_nn = [[0.5, i, i] for i in range(L)]
            inter_n = [[-0.5, i] for i in range(L)]
            dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]
            dynamic = dynamic_inter + dynamic_hop
            pot_n =  [[mu, i] for i in range(L)]
            static = [['n', pot_n]]
            self._H = hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64)
        else:
            # Only one setup implemeneted so far
            raise NotImplementedError()

    def _setup_fom_bh1d_bricks(self, **args_model):
        """ add new methods which can be used e.g. to define the FOM """                
        # New functions to be used to compute the FOM
        L = args_model['L']
        self._op_n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=self._ss_nosym, dtype=np.float64) for i in range(L)]
        if (self._flag_basis_symm):
            def avg_var_occup(V_symm):
                V = self._P.dot(V_symm)
                n_var_sites = [mod.pcModel_qspin._h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
        else:
            def avg_var_occup(V):
                n_var_sites = [mod.pcModel_qspin._h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
        self._avg_var_occup = avg_var_occup
        self._fom_func['varN'] = self._avg_var_occup    


# --------------------------------------------------------------------------- #
#   SIMULATIONS 
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

        nb_ev = args_evolve.pop('nb_ev', 5)
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
    BH1D.info()
    
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
    T = 100

    linear_ramp = pf.LinearFunc(w=1/T , bias =0)
    fom_name = ['f2t2:neg_fluence:0.0001_smooth:0.0005', 'f2t2', 'varN:sqrt', 'fluence', 'smooth', 'varN']
    fom_name_last = ['last:'+f for f in fom_name]

    dico_simul = {'control_obj':linear_ramp, 'L':6, 'Nb':6, 'mu':0, 'T':T, 'dt':0.01, 
                'flag_intermediate':False, 'setup':'1', 'state_init':'GS_i', 
                'state_tgt':'GS_f', 'fom':fom_name_last}

    simul = BH1D(**dico_simul)
    res_fom = simul.Simulate(fom = fom_name, store = True, debug = False)
    
    
    #study Evol of the population in the first energy levels
    if(False):    
        simul = BH1D(**dico_simul)
        nb_ev = 10
        state_t = simul.EvolutionPopAdiab(nb_ev = nb_ev)
        simul.plot_pop_adiab()
        
    #Try the exponential ram        
    if(True):        
        state_t = simul.EvolutionPopAdiab(nb_ev = 5)
        simul.plot_pop_adiab()
    
    if(False):
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

    
    