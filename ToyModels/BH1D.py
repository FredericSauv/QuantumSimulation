#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
from . import Models as mod
import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import matplotlib.pylab as plt


if()
from ..Utility import Helper as ut
from ..Utility.Optim import ParametrizedFunctionFactory as fun
#from ..Utility.Quantum import StateSpace as ss
#from ..Utility.Quantum.StateSpace import FiniteHilbertSpace as  fhs
#from ..Utility.Quantum import Hamiltonian as ham




import numpy as np
import pdb
import importlib as ilib
ilib.reload(mod)
ilib.reload(ss)
ilib.reload(ham)
ilib.reload(fun)
#==============================================================================
#                   ToyModel
#    PURPOSE:
#        Simulate a 1D - BHM model - wrap quspin library
#        
#    MAIN METHODS:
#        - init: 
#        - changeOfPicture : 
#        - getExtraResults : (should be spin off)
#        - compare 
#
#   MAIN VARIABLES
#        - state_t: np array with dim (t (number of time steps) x d (Hilbert dim))
#
#==============================================================================
class BH1D(mod.Models):
    MODEL_PARAMS_DEF = {'T':1, 'J':
    def __init__(self, controlFun = (lambda t: 0), model_params = {}):
        """ Initialize the simulations of the driven BH
        """
        mod.Models.__init__(self, noise = args.get('noise'))

        self.SetUpSystem(controlFun, model_params)
        self.UpdateStateInit(state_init)
        self.UpdateStateTarget(state_target)
        self.T = T
        self.dt = dt
        self._LIST_FOM_WORST = {'fidelityToTarget':0}
        

        #Functions used to compose FigureOfMerit (used in GetFOM)
        def avg_var_occup(V):
            n_var_sites = [variance(op, V) for op in n_sites]
            avg_var_occup = np.average(n_var_sites)
            return avg_var_occup

        self._fom_func = {'max': np.max, 'min': np.min, 'avg': np.average}
        self._fom_func['last'] = getLast
        self._fom_func['neg'] = linear
        self._fom_func['f2t'] =  (lambda x: fhs.fidelity(x, self.state_tgt))
        self._fom_func['f2t2'] =  (lambda x: fhs.fidelity2(x, self.state_tgt))
        self._fom_func['fluence'] =  (lambda x: self.GetFluence(self.T, self.dt))
        self._fom_func['fluenceNorm'] =  (lambda x: self.GetFluence(self.T, self.dt) / 2)
        self._fom_func['smooth'] = (lambda x: self.GetSmoothness(self.T, self.dt))
        self._fom_func['rdmplus'] = (lambda x: x + self._noise_func['fom']())
        self._fom_func['rdmtime'] = (lambda x: x * (1 + self._noise_func['fom']()))
        # New: measuremenprojection on the target_state 
        self._fom_func['proj5'] = (lambda x: fhs.measurement(x, nb =5, measur_basis = self.state_tgt))
        self._fom_func['proj10'] = (lambda x: fhs.measurement(x, nb =10, measur_basis = self.state_tgt))
        self._fom_func['proj100'] = (lambda x: fhs.measurement(x, nb =100, measur_basis = self.state_tgt))
        self._fom_func['proj1000'] = (lambda x: fhs.measurement(x, nb =1000, measur_basis = self.state_tgt))
# --------------------------------------------------------------------------- #
#   SET UP CONTROL
# --------------------------------------------------------------------------- #
    def SetUpSystem(self, controlFun, model_params):
        """  
        """ 
        L = model_params['L']
        self.T = model_params['T']
        self.dt = model_params.get('dt')
        mu = model_params.get('mu', 0)
        setup = params.get('setup', 1)

        if(setup == "1"):
            # L sites cyclical boundary conditions and 1 filling simulated using  QuSpin
            self.setup = setup
            self.controlFun = controlFun
            self._nb_control_func = 1
            
            self._ss = boson_basis_1d(L,Nb=L,sps=3)
            U = self.controlFun
            J = lambda t: 1 -U(t)

            hop = [[-1, i, (i+1)%L] for i in range(L)]
            dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]
            inter_nn = [[0.5, i, i] for i in range(L)]
            inter_n = [[-0.5, i] for i in range(L)]
            dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]
            dynamic = dynamic_inter + dynamic_hop
            pot_n =  [[mu, i] for i in range(L)]
            static = [['n', pot_n]]
            self.H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

            self._n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis,dtype=np.float64) for i in range(L)]



E_SF, V_SF = H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10) # only GS
E_MI, V_MI = H.eigsh(time = T, k=1, which='SA',maxiter=1E10) # only GS
print(E_SF)
print(E_MI)

avg_var_occup(V_SF)
avg_var_occup(V_MI)


        self._ss = None
        self.state_init = None
        self.state_tgt = None
        self.state_t = None
        self.pop_t = None
        self.setup = None
        self.time_array = None

            
            
    def UpdateControlParameters(self, params, indexFun = 0, indexControlFun = None,  **args):
        """  
        Purpose:
            Update the parameters of the control Function
        TRICKY
        NEED TO CHANGE WAY PARAMETRIC FUNCTIONS ARE DEALT WITH
        """
        if(indexControlFun is None):
            if(self._nb_control_func > 1):
                # if several control function split the parmas evenly
                i_params = 0
                #pdb.set_trace()
                for i_fun in range(self._nb_control_func):
                    func = self.controlFun[i_fun]
                    nb_eff = func._nbTotalParams
                    params_to_update = params[i_params:i_params+nb_eff]
                    i_params += nb_eff
                    func.UpdateParams(params_to_update, indexFun = indexFun)
            else:
                self.controlFun.UpdateParams(params, indexFun = indexFun)
        else:
            self.controlFun[indexControlFun].UpdateParams(params, indexFun=indexFun)

# --------------------------------------------------------------------------- #
#   SIMULATIONS 
# --------------------------------------------------------------------------- #
    def Simulate(self, T = None, state_init = None, time_step = None, method = 'PWCH', fom = None, store = False, debug=False):
        """
        Purpose:
            Main entry point to simulate the system
            if fom is not None, will return it, if not the state_t of the system
        """
        if debug:
            pdb.set_trace()
        
        res = self.Evolution(T, state_init, time_step, method, store)
        
        if (fom is not None):
            res = self.ComputeFOM(res, fom)
            print("FOM="+str(res))
        
        return res



    def Evolution(self, T = None, state_init = None, time_step = None, method = 'PWCH', store = True):
        """  
        Purpose:
            Evolve the state_init according to the relevant method and store if
            required
        """
        if(method == 'PWCH'):
            state_t = self.EvolutionPWCH(T, state_init, time_step)
            if store:
                self.state_PWCH = state_t
                self.state_t = state_t
                
        elif(method == 'adiabatic'):
            state_t, energies_t = self.EvolutionAdiabatic(T, state_init, time_step)
            if store:
                self.state_ADIAB = state_t
                self.energies_ADIAB = energies_t
                
        elif(method == 'testing'):
            state_t, state_adiab_t, energies_adiab_t, state_trunc, comp_adiab, comp_resolution = self.EvolutionTesting(T, state_init, time_step, 10)
            if store:
                self.state_PWCH = state_t
                self.state_t = state_t
                self.state_ADIAB = state_adiab_t
                self.energies_ADIAB = energies_adiab_t        
                self.state_RESOL = state_trunc
                self.comp_ADIAB = comp_adiab
                self.comp_RESOL = comp_resolution
        else:
            raise NotImplementedError()
        
        return state_t

# --------------------------------------------------------------------------- #
#   Compute Figure Of Merits
# --------------------------------------------------------------------------- #
    def ComputeFOM(self, st, fom):
        """
        Purpose:
            Compute a potentially composed FOM (or list of FOM)
            i.e. fom = 'F2target_fluence0.2'
        """        
        if(isinstance(fom, list)):
            res=list([])
            for f in fom:
                components = ut.splitString(f)
                tmp = np.sum([self.ComputeFOMAtom(st, c) for c in components])
                res.append(tmp)
        else:
            components = ut.splitString(fom)
            res = np.sum([self.ComputeFOMAtom(st, c) for c in components])
            

        return res       



    def ComputeFOMAtom(self, st, fom):        
        """
        Purpose:
            
            Compute a fom by composing functions
            e.g. ComputeFOMAtom(state, 'lst:f2t:neg:0.3') >> 0.3 * neg(f2t(lst))
            i.e. 0.3*(1 - f2t(st[-1])) i.e. inverse of the fidelity 
            computed on the last state.
        """                
        f2apply = [self._fom_func.get(k, lambda x: float(k)*x) for k in ut.splitString(fom, ":")]
        res = ut.compoFunctions(f2apply, st, order = 0)
        return res 

    def GetFluence(self, T = None, time_step = None, index = None):
        """
            
        """
        if (T is None):
            T = self.T
        if (time_step is None):
            time_step = self.dt
            
        range_t = np.arange(0, T, time_step)
        fl = [getFluence(h[0], range_t) for h in self._HList]
        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res
    
    def GetSmoothness(self, T = None, time_step = None, index = None):
        """
            
        """
        if (T is None):
            T = self.T
        if (time_step is None):
            time_step = self.dt
            
        range_t = np.arange(0, T, time_step)
        if (index is None):
            fl = [getSmoothnes(h[0], range_t) for h in self._HList]
            res = np.sum(fl)
        else:
            res = getSmoothnes(self._HList[index][0], range_t)
        return res
    
    
    
# --------------------------------------------------------------------------- #
#   Custom evolutions
# --------------------------------------------------------------------------- #
    def EvolutionPWCH(self, T = None, state_init = None, time_step = 0.01):
        """  
        Purpose:
            Piece Wise Constant Hamiltonian evolution
        """
        if(state_init is None):
            state_init = self.state_init            

        if(T is None):
            T = self.T
        else:
            self.T = T

        if(time_step is None):
            time_step = self.dt
        else:
            self.dt = time_step            

        state_t = self.H.Evolution(state_init, T, time_step, 'PWCH', intermediate = True)
        self.time_array = self.H.time_array
        return state_t


    def EvolutionAdiabatic(self, T, state_init= None, time_step = 0.01):
        """  
        Purpose:
            Adiabatic evolution of the ground state
        """
        if(state_init is None):
            state_init = self.state_init

        if(T is None):
            T = self.T
        else:
            self.T = T
        
        if(time_step is None):
            time_step = self.dt
        else:
            self.dt = time_step            

        state_t, energies_t = self.H.Evolution(state_init, T, time_step, 'Adiabatic', intermediate = True)
        self.time_array = self.H.time_array
        return state_t, energies_t



    def EvolutionTesting(self, T, state_init = None, time_step = 0.01, coeff_finer = None):
        """  
        Purpose:
            Evolve state according to both adiabatic/PWCH evolution
            Res:
                + How adiabatic is the evolution under ('PWCH') i.e Fidelity(adiab)
                + decreasing the time step by 10 changes something? i.e. Fidelity 
        """
        if(state_init is None):
            state_init = self.state_init

            
        state_t = self.EvolutionPWCH(T, state_init, time_step)
        comp_adiab = None
        comp_resolution = None
        
        #Test1: Adiabaticty
        state_adiab_t, energies_adiab_t = self.EvolutionAdiabatic(T, state_init,  time_step)
        comp_adiab = fhs.fidelity(state_t, state_adiab_t)
        print('Vs Adiabatic evol --> Fidelity = ' + str(np.average(comp_adiab)))

        #Test2 impact on the resolution
        if (coeff_finer is not None):
            state_finer = self.EvolutionPWCH(T, state_init,time_step / int(coeff_finer))
            state_trunc = state_finer[slice(0, len(state_finer), coeff_finer)]
            comp_resolution = fhs.fidelity_avg(state_t, state_trunc)
            print('Multiplying by '+str(coeff_finer)+ ' the time step --> Fidelity = ' + str(comp_resolution))
                        
        return state_t, state_adiab_t, energies_adiab_t, state_trunc, comp_adiab, comp_resolution


    def gen_energy_scaling(self, randomized = True):
        """ generate energy scaling for the different setups (including randomness)
        if needed
        """
        if(self.setup[0] == '1'):
            self.Ex = 1
            self.Ez = 1
            if(randomized and hasattr(self, '_noise_func')):
                if(self._noise_func['Ex'] is not None):                
                    self.Ex *= (1 + self._noise_func['Ex']()) 
                if(self._noise_func['Ez'] is not None):                
                    self.Ez *= (1 + self._noise_func['Ez']())
        else:
            pass
    
    def GetState(self, state = None):
        """ Generate quantum states either from string or array/list of num values 
        """
        if(state is None):
            res = None

        elif(isinstance(state, str)):
            if('GS_i'):
                # GS at t=0
                _, state_res = H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
            elif('GS_f'):
                #GS at t = T
                _, state_res = H.eigsh(time = self.T, k=1, which='SA',maxiter=1E10) 
            else:
                i_res = self._ss.index(state)
                state_res = np.zeros(self._ss.Ns, dtype=np.float64)
                state_res[i_res] =1.0
        else:
            res = np.array(state)

        return res



### ======================= ###
# SOME SUPPORT FUNCTIONS
### ======================= ###

def getFluence(func, range_t, normalized = True):
    """ Compute fluence of the control field
    """
    time_step = np.array(range_t[1:]) - np.array(range_t[:-1])
    val_square = np.array([func(t)**2 for t in range_t[:-1]])
    res = np.sum(np.array(val_square * time_step))
    if(normalized):
        res = res/(range_t[-1] - range_t[0])
    return res

def getSmoothnes(func, range_t, normalized = True):
    """ (ad-hoc) Measure of smoothness for a function (func) evaluated at times
    (range_t)
    res = sum_t (func(t) - func(t-1))**2
    """
    time_step = np.diff(range_t)
    diff_val_square = np.square(np.diff(func(range_t)))
    res = np.sum(np.array(diff_val_square / time_step))
    if(normalized):
        res = res/(range_t[-1] - range_t[0])

    return res
        
def getLast(st):
    #if (len(st.shape) == 2):
    raise NotImplementedError()
                

def linear(val, a = -1, b = 1):
    return (b + a * val)
    
def var_operator(O, **args):
    res = -O.quant_fluctu(**args)
    return res

def variance(O, V):
    """ 
    """
    OV = O.dot(V)
    VOOV = np.asscalar(O.matrix_ele(V, OV))
    VOV2 = O.expt_value(V) ** 2
    var = VOOV -VOV2
    return var

def gen_linear_ramp(v, T = 1, ymin=0, ymax=1):
    ramp = (lambda t: ymin + v * t)
    T = (ymax - ymin)/v
    return ramp, T

def ip(V1, V2):
    return np.dot(np.conj(np.squeeze(V1)), np.squeeze(V2))

def fid(V1, V2):
    return np.square(np.abs(ip(V1, V2)))



### ======================= ###
# TESTING
### ======================= ###

if(__name__ == '__main__'):
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..

    dico_simul={}



