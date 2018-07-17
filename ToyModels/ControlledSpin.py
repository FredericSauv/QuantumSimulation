#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
from . import Models as mod
from ..Utility import Helper as ut
from ..Utility.Quantum import StateSpace as ss
from ..Utility.Quantum.StateSpace import FiniteHilbertSpace as  fhs
from ..Utility.Quantum import Hamiltonian as ham
from ..Utility.Optim import ParametrizedFunctionFactory as fun


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
#        Toy model: Spin or chain of spins like
#        Setup1: simple qbit
#                 H(t) = - f(t) sigmaZ + (1-f(t)) sigmaX 
#        Setup2: Two capacitavely coupled Josephson junction
#                H(t) = sum_i ()   
#
#    MAIN METHODS:
#        - init : 
#        - changeOfPicture : 
#        - getExtraResults : (should be spin off)
#        - compare 
#
#   MAIN VARIABLES
#        - state_t: np array with dim (t (number of time steps) x d (Hilbert dim))
#
#==============================================================================
class ControlledSpin(mod.Models):
    def __init__(self, controlFun = (lambda t: 0), setup = 1, state_init = None, state_target = None, T = 1.0, dt = 0.01, **args):
        """  
        Purpose:
            Store params of the toymodel
            generate a StateSpace associated
        """
        mod.Models.__init__(self, noise = args.get('noise'))

        #randomness params (white noise only)
        #self._stdfom = args.get('noise_fom' ,0.05)
        #self._stdparams = args.get('noise_fom')

        self.SetUpSystem(controlFun, setup, **args)
        self.UpdateStateInit(state_init)
        self.UpdateStateTarget(state_target)
        self.T = T
        self.dt = dt
        self._LIST_FOM_WORST = {'fidelityToTarget':0}
        



        #Functions used to compose FigureOfMerit (used in GetFOM)
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
    def SetUpSystem(self, controlFun, setup, **args):
        """  
        Purpose:
            Generate the State Space and Hamiltonian of the system based 
            on a control function and a predefined setup 
            + setup1: 1qbit H(t)=  f(t) X + (1-f(t)) Z
            + setup2: 1qbit H(t)=  f(t) X + g(t) Z
            
        **args to provide potentially more flexibility in the setup of the models
        """        
        self.setup = setup
        self.controlFun = controlFun
        self._nb_control_func = 1
        #Case of a list of control functions
        if(hasattr(self.controlFun, "__iter__")):
            self._nb_control_func = len(self.controlFun)
        else:
            self._nb_control_func = 1

        if(self.setup[0] == '1'):
            ## 1 Qbit Hamiltonians     
            self.gen_energy_scaling() #Noise implemented here
            
            if(self.setup == '1Q0'):
                # Ref Bukov (with a - sign // verif )
                # f(t) in {-4, 4} or [-4, 4]
                # H(t)=  f(t) X + Z                 
                assert (self._nb_control_func == 1), 'Wrong number of control functions'
                self._ss = ss.OneQbit()
                self._HList = [[self.controlFun, self.Ex * self._ss._X], [lambda t: 1, self.Ez * self._ss._Z]]
                self.H = ham.TDHamiltonian(self._HList, self._ss)
            
            elif(self.setup == '1Q1'):
                # H(t)=  f(t) X + (1-f(t)) Z
                assert (self._nb_control_func == 1), 'Wrong number of control functions'
                self._ss = ss.OneQbit()
                self._HList = [[self.controlFun, self.Ex * self._ss._X], [lambda t: (1-self.controlFun(t)), self.Ez * self._ss._Z]]
                self.H = ham.TDHamiltonian(self._HList, self._ss)
    
            elif(self.setup == '1Q2'):
                # H(t)=  f(t) X + g(t) Z 
                # 1Q1 with relaxed constraints that g(t) = 1-f(t)
                assert (self._nb_control_func == 2), 'Wrong number of control functions'
                self._ss = ss.OneQbit()
                self._HList = [[self.controlFun[0], self.Ex * self._ss._X], [self.controlFun[1], self.Ez * self._ss._Z]]
                self.H = ham.TDHamiltonian(self._HList, self._ss)

        ## 2 Qbits Hamiltonians
        elif(self.setup == '2Q1'):
            # cf. CRAB paper
            # Josephson Junction
            assert (self._nb_control_func == 1), 'Wrong number of control functions'
            self._ss = ss.TwoQbits()
            Ec = args.get('Ec', 1)
            Ej = args.get('Ej', -1)
            H0 = Ec * (self._ss._Z[0] + self._ss._Z[1]) + Ej * (self._ss._X[0] + self._ss._X[1])
            HI = self._ss._Z[0] * self._ss._Z[1]
            
            self._HList = [[lambda t: 1, H0], [self.controlFun, HI]]
            self.H = ham.TDHamiltonian(self._HList, self._ss)
        

        ## 1 QuTrits Hamiltonians        
        elif(self.setup == '1QT1'):
            ## H(t)=  f(t) X + (1-f(t)) Z 
            ## Where Z is diag(-1, 1, 0), X is F Z F^T
            assert (self._nb_control_func == 1), 'Wrong number of control functions'
            self._ss = ss.QuDits(3)
            self._HList = [[self.controlFun, self._ss._X], [lambda t: (1-self.controlFun(t)), self._ss._Z]]
            self.H = ham.TDHamiltonian(self._HList, self._ss)

        elif(self.setup == '1QT2'):
            # STIRAP Setup (2 photon detuning = 0 and RWA approx)
            # H(t)= Delta * |2><2| + Om_P(t) * (|0><2| + |2><0|) + Om_S(t) * (|1><2| + |2><1|) 
            # requires 2 control functions
            # Could add detuning
            assert (self._nb_control_func == 2), 'Wrong number of control functions'
            self._ss = ss.QuDits(3)
            delta = args.get('1p_detuning', 0.1) # One photon detuning            
            HP = self._ss.ket(0,2) + self._ss.ket(2,0)
            HS = self._ss.ket(1,2) + self._ss.ket(2,1)
            H2 = delta * self._ss.ket(2,2) + np.identity(3) ## Detuning + shift of Energy (just used for numerical reason)
            self._HList = [[self.controlFun[0], HP], [self.controlFun[1], HS], [lambda t: 1, H2]]
            self.H = ham.TDHamiltonian(self._HList, self._ss)

        elif(self.setup == '1QT3'):
            #Lambda system with two degenerates level
            # Periodic driving (i.e. no RWA has been applied)
            # cf. Verdeny..
            # H(t) = f(t)(1 + e^{})
            assert (self._nb_control_func == 1), 'Wrong number of control functions'
            raise NotImplementedError()


        elif(self.setup == '1QS1'):
            raise NotImplementedError()
            
        ## Spin Chain
        elif(self.setup == 'NQ0'):
            self.N_spins = args['N_spins']
            raise NotImplementedError()

        
        else:
            raise NotImplementedError()
            
            
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
            #workaround they can be a small mismatch in the final index
            if(len(state_t) > len(state_trunc)):
                state_t_trunc = state_t[:len(state_trunc)]
            elif(len(state_t) < len(state_trunc)):
                state_trunc = state_trunc[:len(state_t)]
                state_t_trunc = state_t
            else:
                state_t_trunc = state_t    
            comp_resolution = fhs.fidelity_avg(state_t_trunc, state_trunc)
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
            



def getFluence(func, range_t, normalized = True):
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
    return st[-1]
                

def linear(val, a = -1, b = 1):
    return (b + a * val)
    


