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


if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import ParametrizedFunctionFactory as fun
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import ParametrizedFunctionFactory as fun

from ..Utility.Quantum.StateSpace import StateSpace as ss
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
# TODO: measure_var
#==============================================================================
class BH1D(mod.Models):
    MODEL_PARAMS_DEF = ('T', 'setup', 'L')
    _LIST_FOM_WORST = {'fidelityToTarget':0}

    def __init__(self, controlFun = (lambda t: 0), model_params = {}):
        """ Initialize the simulations of the driven BH
        """
        mod.Models.__init__(self, noise = model_params.get('noise'))
        self.SetUpTime(model_params)
        self.SetUpSystem(controlFun, model_params)
        self.UpdateStateInit(model_params.get('state_init'))
        self.UpdateStateTarget(model_params.get('state_tgt'))
        self.SetUpFOM()
        


# --------------------------------------------------------------------------- #
#   SET UP CONTROL
# --------------------------------------------------------------------------- #
    def SetUpTime(self, model_params):
        """ Generate time attributes
        """
        self.T = model_params['T']
        self.dt = model_params['dt']
        self.t_array = np.concatenate((np.arange(0, self.T, self.dt), [T]))
        self.intermediate_time = model_params['t_intermediate']
        if(self.self.t_intermediate):
            self.t_simul = self.t_array
        else:
            self.t_simul = self.T

    def SetUpSystem(self, controlFun, model_params):
        """  
        """ 
        self.L = model_params['L']
        self.N = model_params.get('N', self.L) 
        self.mu = model_params.get('mu', 0)
        self.setup = model_params['setup']
        self.sps = model_params.get('sps', 3) 
        self.sym = model_params.get('sym', {})


        if(self.setup  == "1"):
            # L sites cyclical boundary conditions and 1 filling simulated using  QuSpin
            L = self.L
            self.controlFun = controlFun
            self._nb_control_func = 1
            self._ss = boson_basis_1d(L, Nb = self.N, sps = self.sps, **self.sym)
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
            self.H = hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64, **self.sym)
            
            
    def SetUpFOM(self):
        """ Set up functions accessible to compose FigureOfMerit (used in GetFOM)
        """
        self._fom_func = {'max': np.max, 'min': np.min, 'avg': np.average, 'sqrt':np.sqrt}
        self._fom_func['last'] = getLast
        self._fom_func['neg'] = linear
        self._fom_func['f2t'] =  (lambda x: fid(x, self.state_tgt))
        self._fom_func['f2t2'] =  (lambda x: fid2(x, self.state_tgt))
        self._fom_func['fluence'] =  (lambda x: self.GetFluence(self.T, self.dt))
        self._fom_func['fluenceNorm'] =  (lambda x: self.GetFluence(self.T, self.dt) / 2)
        self._fom_func['smooth'] = (lambda x: self.GetSmoothness(self.T, self.dt))
        self._fom_func['rdmplus'] = (lambda x: x + self._noise_func['fom']())
        self._fom_func['rdmtime'] = (lambda x: x * (1 + self._noise_func['fom']()))
        
        # measuremenprojection on the target_state 
        self._fom_func['proj5'] = (lambda x: n_measures(x, nb =5, measur_basis = self.state_tgt))
        self._fom_func['proj10'] = (lambda x: n_measures(x, nb =10, measur_basis = self.state_tgt))
        self._fom_func['proj100'] = (lambda x: n_measures(x, nb =100, measur_basis = self.state_tgt))
        self._fom_func['proj1000'] = (lambda x: n_measures(x, nb =1000, measur_basis = self.state_tgt))

        # measurement based on operators
        self._n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=self._ss, dtype=np.float64) for i in range(self.L)]
        def avg_var_occup(V):
            n_var_sites = [var_operator(op, V) for op in self.n_sites]
            avg_var_occup = np.average(n_var_sites)
            return avg_var_occup

        self._avg_var_occup = avg_var_occup
        self._fom_func['varN'] = self._avg_var_occup 


    def UpdateControlParameters(self, params, indexFun = 0, indexControlFun = None,  **args):
        """ Update the parameters of the control Function
        TODO: NEED TO CHANGE WAY PARAMETRIC FUNCTIONS ARE DEALT WITH
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
    def Simulate(self, time = None, state_init = None, fom = None, store = False, debug=False, **extra_args):
        """ Main entry point to simulate the system
            if fom is not None, will return it, if not the state_t of the system
        """
        if debug:
            pdb.set_trace()
        
        res = self.Evolution(time = time, state_init = state_init, method = 'SE', store = store,**extra_args)

        
        if (fom is not None):
            res = self.ComputeFOM(res, fom)
            print("FOM="+str(res))
        
        return res



    def Evolution(self, time = None, state_init = None, method = 'SE', store = True, **extra_args):
        """  Evolve the state_init according to the relevant method and store if
            required
        """
        if(method == 'SE'):
            state_t = self.EvolutionSE(time, state_init, **extra_args)
            if store:
                self.state_t = state_t
                
        elif(method == 'adiabatic'):
            state_t, energies_t = self.EvolutionAdiabatic(time, state_init,**extra_args)
            if store:
                self.state_ADIAB = state_t
                self.energies_ADIAB = energies_t
                
        elif(method == 'testing'):
            raise NotImplementedError()

        else:
            raise NotImplementedError()
        
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


    def EvolutionAdiabatic(self, time, state_init= None, iterable = False, **args_evolve):
        """ Simulate adiabatic evolution
        """
        raise NotImplementedError()


    def EvolutionTesting(self, T, state_init = None, time_step = 0.01, coeff_finer = None):
        """   
        """
        raise NotImplementedError()

# --------------------------------------------------------------------------- #
#   Compute Figure Of Merits
# --------------------------------------------------------------------------- #
    def ComputeFOM(self, st, fom):
        """Compute a potentially composed FOM (or list of FOM)
            i.e. fom = ['lst:f2t:neg:0.3, last:f2t2']
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
        """Compute a fom by composing functions
            e.g. ComputeFOMAtom(state, 'lst:f2t:neg:0.3') >> 0.3 * neg(f2t(lst))
            i.e. 0.3*(1 - f2t(st[-1])) i.e. inverse of the fidelity 
            computed on the last state.
        """                
        f2apply = [self._fom_func.get(k, lambda x: float(k)*x) for k in ut.splitString(fom, ":")]
        res = ut.compoFunctions(f2apply, st, order = 0)
        return res 

    def GetFluence(self, time = None, index = None):
        """
        """
        if (time is None):
            time = self.t_array
        
        if ut.is_iter(self.controlFun):
            fl = [getFluence(f, time) for f in self.controlFun]
        else:
            fl = getFluence(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])

        return res
    
    def GetSmoothness(self, time = None, time_step = None, index = None):
        """
            
        """
        if (time is None):
            time = self.t_array

        if ut.is_iter(self.controlFun):
            fl = [getSmoothnes(f, time) for f in self.controlFun]
        else:
            fl = getSmoothnes(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res
    
    
    

# --------------------------------------------------------------------------- #
#   Custom evolutions
# --------------------------------------------------------------------------- #
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
                _, state_res = self.H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
            elif('GS_f'):
                #GS at t = T
                _, state_res = self.H.eigsh(time = self.T, k=1, which='SA',maxiter=1E10) 
            else:
                i_res = self._ss.index(state)
                state_res = np.zeros(self._ss.Ns, dtype=np.float64)
                state_res[i_res] =1.0
        else:
            res = np.array(state)

        return res



### ======================= ###
# SOME SUPPORT FUNCTIONS for FOM: 
#   + getFluence()
#   + getSmoothness()
#   + getLast()
#   + linear()
#
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
    """ return last element
    """
    return st[-1]
                
def linear(val, a = -1, b = 1):
    """ f(x) = b + ax, by def f(x) = 1 - x
    """
    return (b + a * val)
    
    
def var_operator2(O, V, **args):
    res = -O.quant_fluctu(V, **args)
    return res


def var_operator(O, V):
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

def fid2(V1, V2):
    return np.square(np.abs(ip(V1, V2)))

def fid(V1, V2):
    return np.abs(ip(V1, V2))

def probaFromState(ket1):
    """ Gen proba distrib from a quantum state
    """
    return np.square(np.abs(ket1))

def n_measures(ket1, nb = 1, measur_basis = None, num_precis = 1e-6):        
    """Projective measurement in a basis (by default the basis in which the ket 
    is represented) 
    Returns an array with each entry corresponding to the frequency of the 
    measurement based on nb measurements
    
    !!CONVENTION!! ket_proj = N x D (with D dim of the Hilbert space and N the
    number of elements provided)
    
    #TODO: only work for one ket (as opposed to array of kets over time)
    # Ortho measurement only
    #TODO: Could probably do better (concise + dealing with different shapes of
    data)
    """
    #pdb.set_trace()
    dim_ket = len(ket1)
    
    single_value = False
    if(measur_basis is None):
        measur_basis = np.eye(dim_ket) #Computational basis
    elif (len(measur_basis.shape) == 1):
        assert(len(measur_basis) == dim_ket), "problem with the dim"
        measur_basis = [measur_basis] # just reshape it
        single_value = True
    else:
        assert(measur_basis.shape[1] == dim_ket)
        
    index_measure = np.arange(len(measur_basis))
    proj = [ip(basis, ket1) for basis in measur_basis]
    proba = probaFromState(proj)
    assert(np.sum(proba)<=(1.0 + num_precis)), "not a valid proba distrib" #should it be relaxed 
    proba_cum = np.concatenate(([0],np.cumsum(proba)))
    samples = np.random.sample(nb)
    observations = [np.sum((samples > proba_cum[i]) * (samples <= proba_cum[i+1])) for i in index_measure]
    frequencies = np.array(observations)/nb
    if(single_value):
        assert(len(frequencies) == 1), "pb"
        frequencies = frequencies[0]
    return frequencies


### ======================= ###
# TESTING
### ======================= ###

if(__name__ == '__main__'):
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..

    v = 0.1
    U, T = gen_linear_ramp(v)
    dico_simul={'L':5, 'mu':0, 'T':T, 'dt':0.1, 't_intermediate':False,
                'setup':1, 'state_init':'GS_i', 'state_target':'GS_f'}

    simul = BH1D(U, dico_simul)
    fom_name = ['f2t2', 'varN:sqrt']
    res_fom = simul.Simulate(fom = fom_name, store = False, debug=True)