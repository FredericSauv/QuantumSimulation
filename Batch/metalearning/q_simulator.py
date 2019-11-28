#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: frederic
"""
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
from scipy.interpolate import interp1d
import numpy as np 
import pdb


DEFAULT_MODEL = {'L':5,'Nb':5,'sps':None,'kblock':None,'pblock':None}
DEFAULT_CONTROL = {'T':5,'nb_params':5, 'alpha':0}
#'dt':0.01,'flag_intermediate':False,'setup':'1','state_init':'GS_i','state_tgt':'GS_inf'

class Simulator():
    """ Simulate a 1D - Bose Hubbard model. """

    def __init__(self, args_control=DEFAULT_CONTROL, args_qmodel=DEFAULT_MODEL, debug=False):
        """ Initialize the simulations of the driven BH  """
        if debug: pdb.set_trace()
        self._setup_control(args_control) 
        self._setup_qmodel(args_qmodel)
        self._f_calls = 0

    def _setup_control(self, args_control):
        self.T = args_control['T']
        self.nb_params = args_control['nb_params']
        self.control_fun = Control(**args_control)

    def _setup_qmodel(self, args_qmodel):
        """ Setup the quantum model:   
        -- define the Hamiltonian H = - J(t) Sum_l (b_l b_{l+1}^t + h.c.) 
                                    + U(t) Sum_l (n_l (n_{l}-1)) + mu Sum_l (n_l)   
        with cyclical boundary conditions, U(t) = control_fun(t), J(t) = 1- U(t)
        """
        # Define the hamiltonian
        self._verbose = args_qmodel.get('verbose', False)
        config_bh1D = {k:args_qmodel.get(k, v) for k, v in DEFAULT_MODEL.items()}
        self._basis = boson_basis_1d(**config_bh1D)
        self.L = config_bh1D['L']
        self.Nb = config_bh1D['Nb']
        self.mu = args_qmodel.get('mu',0)
        U = self.control_fun
        args_U = []
        J = lambda t: 1 - U(t)
        args_J = []    
        hop = [[-1, i, (i+1)%self.L] for i in range(self.L)] # Cyclical boundaries
        dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]
        inter_nn = [[0.5, i, i] for i in range(self.L)]
        inter_n = [[-0.5, i] for i in range(self.L)]
        dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]
        dynamic = dynamic_inter + dynamic_hop
        pot_n =  [[self.mu, i] for i in range(self.L)]
        static = [['n', pot_n]]
        self._H = hamiltonian(static, dynamic, basis=self._basis, dtype=np.float64)

        # Define initial and target state
        _, init = self._H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
        _, tgt = self._H.eigsh(time = np.inf, k=1, which='SA',maxiter=1E10)
        self.state_init = np.squeeze(init)
        self.state_tgt = np.squeeze(tgt)


#        self._op_n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=self._basis, dtype=np.float64) for i in range(self.L)]
#        Ns = self._basis.Ns
#        self._basis_fock = np.array([self._get_basis_to_fock(s) for s in range(Ns)])

    def __call__(self, params=None):
        """ Update the parameters of the control and simulate the system"""
        self._f_calls += 1
        if params is None: params = self.control_fun.parameters
        if np.ndim(params)>1:
            res = np.array([self.__call__(p) for p in params])
        else:
            self.control_fun.parameters = params
            res = self.Simulate(fom='f2t2')
            if self._verbose: print(params, res)
        return res


    def Simulate(self, store = False, fom = None,  **extra_args):
        """ Entry point to simulate the system. 
        If fom is not None, it will  it, if not return the state_t of the system.
        """
        state_t = self._H.evolve(self.state_init, t0 = 0, times = self.T, iterate=False, **extra_args)
        if store: self.state_t = state_t
        if fom == 'f2t2':
            res = self.fid2tgt(state_t)
        else:
            res = state_t
        return res

    def fid2tgt(self, V):
        """ compute the fidelity(square conv) between V1 and the target state"""
        res = np.abs(np.dot(np.conj(np.squeeze(self.state_tgt)), np.squeeze(V)))
        return res
    
    def _state_to_occup_nb(self, st):
        exp_occup = np.array([op.expt_value(st) for op in self._op_n_sites])
        return exp_occup

    def _get_basis_to_fock(self, i):
        """ get the i-th basis vector represented in the focks basis """
        Ns = self._basis.Ns
        repr_vector = np.zeros(Ns)
        repr_vector[i] = 1
        repr_fock = self._state_to_occup_nb(repr_vector)
        
        return repr_fock

class Control():
    """ Generate the control function f(x) = w(h(x), alpha)
    w(y, alpha): warper depending on alpha. act as identity when alpha is set to 0
    h(x): cubic spline fitted on the parameters and boundary conditions
    """
    def __init__(self, T, nb_params, alpha=0,**xargs):
        self.T = T
        self.dt = T / (nb_params+1)
        self._list_t = np.arange(0,nb_params+2) * self.dt 
        self._list_values = np.zeros(nb_params + 2)
        self._list_values[-1] = 1
        self.parameters = np.arange(1, nb_params+1) / (nb_params+1)
        self.alpha = alpha
        self.xargs = xargs

    @property
    def parameters(self):
        """ return the parameters, i.e. all the values contained in list_values except first 
        and last which are effectively some boundaryy conditions"""
        return self._list_values[1:-1]

    @parameters.setter
    def parameters(self, value):
        """ When setting new parameters, ensure that a new spline is fitted"""
        values_update = np.clip(value, 0, 1)
        self._list_values[1:-1] = values_update
        self._interpolator = interp1d(x=self._list_t, y=self._list_values, kind='cubic', fill_value = (0,1), bounds_error = False)

    @property
    def alpha(self):
        """ get the underlyying alpha"""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        if value == 0:
            self._K = 1
        else:
            self._K = 1/(1-np.exp(-self.alpha))


    def __call__(self, X):
        """ return f(X)"""
        res = self._interpolator(X)
        if self.alpha != 0:
            res = self._K * (1 - np.exp(-self.alpha * res))
        return np.clip(res, 0,1)