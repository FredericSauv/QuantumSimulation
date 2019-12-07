#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
# sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
"""
import logging
logger = logging.getLogger(__name__)
import sys, pdb
import numpy as np 
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import imp


if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.ToyModels import ModelBase as mod
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
else:
    from ...Utility.Optim import pFunc_base as pf
    from .. import ModelBase as mod

#imp.reload(mod)

class BH1D(mod.pcModel_qspin):
    """ Simulate a 1D - MH model. Implementation of the pcModel_qspin """
    _LIST_ARGS = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS['setup'] = '<str> string pointing to one of the setup'
    _LIST_ARGS['L'] = '<int> Length of the chain'
    
    _LIST_ARGS_OPT = mod.pcModel_qspin._LIST_ARGS
    _LIST_ARGS_OPT['Nb'] = ('<int>Number of particles', _LIST_ARGS['L'])
    _LIST_ARGS_OPT['mu'] = ('<float> chemical potential', 0)
    _LIST_ARGS_OPT['sps'] = ('<int> Number of particles max per site', None)
    _LIST_ARGS_OPT['kblock'] =('<int/None> if int reduce the dynamics to the k-th momentum block (0 to L)', None)
    _LIST_ARGS_OPT['pblock'] =('<int/None> if int reduce the dynamics to the k-th parity block (-1 or 1)', None)
    # _LIST_ARGS_OPT['sym'] = ('<dict> definition of the symmetries to use c.f. quspin doc', {}) 
    _LIST_ARGS_OPT['flag_store'] = ('<bool> Allow the storage of intermediate results', False)  

    def __init__(self, **args_model):
        """ Initialize the simulations of the driven BH  """
        mod.pcModel_qspin.__init__(self, **args_model)
        #self._setup_basis(**args_model)
        #self._setup_H(**args_model)
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
            self._op_n_sites = None ## there is way to define it using projection
        else:
            L = args_model['L']
            self._flag_basis_symm = False
            self._ss_nosym = self._ss
            self._P = None #Id      
            self._op_n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=self._ss_nosym, dtype=np.float64) for i in range(L)]
            i_iplusN = [[1, i, (i + L-1)%L] for i in range(L)] # Cyclical boundaries
            self._op_correl_N = [hamiltonian([['+-',i_iplusN]], [], basis=self._ss_nosym, dtype=np.float64, check_herm=False) for i in range(L)]
            Ns = self._ss.Ns
            #NEW: get the basis.. A little bit convoluted may be an easier way to do that
            # Dim_H x Nbsites
            self._basis_fock = np.array([self._get_basis_to_fock(s) for s in range(Ns)])
            # store the index of MI (i.e. 11...11)
            self._index_basis_allone = np.where([np.allclose(b, 1.) for b in self._basis_fock])[0][0]
            self._index_basis_oneone = [np.where([b[i] == 1. for b in self._basis_fock])[0] for i in range(L)]
        
    def _setup_H(self, **args_model):
        """  Bose Hubbard Hamiltonians:
        H = - J(t) Sum_l (b_l b_{l+1}^t + h.c.) + U(t) Sum_l (n_l (n_{l}-1)) + mu Sum_l (n_l)   
        ++ setup 1: cyclical boundary conditions, U(t) = control_fun(t), J(t) = 1- U(t)

        """ 
        ## Try to find them in the extra args or look if they have already been set up
        if (args_model.get('setup') is not None):
            self.setup = args_model.get('setup')
        if (args_model.get('L') is not None):
            self.L = args_model.get('L')
        if (args_model.get('Nb') is not None):
            self.Nb = args_model.get('Nb')
        if (args_model.get('mu') is not None):
            self.mu = args_model.get('mu')
        elif(not hasattr(self, 'mu')):        
            self.mu = self._LIST_ARGS_OPT['mu'][1]
 
        if(self.setup  == "1"):  
            assert (self.n_controls == 1), "number of control functions don't match"
            U = self.control_fun[0]
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
            self._H = hamiltonian(static, dynamic, basis=self._ss, dtype=np.float64)
        else:
            # Only one setup implemeneted so far
            raise NotImplementedError()

    def _setup_fom_bh1d_bricks(self, **args_model):
        """ add new methods which can be used e.g. to define the FOM """                
        # New functions to be used to compute the FOM        
        if (self._flag_basis_symm):
            def avg_var_occup(V_symm):
                V = self._P.dot(V_symm)
                n_var_sites = [mod.pcModel_qspin._h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
            self._avg_var_occup = None
            self._fom_func['varN'] = None
            
            # measured variance TODO
            self._avg_varN_measured = None
            self._fom_func['varN5'] = None
            self._fom_func['varN10'] = None
            self._fom_func['varN100'] = None
            self._fom_func['varN1000'] = None
            self._fom_func['varN10000'] = None
            self._fom_func['varN100000'] = None
        else:
            #bad code on several aspects: 
            def avg_var_occup(V):
                """ occupation number variance averaged over all the sites"""
                n_var_sites = [mod.pcModel_qspin._h_variance(op, V) for op in self._op_n_sites]
                avg_var_occup = np.average(n_var_sites)
                return avg_var_occup
            
            def proba_basis(V):
                """ Return for a given state the probabilities associated to the
                underlying basis
                """
                proba = np.square(np.abs(np.squeeze(V))) 
                return proba
            
            def proj_onto_sites(V, nb):
                """ Samples of spatial projective measurements, i.e. 
                projection in terms of particle position return list of arrays 
                containing the number of particles measured at each site
                """
                proba = proba_basis(V) # probabilities of being projected onto one basis element
                sum_prob = np.sum(proba) # verif probas are fine (i.e. sum to one)
                if(np.allclose(sum_prob, 1)):
                    proba = proba/sum_prob
                else:
                    logger.warning('In avg_var_occup')
                meas_index = np.random.choice(len(proba), nb, p=proba) # generate outcome of measurement
                samples = np.array([self._basis_fock[i] for i in meas_index]) #returned in an exploitable form
                return samples
            
            def avg_var_occup_measured(V, nb):
                meas_occup = proj_onto_sites(V,nb)
                meas_var = np.var(meas_occup)
                meas_avg_var = np.mean(meas_var)
                return meas_avg_var
            
            def allone_proba(V):
                """ proba associated to the |11...11> states"""
                proba = proba_basis(V)
                assert np.all(proba < 1+1e-6) and np.all(proba > -1e-6)
                proba = np.clip(proba, 0., 1.)
                proba_allone = proba[self._index_basis_allone]
                return proba_allone
                
                
            def allone_frequency(V, nb):
                """ frequency associated to the |11...11> states obtained over 
                nb repetitions"""
                proba_allone = allone_proba(V)
                freq = np.random.binomial(nb, proba_allone) / nb
                return freq
            
            def averageone_proba(V):
                """ average (over all the sites) proba associated to seing exactly
                one particle per site i.e. p = <p(n_i = 1)>_i """
                proba = proba_basis(V)
                proba_each_one = [np.sum(proba[ind]) for ind in self._index_basis_oneone]
                proba_avg = np.average(proba_each_one)
                assert (proba_avg < 1+1e-6) and (proba_avg > -1e-6)
                proba_avg = np.clip(proba_avg, 0., 1.)
                return proba_avg
            
            def averageone_frequency(V, nb):
                """ frequency (over all the sites) proba associated to seing ecatly
                one particle i.e. p = <p(n_i = 1)>_i """
                samples = proj_onto_sites(V, nb)
                freq = np.sum(samples == 1) / (nb * self.L)
                return freq
            
            def eachone_proba(V):
                """ probas associated to seing exactly one particle per site 
                i.e. p = [p(n_1 = 1),...,p(n_1 = L)]"""
                proba = proba_basis(V)
                proba_each_one = np.array([np.sum(proba[ind]) for ind in self._index_basis_oneone])
                assert np.all(proba_each_one < 1+1e-6) and np.all(proba_each_one > -1e-6)
                proba_each_one = np.clip(proba_each_one, 0., 1.)
                return proba_each_one
            
            def eachone_frequency(V, nb):
                """ frequency (for each site) of seing ecatly one particle"""
                samples = proj_onto_sites(V, nb)
                freq = np.average(samples == 1, axis=0)
                return freq
            
            
            list_freq = [1,2,3,4,5,10,25,50,100,500,1000,5000,10000,50000,100000,500000]
            # based on variance of the occupation number            
            self._avg_var_occup = avg_var_occup
            self._fom_func['varN'] = self._avg_var_occup
            self._avg_varN_measured = avg_var_occup_measured
            self._fom_func.update({'varN'+str(nb):(lambda x, nb=nb: self._avg_varN_measured(x, nb)) for nb in list_freq})
            
            # based on frequency of seeing one boson everywhere
            self._allone_proba = allone_proba
            self._fom_func['freqMI'] = self._allone_proba
            self._allone_frequency = allone_frequency
            self._fom_func.update({'freqMI'+str(nb):(lambda x, nb=nb: self._allone_frequency(x, nb)) for nb in list_freq})

            # measured average freq of one particle realization
            self._averageone_proba = averageone_proba
            self._fom_func['freqAvgOne'] = self._averageone_proba
            self._averageone_frequency = averageone_frequency
            self._fom_func.update({'freqAvgOne'+str(nb):(lambda x, nb=nb: self._averageone_frequency(x, nb)) for nb in list_freq})
            
            # measured freq one particle realization for each site
            self._eachone_proba = eachone_proba
            self._fom_func['freqEachOne'] = self._eachone_proba
            self._eachone_frequency = eachone_frequency
            self._fom_func.update({'freqEachOne'+str(nb):(lambda x, nb=nb: self._eachone_frequency(x, nb)) for nb in list_freq})

    def _state_to_occup_nb(self, st):
        exp_occup = np.array([op.expt_value(st) for op in self._op_n_sites])
        return exp_occup

    def _get_basis_to_fock(self, i):
        """ get the i-th basis vector represented in the focks basis 
        Works only if no symm is used"""
        if(self._flag_basis_symm):
            logger.warning("cannot use _get_basis_to_fock when symmetrieshas"
                           "been used to build the basis")
            repr_fock = None
        else:
            Ns = self._ss.Ns
            repr_vector = np.zeros(Ns)
            repr_vector[i] = 1
            repr_fock = self._state_to_occup_nb(repr_vector)
            
        return repr_fock




### ======================= ###
# TESTING
### ======================= ###
if(__name__ == '__main__'): 
    BH1D.info()
    
    # Create a 1D BH chain linearly driven
    # evolve the GS of H(t=0) to T
    # observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
    T = 3

    linear_ramp = pf.LinearFunc(w=1/T , bias =0)
    fom_name = ['f2t2:neg_fluence:0.0001_smooth:0.0005', 'f2t2', 'varN:sqrt', 'fluence', 'smooth', 'varN']
    fom_name_last = ['last:'+f for f in fom_name]

    dico_simul = {'control_obj':linear_ramp, 'L':6, 'Nb':6, 'mu':0, 'T':T, 'dt':0.01, 
                'flag_intermediate':False, 'setup':'1', 'state_init':'GS_i', 
                'state_tgt':'GS_f', 'fom':fom_name_last}

    simul = BH1D(**dico_simul)
    if(False):
        res_fom = simul.Simulate(fom = fom_name, store = True, debug = False)
    
    #study Evol of the population in the first energy levels
    if(False):    
        simul = BH1D(**dico_simul)
        nb_ev = 10
        state_t = simul.EvolutionPopAdiab(nb_ev = nb_ev)
        simul.plot_pop_adiab()
        
    

    
    