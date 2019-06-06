# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""

import logging 
logger = logging.getLogger(__name__)

import sys
import pdb
import numpy as np
from numpy import inf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy.random as rdm
import time
from qutip import sigmax, sigmaz, sigmay, mesolve, Qobj, Options, identity, tensor, basis, cnot, rx, ry, controlled_gate
import qutip.logging_utils as logging
logger = logging.get_logger()
from scipy.special import erfinv
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
from fidcompnoisy import FidCompUnitNoisy
import scipy.optimize 

if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
else:
    sys.path.append('/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/QuantumSimulation/')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
    
X, Y, Z, I, zero, one = sigmax(), sigmay(), sigmaz(), identity(2), basis(2,0), basis(2,1)
class BatchFS(BatchBase):
    """Implement few shots simulations for batching.
    Provides different methods for optimization / estimation
            
    """
    
    def setup_QTIP_model(self, model_config):
        """ Setup the model in QuTip allowing to compute the dynamics and FoM
        """        
        #print(zero)
        model = model_config['model']
        verbose = model_config.get('verbose', False)
        agg = model_config.get('aggregate', 'no') #implemented 'no', 'fid', 'close', 'perfect'
        agg = 'fid' if(agg == True) else agg
        agg = 'no' if(agg == False) else agg


        x_tgt = None
        #define an Hamiltonian H(t) = - Sz - hx(t) Sx
        if(model == 1):
            T = model_config['T']
            lbnd = -4
            ubnd = 4
            H_d, H_c, H_i, H_f = - Z, [-X], -Z + 2*X, -Z -2*X
            self.n_ctrls = 1
            self.phi_0 = H_i.eigenstates(eigvals = 1)[1][0]
            self.phi_tgt = H_f.eigenstates(eigvals = 1)[1][0]
            self.n_ts = model_config['n_ts']
            self.n_params = self.n_ts * self.n_ctrls
            self.domain = [(lbnd, ubnd) for _ in range(self.n_params)]
            
            logger.info("Creating and configuring control optimisation objects")
            # Create the OptimConfig object
            cfg = optimconfig.OptimConfig()
            cfg.log_level = logging.INFO

            # Create the dynamics object
            dyn = dynamics.DynamicsUnitary(cfg)
            dyn.num_tslots = self.n_params
            dyn.evo_time = T
            dyn.target = self.phi_tgt
            dyn.initial = self.phi_0
            dyn.drift_dyn_gen = H_d
            dyn.ctrl_dyn_gen = H_c
            dyn.params_lbnd = -4
            dyn.params_ubnd = 4
        
            # Define the figure of merit
            #noise output      
            noise_n_meas = model_config.get('noise_n_meas', 1)
            if noise_n_meas == 'inf': noise_n_meas = inf 
            #if noise_n_meas > 10000 : noise_n_meas = inf
            noise_mean = model_config.get('noise_mean', 1)
            noise_std = model_config.get('noise_std', 0)
            noise_input = model_config.get('noise_input', 0)
            noise_b_meas = model_config.get('noise_b_meas')
            noise_type = model_config.get('noise_type')         
    
            # CUSTOM FIDELITY COMPUTER
            fidcomp = FidCompUnitNoisy(dyn)
            fidcomp.noise_mean = noise_mean
            fidcomp.noise_std = noise_std
            fidcomp.noise_n_meas = noise_n_meas
            fidcomp.noise_b_meas = noise_b_meas
            fidcomp.noise_type = noise_type#should be after others
            dyn.fid_computer = fidcomp
            zero_amps = np.zeros([self.n_ts, self.n_ctrls])
            dyn.initialize_controls(zero_amps)
            self.fid_zero = dyn.fid_computer.get_fidelity_perfect()
            logger.info("With zero field fid (square): {}".format(model, self.fid_zero))        
    
            self.dyn = dyn
            self.n_meas = noise_n_meas
            self.p_tgt = self.dyn.fid_computer.get_ptarget()
            self.nb_output = 1 if fidcomp.noise_b_meas is None else len(fidcomp.noise_b_meas)
            logger.info("Optim model {0} with {1} params between {2}-{3} and T= {4}".format(model, 
                                            self.n_params, dyn.params_lbnd, dyn.params_ubnd, T))
            logger.info("dynamics has been created with QuTip and saved self.dyn")
            
            #use during optimization
            def f(x, verbose = verbose, noise=noise_input, N = None, aggregate = None):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(np.ndim(x_n)>1):
                    res = np.array([f(x_one, verbose,noise=0, N=N,aggregate=aggregate) for x_one in x_n])
                else:
                    self.call_f_single += 1
                    self.call_f += 1
                    amps = np.reshape(x_n, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    res = self.dyn.fid_computer.get_noisy_fidelity()
                    if(verbose):
                        res_perfect = self.dyn.fid_computer.get_fidelity_perfect()
                        print([res, res_perfect])
                        print(np.squeeze(self.dyn.ctrl_amps))
                return np.atleast_1d(res)
    
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    res = np.array([f_test(x_one) for x_one in x])
                else:
                    amps = np.reshape(x, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    res = self.dyn.fid_computer.get_fidelity_perfect()
                    if(verbose):
                        print(res)
                        print(np.squeeze(self.dyn.ctrl_amps))
                return res

        #Simple H(a, b) = a (Z(1-b))#
        #A phi_target is randomly generated            
        elif(model==2):
            self.n_params = 2
            options_evolve = Options(store_states=True)
            self.phi_0 = Qobj(np.array([1., 0.]))
            all_e = [X, Y, Z]
            self.n_meas = model_config.get('noise_n_meas', 1)
            if self.n_meas == 'inf': self.n_meas = inf
            self.n_meas_index = model_config.get('n_meas_index')         
            noise_input = model_config.get('noise_input', 0)
            self.nb_output = 3 if self.n_meas_index is None else 1

            #gen random phi_target and p_tgt associated
            if model_config.get('model_version', 1) == 1:
                self.domain = [(0, 2 * np.pi), (0,1)]
                x_tgt = np.array([rdm.uniform(*d) for d in self.domain])
                H_tgt = get_HZY(x_tgt)
            elif model_config.get('model_version', 1) == 2:
                self.domain = [(0, 2 * np.pi), (0,1)]
                x_tgt = np.array([rdm.uniform(*d) for d in self.domain])
                H_tgt = get_HZY_2(x_tgt)
            else:
                raise NotImplementedError()
            
            
            
            evol_tgt = mesolve(H_tgt, self.phi_0, tlist = [0., 1.], e_ops=all_e, options = options_evolve)
            final_expect_tgt = [e[-1] for e in evol_tgt.expect]
            self.phi_tgt = evol_tgt.states[-1]
            self.p_tgt = np.array([(1 + e)/2 for e in final_expect_tgt])
            self.fid_zero = None #we are not interested by this figure for this model
            #logger.info
            
            def f(x, verbose = verbose, noise=noise_input, N = self.n_meas, aggregate = agg):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(x_n.shape[0] != self.n_params):
                    res = np.array([f(x_one, verbose,noise=0, N=N,aggregate=aggregate) for x_one in x_n])
                else:
                    self.call_f_single += 1
                    H = get_HZY(x)
                    e = all_e if self.n_meas_index is None else [all_e[self.n_meas_index]]
                    evol = mesolve(H, self.phi_0, tlist = [0., 1.], e_ops=e, options = options_evolve)
                    #final_state = evol.states[-1]
                    final_expect = [e[-1] for e in evol.expect]
                    proba = np.array([(1 + e)/2 for e in final_expect])
                    assert np.any(proba < 1 + 1e-5), "proba > 1: {}".format(proba)
                    assert np.any(proba > -1e-5), "proba > 1: {}".format(proba)
                    proba = np.clip(proba, 0, 1)
                    if (N == np.inf): 
                        res = proba 
                        self.call_f += 1
                    else:
                        res = rdm.binomial(N, proba) / N
                        self.call_f += len(e) * N
                    if aggregate == 'fid':
                        res = 1/2 * (1 + np.dot(2*self.p_tgt-1,2*res-1))
                    if aggregate == 'close':
                        res = 1 - np.average(np.abs(self.p_tgt-res))
                    if(verbose): print(x, res, proba)
                return np.atleast_1d(res)
    
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    fid = np.array([f_test(x_one) for x_one in x])
                else:
                    H = get_HZY(x)
                    evol = mesolve(H, self.phi_0, tlist = [0., 1.], options = options_evolve)
                    final_state = evol.states[-1]
                    fid_prenorm = np.square(np.abs((self.phi_tgt.dag() * final_state).tr()))
                    fid = fid_prenorm / self.phi_tgt.norm()
                    if(verbose): print(x, fid)
                return fid
        
        elif(model==3):
            """ 2 qubits gate -- not finished yet"""
            self.n_ts = model_config['n_ts']
            T = model_config['T']
            self.n_ctrls = 4
            l_bnd = -5.
            u_bnd = 5.
            self.n_params = self.n_ctrls * self.n_ts
            self.domain = [(l_bnd , u_bnd) for _ in range(self.n_params)]
            self.phi_0 = identity(4) #start
            self.phi_tgt = cnot() # tget
            
            ## MODEL
            H_d = 0.5*(tensor(X, X) + tensor(Y, Y) + tensor(Z, Z)) # Drift Hamiltonian
            H_c = [tensor(X, I), tensor(Y, I), tensor(I, X), tensor(I, Y)] # The (four) control Hamiltonians
            
            logger.info("Creating and configuring control optimisation objects")
            # Create the OptimConfig object
            cfg = optimconfig.OptimConfig()
            cfg.log_level = logging.INFO

            # Create the dynamics object
            dyn = dynamics.DynamicsUnitary(cfg)
            dyn.num_tslots = self.n_ts
            dyn.evo_time = T
            dyn.target = self.phi_tgt
            dyn.initial = self.phi_0
            dyn.drift_dyn_gen = H_d
            dyn.ctrl_dyn_gen = H_c
            dyn.params_lbnd = l_bnd
            dyn.params_ubnd = u_bnd
        
            # Define the figure of merit
            #noise output
            basis_init_default = [tensor(zero, zero), tensor(zero, one), tensor(one, zero), tensor(one, one)]
            basis_meas_default = [tensor(zero, zero), tensor(zero, one), tensor(one, one), tensor(one, zero)]
            noise_n_meas = model_config.get('noise_n_meas', 1)
            if noise_n_meas == 'inf': noise_n_meas = inf
            noise_mean = model_config.get('noise_mean', 1)
            noise_std = model_config.get('noise_std', 0)
            noise_input = model_config.get('noise_input', 0)
            noise_init_states = model_config.get('noise_b_init', basis_init_default)
            noise_b_meas = model_config.get('noise_b_meas', basis_meas_default)
            noise_type = model_config.get('noise_type')         
    
            # CUSTOM FIDELITY COMPUTER
            fidcomp = FidCompUnitNoisy(dyn)
            fidcomp.noise_mean = noise_mean
            fidcomp.noise_std = noise_std
            fidcomp.noise_n_meas = noise_n_meas
            fidcomp.noise_b_meas = noise_b_meas
            fidcomp.noise_type = noise_type#should be after others
            fidcomp.noise_b_init = noise_init_states
            
            dyn.fid_computer = fidcomp
            zero_amps = np.zeros([self.n_ts, self.n_ctrls])
            dyn.initialize_controls(zero_amps)
            self.fid_zero = dyn.fid_computer.get_fidelity_perfect()
            logger.info("With zero field fid (square): {}".format(model, self.fid_zero))        
    
            self.dyn = dyn
            self.n_meas = noise_n_meas
            self.p_tgt = self.dyn.fid_computer.get_ptarget()#self.dyn.fid_computer.get_ptarget()
            self.nb_output = 1 if fidcomp.noise_b_meas is None else len(fidcomp.noise_b_meas)
            logger.info("Optim model {0} with {1} params between {2}-{3} and T= {4}".format(model, 
                                            self.n_params, dyn.params_lbnd, dyn.params_ubnd, T))
            logger.info("dynamics has been created with QuTip and saved self.dyn")
            
            #use during optimization
            def f(x, verbose = verbose, noise=noise_input, N=None, aggregate = None):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(np.ndim(x_n)>1):
                    res = np.array([f(x_one, verbose,noise=0, N=N, aggregate=aggregate) for x_one in x_n])
                else:
                    self.call_f += 1
                    self.call_f_single += 1
                    amps = np.reshape(x_n, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    
                    res = self.dyn.fid_computer.get_noisy_fidelity()
                    if(verbose):
                        res_perfect = self.dyn.fid_computer.get_fidelity_perfect()
                        print([res, np.average(res), res_perfect])
                        print(np.squeeze(self.dyn.ctrl_amps))
                return np.atleast_1d(res)
    
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    res = np.array([f_test(x_one) for x_one in x])
                else:
                    amps = np.reshape(x, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    res = self.dyn.fid_computer.get_fidelity_perfect()
                    if(verbose):
                        print(res)
                        print(np.squeeze(self.dyn.ctrl_amps))
                return res
        
        elif(model==4):
            ## 2 qubits dynamics with only two params
            self.n_params = 2
            self.domain = [(0, 2 * np.pi), (0, 2 * np.pi)]
            self.phi_0 = tensor(zero, zero)
            #all_e = [tensor(X, I), tensor(Y, I), tensor(Z, I), tensor(I, X), tensor(I, Y), tensor(I, Z)]
            all_e = [tensor(X, X), tensor(Y, Y), tensor(Z, Z)]
            self.n_meas = model_config.get('noise_n_meas', 1)
            if self.n_meas == 'inf': self.n_meas = inf
            self.n_meas_index = model_config.get('n_meas_index')         
            n_input = model_config.get('noise_input', 0)
            n_gate = model_config.get('noise_gate', 0)
            n_ro = model_config.get('noise_readout', 0)
            repetition = model_config.get('noise_repetition', 0)
            self.nb_output = 3 if self.n_meas_index is None else 1
        
            #Fixed phi_tgt (Bell state) - may be random later on
            #x_tgt = np.array([rdm.uniform(*d) for d in self.domain])
            x_tgt = np.array((np.pi/2, np.pi))
            U_tgt = get_UCPRY(x_tgt, n_gate=0)
            self.phi_tgt = U_tgt * self.phi_0
            self.e_tgt = [real_with_test(e.matrix_element(self.phi_tgt, self.phi_tgt)) for e in all_e]
            self.p_tgt = np.array([(1 + e)/2 for e in self.e_tgt])
            self.fid_zero = None #we are not interested by this figure for this model

        
            def f(x, verbose = verbose, N = self.n_meas, aggregate = agg, n_input=n_input, 
                  n_gate=n_gate, n_ro=n_ro, repetition = repetition):
                """
                n_input: noise in the input
                n_ro: noise in readout
                n_gate: noise in gates
                repetition: when n_gate is not 0 how to deal with it
                            
                """
                x_n = np.clip(x + rdm.norm(0.0, n_input, np.shape(x)), dyn.params_lbnd, 
                              dyn.params_ubnd) if n_input>0 else x
                if(x_n.shape[0] != self.n_params):
                    res = np.array([f(x_one, verbose, N=N, aggregate=aggregate, 
                                      n_input=0, n_gate=n_gate,n_ro=n_ro, repetition=repetition) for x_one in x_n])
                else:
                    list_e = all_e if self.n_meas_index is None else [all_e[self.n_meas_index]]
                    if (n_gate >0) and (repetition ==1) and (N < np.inf) and (N>1):
                        measurement = [f(x, verbose, N=1, aggregate=aggregate, n_input=n_input,
                                         n_gate=n_gate, n_ro=n_ro, repetition = 0) for _ in range(N)]                        
                    else:
                        self.call_f_single += 1
                        U = get_UCPRY(x, n_gate=n_gate)                    
                        proba = get_proba_after_evol(U, self.phi_0, list_e, n_ro)
                        measurement, n_call = get_measurement_from_proba(proba, N, aggregate, self.p_tgt)
                        self.call_f += n_call
                    res = np.squeeze(np.average(np.atleast_2d(measurement),0))+0
                    if(verbose): print(x, res, proba)
                return np.atleast_1d(res)
        
        
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    fid = np.array([f_test(x_one) for x_one in x])
                else:
                    U = get_UCPRY(x)
                    phi_f = U * self.phi_0
                    fid_prenorm = np.square(np.abs((self.phi_tgt.dag() * phi_f).tr()))
                    fid = fid_prenorm / self.phi_tgt.norm()
                    if(verbose): print(x, fid)
                return fid
            

        elif(model==5):
            self.n_params = 6
            if model_config.get('reduce_domain', False):
                self.domain = [(-np.pi/4, 5 * np.pi/4)]*6
            else:
                self.domain = [(0, 2 * np.pi)]*6
            self.phi_0 = tensor(zero, zero, zero)
            all_e = [tensor(I, Z, Z), tensor(X, X, X), tensor(X, Y, Y), tensor(Y, X, Y), tensor(Y, Y, X), tensor(Z, I, Z), tensor(Z,Z,I)]
            self.n_meas = model_config.get('noise_n_meas', 1)
            if self.n_meas == 'inf': self.n_meas = inf
            self.n_meas_index = model_config.get('n_meas_index')         
            n_input = model_config.get('noise_input', 0)
            n_gate = model_config.get('noise_gate', 0)
            n_ro = model_config.get('noise_readout', 0)
            repetition = model_config.get('noise_repetition', 0)
            self.nb_output = 3 if self.n_meas_index is None else 1
        
            #Fixed phi_tgt (Bell state) - may be random later on
            x_tgt = np.array((np.pi/2, np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2))     #x_tgt = np.array([rdm.uniform(*d) for d in self.domain])
            U_tgt = get_UGHZ(x_tgt, n_gate=0)
            # all possible solutions in the domain
            x_tgt = np.array([[1., 3., 2., 3., 1., 3.], [3., 1., 2., 1., 3., 3.], [3., 3., 4., 1., 1., 3.],
                              [3., 1., 4., 3., 1., 1.], [1., 1., 4., 3., 3., 3.],[1., 3., 4., 1., 3., 1.],
                              [1., 1., 0., 3., 3., 3.],[1., 1., 2., 1., 1., 1.],[3., 3., 2., 3., 3., 1.],
                              [1., 3., 0., 1., 3., 1.],[3., 1., 0., 3., 1., 1.],[3., 3., 0., 1., 1., 3.]]) * np.pi/2
            
            self.phi_tgt = U_tgt * self.phi_0
            self.e_tgt = [real_with_test(e.matrix_element(self.phi_tgt, self.phi_tgt)) for e in all_e]
            self.p_tgt = np.array([(1 + e)/2 for e in self.e_tgt])
            self.fid_zero = None #we are not interested by this figure for this model
            
            

            
            
            def f(x, verbose = verbose, N = self.n_meas, aggregate = agg, n_input=n_input, 
                  n_gate=n_gate, n_ro=n_ro, repetition = repetition):
                """
                n_input: noise in the input
                n_ro: noise in readout
                n_gate: noise in gates
                repetition: when n_gate is not 0 how to deal with it
                            
                """
                x_n = np.clip(x + rdm.norm(0.0, n_input, np.shape(x)), dyn.params_lbnd, 
                              dyn.params_ubnd) if n_input>0 else x
                if(x_n.shape[0] != self.n_params):
                    res = np.array([f(x_one, verbose, N=N, aggregate=aggregate, 
                                      n_input=0, n_gate=n_gate,n_ro=n_ro, repetition=repetition) for x_one in x_n])
                else:
                    list_e = all_e if self.n_meas_index is None else [all_e[self.n_meas_index]]
                    if (n_gate >0) and (repetition ==1) and (N < np.inf) and (N>1):
                        measurement = [f(x, verbose, N=1, aggregate=aggregate, n_input=n_input,
                                         n_gate=n_gate, n_ro=n_ro, repetition = 0) for _ in range(N)]                        
                    else:
                        self.call_f_single += 1
                        U = get_UGHZ(x, n_gate=n_gate)                    
                        proba = get_proba_after_evol(U, self.phi_0, list_e, n_ro)
                        measurement, n_call = get_measurement_from_proba(proba, N, aggregate, self.p_tgt)
                        self.call_f += n_call
                    res = np.squeeze(np.average(np.atleast_2d(measurement),0))+0
                    if(verbose): print(x, res, proba)
                return np.atleast_1d(res)
        
        
        
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    fid = np.array([f_test(x_one) for x_one in x])
                else:
                    U = get_UGHZ(x)
                    phi_f = U * self.phi_0
                    fid_prenorm = np.square(np.abs((self.phi_tgt.dag() * phi_f).tr()))
                    fid = fid_prenorm / self.phi_tgt.norm()
                    if(verbose): print(x, fid)
                return fid
        
        elif(model == 6):
            T = model_config['T']
            lbnd = -4
            ubnd = 4
            H_d, H_c, H_i, H_f = - Z, [-X], -Z + 2*X, -Z -2*X
            self.n_ctrls = 1
            self.phi_0 = H_i.eigenstates(eigvals = 1)[1][0]
            self.phi_tgt = H_f.eigenstates(eigvals = 1)[1][0]
            self.n_ts = model_config['n_ts']
            self.n_params = self.n_ts * self.n_ctrls
            self.domain = [(lbnd, ubnd) for _ in range(self.n_params)]
            x_tgt = None
            if model_config.get('model_version', 1) == 1: # shot 2 target
                all_e = [self.phi_tgt * self.phi_tgt.dag()] # projector onto phi_tgt
                def e_to_p(e):
                    return e
            elif model_config.get('model_version', 1) == 2:
                all_e = [X, Y, Z]
                def e_to_p(e):
                    return (1 + e)/2
            else:
                raise NotImplementedError()

            self.e_tgt = [real_with_test(e.matrix_element(self.phi_tgt, self.phi_tgt)) for e in all_e]
            self.p_tgt = np.array([(1 + e)/2 for e in self.e_tgt])

            logger.info("Creating and configuring control optimisation objects")
            # Create the OptimConfig object
            cfg = optimconfig.OptimConfig()
            cfg.log_level = logging.INFO

            # Create the dynamics object
            dyn = dynamics.DynamicsUnitary(cfg)
            dyn.num_tslots = self.n_params
            dyn.evo_time = T
            dyn.target = self.phi_tgt
            dyn.initial = self.phi_0
            dyn.drift_dyn_gen = H_d
            dyn.ctrl_dyn_gen = H_c
            dyn.params_lbnd = -4
            dyn.params_ubnd = 4
        
            # Define the figure of merit
            #noise output      
            self.n_meas = model_config.get('noise_n_meas', 1)
            if self.n_meas == 'inf': self.n_meas = inf       
            noise_input = model_config.get('noise_input', 0)
            self.nb_output = len(all_e)
    
            #  FIDELITY COMPUTER
            fidcomp = FidCompUnitNoisy(dyn)
            dyn.fid_computer = fidcomp
            zero_amps = np.zeros([self.n_ts, self.n_ctrls])
            dyn.initialize_controls(zero_amps)
            self.fid_zero = dyn.fid_computer.get_fidelity_perfect()
            logger.info("With zero field fid (square): {}".format(model, self.fid_zero))        
    
            self.dyn = dyn
            logger.info("Optim model {0} with {1} params between {2}-{3} and T= {4}".format(model, 
                                            self.n_params, dyn.params_lbnd, dyn.params_ubnd, T))
            logger.info("dynamics has been created with QuTip and saved self.dyn")
            
            #use during optimization
            def f(x, verbose = verbose, noise=noise_input, N = self.n_meas, aggregate = agg):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(np.ndim(x_n)>1):
                    res = np.array([f(x_one, verbose,noise=0, N=N,aggregate=aggregate) for x_one in x_n])
                else:
                    self.call_f_single += 1
                    amps = np.reshape(x_n, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    res_perfect = self.dyn.fid_computer.get_fidelity_perfect()
                    k = self.dyn.tslot_computer._get_timeslot_for_fidelity_calc()
                    phi_T = Qobj(self.dyn._fwd_evo[k])

                    final_expect = np.array([real_with_test(e.matrix_element(phi_T, phi_T)) for e in all_e])
                    proba = e_to_p(final_expect)
                    assert np.any(proba < 1 + 1e-5), "proba > 1: {}".format(proba)
                    assert np.any(proba > -1e-5), "proba > 1: {}".format(proba)
                    proba = np.clip(proba, 0, 1)
                    if (N == np.inf): 
                        res = proba 
                        self.call_f += 1
                    else:
                        res = rdm.binomial(N, proba) / N
                        self.call_f += len(all_e) * N
                    if aggregate == 'fid':
                        res = 1/2 * (1 + np.dot(2*self.p_tgt-1,2*res-1))
                    if aggregate == 'close':
                        res = 1 - np.average(np.abs(self.p_tgt-res))
                    if(verbose): print(x, res, proba, res_perfect)
                return np.atleast_1d(res)
    
            #Use for testing the optimal pulse
            def f_test(x, verbose = verbose):
                if(x.shape[0] != self.n_params):
                    fid = np.array([f_test(x_one) for x_one in x])
                else:
                    amps = np.reshape(x, (self.n_ts, self.n_ctrls))
                    self.dyn.update_ctrl_amps(amps)
                    fid = self.dyn.fid_computer.get_fidelity_perfect()
                    if(verbose): print(x, fid)
                return fid

        else:
            raise NotImplementedError()
        #figure of merit
        self.f_tgt = None if (self.p_tgt is None) else probit(self.p_tgt)

        return f, f_test, x_tgt


    def run_one_procedure(self, config):
        """ implemention of what is a procedure.
        """
        self.call_f = 0
        self.call_f_single = 0
        self.call_f_test = 0
        model_config = config['model']
        optim_config = config['optim']
        if(model_config.get('debug', False) or optim_config.get('debug', False)):
            pdb.set_trace()
        model_config['aggregate'] = optim_config.get('aggregate', 'no') # ARgh
        np.random.seed(config['_RDM_SEED'])
        self.f, self.f_test, self.x_tgt = self.setup_QTIP_model(model_config)
        
        #setting up the optimizer
        type_optim = optim_config.get('type_optim', 'BO')
        logger.info('p_tgt: {0}'.format(self.p_tgt))
        logger.info('f_tgt: {0}'.format(self.f_tgt))
        nb_init = optim_config['nb_init']
        nb_iter = optim_config['nb_iter']
        save_extra = optim_config.get('save_extra', False)
        nb_total = nb_init + nb_iter
        time_start = time.time()
        
        if(type_optim == 'RANDOM'):
            X_init = np.transpose([np.atleast_1d(np.random.uniform(*d, nb_total)) for d in self.domain])    
            Y_init = self.f(X_init)
            if(self.p_tgt is not None):
                assert len(self.p_tgt) == Y_init.shape[1]
                temp = np.abs(Y_init - np.repeat(np.squeeze(self.p_tgt)[np.newaxis, :], len(Y_init), 0))
            else:
                temp = Y_init
            index_best = np.argmin(np.average(temp, 1))
            x_seen = X_init[index_best]
            x_exp = x_seen
            test = self.f_test(x_exp)
            abs_diff = 1 - test

            dico_res = {'test':test, 'p_tgt':self.p_tgt, 'f_tgt':self.f_tgt, 
                'nb_output':self.nb_output, 'x':x_seen[0], 'x_exp':x_exp[0], 
                'fid_zero_field':self.fid_zero, 'phi_0': Qobj2array(self.phi_0), 
                'phi_tgt':Qobj2array(self.phi_tgt), 'time_fit':0, 'time_suggest':0} 

        if(type_optim == 'CRAB'):
            raise NotImplementedError()
        
        elif(type_optim == 'GRAPE'):
            not NotImplementedError()
        
        elif(type_optim == 'BFGS'):
            X_init = np.array([rdm.uniform(*d) for d in self.domain]) 
            f_BFGS = lambda x: 1-self.f(x)
            optim = scipy.optimize.minimize(f_BFGS, x0=X_init, method='BFGS', options={'maxiter': nb_init+nb_iter-1, 'disp': False, 'return_all': False})
            cum_time = time.time() - time_start
            x_exp = optim['x']
            x_seen = x_exp
            test = self.f_test(x_exp)
            #test_exp = f_test(xy_exp[0])
            abs_diff = 1 - test
            dico_res = {'test':test, 'p_tgt':self.p_tgt, 'f_tgt':self.f_tgt,
                        'x':x_seen, 'x_exp':x_exp, 'abs_diff':abs_diff,
                        'fid_zero_field':self.fid_zero,'phi_0': Qobj2array(self.phi_0), 
                        'phi_tgt':Qobj2array(self.phi_tgt), 'time_suggest':cum_time, 
                        'time_fit':0, 'message_BFGS':optim['message']}        
        
        elif(type_optim == 'LBFGSB'):
            X_init = np.array([rdm.uniform(*d) for d in self.domain]) 
            f_BFGS = lambda x: 1-self.f(x)
            optim = scipy.optimize.minimize(f_BFGS, x0=X_init, method='L-BFGS-B',bounds = self.domain, options={'maxiter': nb_init+nb_iter-1, 'disp': False, 'return_all': False})
            cum_time = time.time() - time_start
            x_exp = optim['x']
            x_seen = x_exp
            test = self.f_test(x_exp)
            #test_exp = f_test(xy_exp[0])
            abs_diff = 1 - test
            dico_res = {'test':test, 'p_tgt':self.p_tgt, 'f_tgt':self.f_tgt,
                        'x':x_seen, 'x_exp':x_exp, 'abs_diff':abs_diff,
                        'fid_zero_field':self.fid_zero,'phi_0': Qobj2array(self.phi_0), 
                        'phi_tgt':Qobj2array(self.phi_tgt), 'time_suggest':cum_time, 
                        'time_fit':0, 'message_BFGS':optim['message'],'x_init':X_init}  

             
        #Bayesian Optimization 2 flavors 'BO' and 'BO_NOOPTIM'
        # 'BO' classical bayesian optimization
        # 'BO_NOOPTIM' all the x are randomly generated and GP is fitted 
        #              x_best is decided based on this model
        elif 'BO' in type_optim: 
            ### Main
            dico_res, self.time_all_bo, self.time_fit_bo, self.time_suggest_bo = {}, 0, 0, 0
            self.set_up_BO(optim_config)
            self.max_time_bo = self.bo_args['max_time_bo'] 
            if(save_extra):
                    dico_res.update(self.get_info_BO(i_beg = None, i_end = None, tag='init0_'))
            self.BO.run_optimization(max_iter = self.bo_args['nb_iter_bo'], eps = 0, max_time = self.max_time_bo)
            self.update_BO_time()
            if(save_extra):
                dico_res.update(self.get_info_BO(i_beg = None, i_end = None, tag='explor0_'))
    
            nb_exploit = self.bo_args['nb_exploit']
            if nb_exploit>0:
                self.set_up_BO_exploit()
                logger.info('Exploitation (i.e. ucb with k=0) for {}'.format(nb_exploit))
                self.BO.run_optimization(nb_exploit, max_time = self.max_time_bo)
                self.update_BO_time()
                if(save_extra):
                    dico_res.update(self.get_info_BO(i_beg = None, i_end = None, tag='exploit0_'))
            
            ### Extra steps
            nb_polish = self.bo_args['nb_polish']
            nb_to_keep = self.bo_args['nb_to_keep']
            nb_more = self.bo_args['nb_more']
            nb_iter_polish = self.bo_args['nb_iter_polish']
            polish_step = 0
            X_keep_track = np.c_[(self.BO.X, np.zeros(len(self.BO.X)))]
            Y_keep_track = np.c_[(self.BO.Y, np.zeros(len(self.BO.Y)))]
            self.save_hp_values()
            while nb_polish > 0 and self.max_time_bo > 0:
                more = nb_more[polish_step]
                new_iter = nb_iter_polish[polish_step]
                keep = nb_to_keep[polish_step]
                logger.info('Polish, nb to keep {}, X times more shots {} '.format(keep, nb_more))
                nb_polish -= 1  
                polish_step += 1
                self.set_up_BO(optim_config, nb_restrict_data = keep, restrict_domain = True, adapt_shots=more)   
                if(save_extra):
                    dico_res.update(self.get_info_BO(tag='init' + str(polish_step) + '_'))
                self.BO.run_optimization(max_iter = new_iter, eps = 0, max_time = self.max_time_bo)
                if(save_extra):
                    dico_res.update(self.get_info_BO(tag='explor' + str(polish_step) + '_'))
                if nb_exploit>0:
                    self.set_up_BO_exploit()
                    logger.info('Exploitation (i.e. ucb with k=0) for {}'.format(nb_exploit))
                    self.BO.run_optimization(nb_exploit, max_time=self.max_time_bo)
                    self.update_BO_time()
                    if(save_extra):
                        dico_res.update(self.get_info_BO(tag='exploit' + str(polish_step) + '_'))

                                
                X_keep_track = np.r_[(X_keep_track, np.c_[(self.BO.X, polish_step*np.ones(len(self.BO.X)))])]
                if(self.BO.Y.shape[1] == (Y_keep_track.shape[1]-1)):
                    Y_keep_track = np.r_[(Y_keep_track, np.c_[(self.BO.Y, polish_step*np.ones(len(self.BO.Y)))])]
                else:
                    to_add_tmp = [Y_keep_track] * (Y_keep_track.shape[1]-1)
                    to_add_tmp.append(polish_step*np.ones(len(self.BO.Y)))
                    
                    #Y_keep_track = np.r_[(Y_keep_track,  np.c_[to_add_tmp])]
            
            dico_res.update(self.get_info_BO(tag=''))            
            dico_res.update({'params_BO_names': self.BO.model.model.parameter_names(), 'p_tgt':self.p_tgt, 
                        'f_tgt':self.f_tgt, 'nb_output':self.nb_output, 'abs_diff':1-
                        dico_res['test'], 'time_allbo':self.time_all_bo, 'time_fit':self.time_fit_bo, 
                        'time_suggest':self.time_suggest_bo, 'fid_zero_field':self.fid_zero,'phi_0':   
                        Qobj2array(self.phi_0), 'phi_tgt':Qobj2array(self.phi_tgt), 'polish_step':polish_step, 
                        'nb_polish':nb_polish, 'nb_more':nb_more, 'nb_to_keep':nb_to_keep})
                         
        cum_time = time.time() - time_start        
        dico_res.update({'time_all':cum_time,'x_tgt':self.x_tgt, 'call_f':self.call_f, 
                         'call_f_single':self.call_f_single,'call_f_test': self.call_f_test})
        return dico_res 

    @classmethod
    def _process_collection_res(cls, collection_res, **xargs):
        """ Process the collected res. As it is custom to the type of results
        it has to be implemented in the subclasses"""
        processed = {}
        for k, v in collection_res.items():
            #test = _stats_one_field('test', v)
            test_exp = _stats_one_field('test_exp', v)
            test = _stats_one_field('test', v)
            abs_diff = _stats_one_field('abs_diff', v)
            call_f = _stats_one_field('call_f', v)
            time_elapsed = _stats_one_field('time_elapsed', v)
            time_fit = _stats_one_field('time_fit', v)
            time_suggest = _stats_one_field('time_suggest', v)
            best_res = _get_some_res('test', v, np.max)
            median_res = _get_some_res('test', v, np.median)
            processed.update({k:{'test_exp': test_exp, 'test':test, 
                'abs_diff':abs_diff, 'time_elapsed':time_elapsed,'call_f':call_f, 
                'best_res': best_res, 'median_res':median_res, 'time_fit':time_fit,
                'time_suggest':time_suggest}})
        return processed


    def update_BO_time(self):
        self.time_all_bo += self.BO.cum_time
        self.time_fit_bo += self.BO.cum_time_fit
        self.time_suggest_bo += self.BO.cum_time_suggest
        self.max_time_bo -= self.BO.cum_time
            

    def set_up_BO_exploit(self):
        self.BO.acquisition_type = self.BO.acquisition_type.replace('EI', 'LCB')
        self.BO.acquisition_weight = 0.000001
        self.BO.kwargs['acquisition_weight'] = 0.000001
        self.BO.acquisition = self.BO._acquisition_chooser()
        self.BO.evaluator = self.BO._evaluator_chooser()    
        

    def set_up_BO(self, optim_config, nb_restrict_data = None, restrict_domain = False, adapt_shots=1):
        """ set up the BO object. At the end the model is initiated
        Restrict domain // Implement rules for the adaptation of the number of shots // 
        Deal with the fixing of the constraints and the forcing the use of grad
        
        adapt_shots:
            <int> N: multi
            <str> 'Strat1': 
        """
        MAX_INCREASE = 5000
        if nb_restrict_data is not None:
            (_,_), (x_exp, _) = self.BO.get_best()
            X_to_keep, Y_to_keep = filter_X(self.BO.X, self.BO.Y, x_exp, nb_restrict_data)
        else:
            if(hasattr(self, 'BO')):
                X_to_keep = self.BO.X 
                Y_to_keep = self.BO.Y
            else:
                X_to_keep = None
                Y_to_keep = None

        if restrict_domain:
            self.domain = [(mi, ma) for mi, ma in zip(np.min(X_to_keep, 0), np.max(X_to_keep, 0))]

        if type(adapt_shots) == int: #base case
            if adapt_shots == 1:
                optim_config.update({'X_init':X_to_keep, 'Y_init':Y_to_keep})
            else:
                self.n_meas *= adapt_shots
                optim_config.update({'X_init':X_to_keep, 'Y_init':None})

            self.bo_args, f_BO = self.get_BO_args(optim_config)
            self.BO = GPyOpt.methods.BayesianOptimization(f_BO, **self.bo_args)            
            force_grad_acq = optim_config.get('force_grad_acq',False)
            if force_grad_acq: self.BO.acquisition.analytical_gradient_acq = True
            hp_restart = self.bo_args['hp_restart']
            if hp_restart: 
                self.BO._create_model(self.BO.normalization_type)
                self.restore_hp_values() ### What about first step
            if self.hp_constrains is not None: 
                self.BO._create_model(self.BO.normalization_type)
                self.constrain_hp(self.hp_constrains)


        else:
            model_mean, model_std = self.BO.acquisition._compute_acq_splitted(X_to_keep)
            model_std_avg = np.median(model_std)
            model_mean_disp = np.std(model_mean)
            try:
                alpha = np.sqrt(np.median(model_mean, 0) * (1 - np.median(model_mean, 0))/self.n_meas) 
            except:
                alpha = 0.5
            std_data = np.average(np.std(Y_to_keep, 0))/self.n_meas
            
            if adapt_shots == 'strat1':
                coeff = int(np.clip(np.square(alpha/std_data) , 1, MAX_INCREASE))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)
                    
            elif adapt_shots == 'strat2':
                coeff = int(np.clip(np.square(model_mean_disp/model_std_avg), 1,MAX_INCREASE))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)

            elif adapt_shots == 'strat3': 
                coeff =  int(np.clip(np.square(model_mean_disp/model_std_avg), 1, 100))
                self.set_up_BO(optim_config, nb_restrict_data = nb_restrict_data, restrict_domain = restrict_domain, adapt_shots=coeff)
                self.set_up_BO(optim_config, nb_to_keep = None, restrict_domain = False, adapt_shots='strat2')

            else:
                raise NotImplementedError()
                
                





    def get_BO_args(self, optim_config):
        nb_init = optim_config['nb_init']
        nb_iter = optim_config['nb_iter']
        type_acq = optim_config['type_acq']
        is_acq_target = type_acq.find('target') > 0
        logger.info('type_acq: {}'.format(type_acq))
        type_lik = optim_config['type_lik']
        mo = optim_config.get('mo')
        nb_anchors = optim_config.get('nb_anchors', 15)
        optim_num_samples= optim_config.get('optim_num_samples', 10000)
        acq_weight = optim_config.get('acq_weight', 4)
        acquisition_weight_lindec = optim_config.get('acquisition_weight_lindec', True)
        model_update_interval= optim_config.get('model_update_interval', 1)
        hp_update_interval= optim_config.get('hp_update_interval', 1)
        num_cores = optim_config.get('num_cores', 1)
        max_iters = optim_config.get('max_iters', 1000) # used when updating the hyper-parameters
        optimize_restarts = optim_config.get('optimize_restarts',5) # for the hyperparameters fitting
        self.hp_constrains = optim_config.get('hp_constrains', None)
        switch_to_gauss = optim_config.get('switch_to_gauss', None)
        n_meas = self.n_meas
        
        # some redundancy here 
        aggregate = optim_config.get('aggregate', 'no')
        aggregate = 'fid' if(aggregate == True) else aggregate
        aggregate = 'no' if(aggregate == False) else aggregate
        
        if (switch_to_gauss is not None) and (n_meas > switch_to_gauss):
            #overwride parameters
            type_lik = None
            aggregate = 'fid'
            is_acq_target = False
            type_acq = type_acq.split('_')[0]
            mo=None
            logger.warning('nb shots: {}, Swith to Gaussian likelihood with fidelity aggregated output'.format(n_meas))

        if(optim_config.get('type_optim', 'BO') == 'BO_NOOPTIM'):
            nb_init_bo = nb_init + nb_iter
            nb_iter_bo = 0
            max_time_bo = optim_config.get('max_time', 23.5*3600)
        else:
            nb_init_bo = nb_init
            nb_iter_bo = nb_iter
            max_time_bo = optim_config.get('max_time', 23.5*3600)
            
        f_fact = self.n_meas if type_lik == 'binomial' else 1
        if is_acq_target:
            f_wrap = lambda x: self.f(x, N=self.n_meas, aggregate=aggregate) 
            self.warp_f = lambda x:x
        else: 
            f_wrap = lambda x: 1-self.f(x, N=self.n_meas, aggregate=aggregate)
            self.warp_f = lambda x:1-x
        f_BO = lambda x: f_fact * f_wrap(x)
        
        
        if optim_config.get('X_init', None) is not None:
            X_init = optim_config['X_init']            
        else:
            X_init = np.transpose([rdm.uniform(*d, nb_init_bo) for d in self.domain]) 

        if optim_config.get('Y_init', None) is not None:
            Y_init = optim_config['Y_init']
        else:
            Y_init = f_BO(X_init)
        bounds_bo = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(self.domain)]

        if 'polish' not in optim_config:
            nb_polish = 0
            nb_to_keep = 0
            nb_more = 0
            hp_restart = False
            nb_iter_polish = 0
        else:
            polish_dico = optim_config['polish']
            nb_polish = polish_dico.get('nb_polish',0)
            nb_to_keep = make_iter_if_not_iter(polish_dico.get('nb_to_keep', nb_init), nb_polish)
            nb_more = make_iter_if_not_iter(polish_dico.get('nb_more', 1), nb_polish)
            nb_iter_polish = make_iter_if_not_iter(polish_dico.get('nb_iter', nb_iter_bo), nb_polish)
            hp_restart = polish_dico.get('hp_restart', False)
        nb_exploit = optim_config.get('exploitation_steps',0)
        
        bo_args = {'model_update_interval':model_update_interval, 'X':X_init,
                'Y':Y_init, 'domain': bounds_bo, 'optim_num_anchor':nb_anchors, 
                'optim_num_samples':optim_num_samples, 'num_cores':num_cores, 
                'max_iters':max_iters, 'optimize_restarts':optimize_restarts,
                'hp_update_interval':hp_update_interval, 'nb_iter_bo':nb_iter_bo,
                'max_time_bo':max_time_bo, 'nb_polish':nb_polish, 'nb_to_keep':nb_to_keep,
                'nb_more':nb_more, 'nb_exploit':nb_exploit, 'hp_restart':hp_restart, 
                'nb_iter_polish':nb_iter_polish}
        
        if type_acq == 'EI':
            bo_args.update({'acquisition_type':'EI'})
        elif type_acq == 'LCB':
            bo_args.update({'acquisition_type':'LCB', 'acquisition_weight':acq_weight, 
                            'acquisition_weight_lindec':acquisition_weight_lindec})
        elif type_acq == 'EI_target':
            bo_args.update({'acquisition_type':'EI_target', 'acquisition_ftarget': self.p_tgt})
        
        elif type_acq == 'LCB_target_oneq':
            acq_nbq = optim_config['acq_nbq']
            bo_args.update({'acquisition_type':'LCB_oneq', 'acquisition_ftarget': self.p_tgt,
                            'acquisition_weight':acq_weight,'acquisition_weight_lindec':acquisition_weight_lindec,'acq_nbq':acq_nbq})
        
        elif type_acq == 'LCB_target':
            bo_args.update({'acquisition_type':'LCB_target','acquisition_weight':acq_weight, 
                            'acquisition_weight_lindec':acquisition_weight_lindec, 'acquisition_ftarget': self.p_tgt})
        else:
            logger.error('type_acq {} not recognized'.format(type_acq))
  
        if type_lik == 'binomial':
            logger.info('FTARGET is used by BO')
            bo_args.update({ 'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                'likelihood':'Binomial_' + str(self.n_meas), 'normalize_Y':False})
            if type_acq != 'LCB_target_oneq':
                bo_args.update({'acquisition_ftarget':self.f_tgt})
        # multioutput
        if mo is not None:
            bo_args.update({'mo':mo})
            self.nb_output = mo['output_dim']
        else:
            self.nb_output = 1
        self.type_likelihood = type_lik
        
        return bo_args, f_BO
    
    def get_info_BO(self, i_beg = None, i_end = None, tag=''):
        """ Extract domain""" 
        res = {}
        (x_seen, _), (x_exp,_) = self.BO.get_best()
        x = self.BO.X
        y = self.BO.Y
        if i_beg is not None:
            x, y = x[i_beg:], y[i_beg:]
        if i_end is not None:
            x, y = x[:-i_end], y[:-i_end]
        res[tag + 'bo_x'] = x
        res[tag + 'bo_y'] = y
        bo_acq, bo_acq_std = self.BO.acquisition._compute_acq_splitted(x)
        if self.BO.normalize_Y:
            norm_args = self.BO.norm_args
            #normalization_type = self.normalization_type
            norm_mean = norm_args['mean']
            norm_std = norm_args['std']
            bo_acq = norm_std * bo_acq + norm_mean
            bo_acq_std = norm_std * bo_acq_std
        else:
            norm_mean = 0.
            norm_std = 1.
        bo_acq = self.warp_f(bo_acq)
        res[tag + 'norm_mean'] = norm_mean
        res[tag + 'norm_std'] = norm_std
        res[tag + 'bo_acq'] = bo_acq
        res[tag + 'bo_acq_std'] = bo_acq_std
        res[tag + 'bo_tgt'] = self.f_test(x)
        res[tag + 'bo_args'] = np.copy(self.BO.model.model.param_array), 
        res[tag + 'bo_args_names'] = self.BO.model.model.parameter_names()
        res[tag + 'domain'] = np.array([d['domain'] for d in self.BO.domain])
        res[tag + 'x_exp'] = x_exp
        res[tag + 'x_seen'] = x_seen
        res[tag + 'test'] = self.f_test(x_exp)
        res[tag + 'nb_s'] = self.n_meas
        res[tag + 'call_f'] = self.call_f
        res[tag + 'call_f_single'] = self.call_f_single
        try:
            res[tag + 'alpha'] = np.sqrt(np.average(bo_acq, 0) * (1 - np.average(bo_acq, 0))/self.n_meas) 
        except:
            res[tag + 'alpha'] = 0.5
        
        return res
    
    def constrain_hp(self, constrains_dico):
        for k, v in constrains_dico.items():
            str_param = '.*'+k
            if np.any([k in name for name in self.BO.model.model.parameter_names()]):
                if v == 'positive':
                    self.BO.model.model[str_param].constrain_positive()
                elif len(v) == 2:
#                    v_min, v_max = v[0], v[1]
#                    if('variance' in k) and self.BO.normalize_Y:
#                        scale = np.square(self.BO.args_norm['std'])                        
#                    
                    self.BO.model.model[str_param].constrain_bounded(v[0], v[1], warning = False)
                else:
                    self.BO.model.model[str_param] = v
                
    def save_hp_values(self):
        self.hp_vals = {name:[val] for name, val in zip(self.BO.model.model.parameter_names(),
                        self.BO.model.model.param_array)}
        
    def restore_hp_values(self):
        if(hasattr(self, 'hp_vals')):
            self.constrain_hp(self.hp_vals)
        else:
            logger.warning("no hp_vals to restore")
            
def filter_X(X, Y,  x_best, nb):
    d_to_best = dist_to(X, x_best)
    return X[np.argsort(d_to_best)[:nb]], Y[np.argsort(d_to_best)[:nb]]

#=======================================#
# HELPER FUNCTIONS
#=======================================#
def get_proba_after_evol(U, phi_0, list_e, n_ro):
     phi_f = U * phi_0
     final_expect = [real_with_test(e.matrix_element(phi_f, phi_f)) for e in list_e]
     proba = np.array([(1 + e)/2 for e in final_expect])
     proba = corrupt_readout_qubit(proba, n_ro)
     assert np.any(proba < 1 + 1e-5), "proba > 1: {}".format(proba)
     assert np.any(proba > -1e-5), "proba > 1: {}".format(proba)
     proba = np.clip(proba, 0, 1)
     return proba
 
def get_measurement_from_proba(proba, N, aggregate, p_tgt=None):
    if (N == np.inf): 
        res = proba 
        n_call = 1
    else:
        res = rdm.binomial(N, proba) / N
        n_call = len(res)*N
    if aggregate == 'fid':
        res = 1/8 * (1 + np.dot(2*p_tgt-1,2*res-1))
    elif aggregate == 'close':
        res = np.average(np.abs(p_tgt-res))
    return res, n_call

def make_iter_if_not_iter(x, nb_elements):
    if  hasattr(x, '__iter__'):
        assert len(x) == nb_elements, "pb len nb_more = {} while nb_polish is {} ".format(len(nb_elements))
        x_iter = x
    else:
        x_iter = [x] * nb_elements
    return x_iter


def Qobj2array(qobj):
    if hasattr(qobj,'full'):
        return qobj.full() 
    else:
        return qobj


def ansatz_H(angle, n_gate= 0):
    """ exactly H if angle = pi/2"""
    gate = ry(angle) * Z
    if n_gate > 0:
        gate *= ansatz_N(n_gate)
    return gate

def ansatz_X(angle, n_gate= 0):
    """ -j X if angle = pi"""
    gate = rx(angle)
    if n_gate > 0:
        gate *= ansatz_N(n_gate)
    return gate

def ansatz_I(n_gate = 0):
    gate = I
    if n_gate > 0:
        gate *= ansatz_N(n_gate)
    return gate
        

def layer_N(nb_qubits, n_gate = 0):
    """one layer of nb _qubits independent I or Noise gates"""
    return tensor(*[ansatz_I(n_gate)for _ in range(nb_qubits)])
    
    
def ansatz_N(epsilon):
    """ noisy gate: G = exp(-i n.sigma) """
    ampl = rdm.normal(scale=epsilon*np.pi)
    direction = rdm.normal(size=3)
    direction /= np.sqrt(np.dot(direction, direction))
    H = -1.j * ampl * (direction[0] * X + direction[1] * Y + direction[2] * Z)
    return H.expm()
    
def get_UGHZ(x, n_gate = 0):
    """ parametrized (6 parameters) circuitry which can create a c. GHZ 
    state i.e. |000> - |111>"""
    g1 = tensor(ansatz_H(x[0], n_gate), ansatz_H(x[1], n_gate), ansatz_X(x[2], n_gate))
    g2 = cnot(3, 1,2)
    g3 = cnot(3, 0,2)
    g4 = tensor(ansatz_H(x[3], n_gate), ansatz_H(x[4], n_gate), ansatz_H(x[5], n_gate))
    g5 = layer_N(3, n_gate)
    return g5 * g4 * g3 * g2 * g1

def get_HZY(theta):
    """ setup a parametrized Hamiltonian """
    alpha = theta[0]
    beta = theta[1]
    H = alpha * beta * Z + Y * alpha * np.sqrt(1. - np.square(beta))
    return H

def get_HZY_2(theta):
    """ setup a parametrized Hamiltonian - different parametrization as the one
    above - but in essence same Hamiltonian"""        
    alpha = theta[0]
    beta = theta[1]
    H = alpha * Z + Y * beta
    return H

def get_UCPRY(x, n_gate=0):
    """ setup a parametrized Unitary in the form of two parametrized gates """
    n1 = layer_N(nb_qubits=2, n_gate = n_gate)
    g1 = tensor(ry(x[0]), I)
    n2 = layer_N(nb_qubits=2, n_gate = n_gate) 
    g2 = controlled_gate(ry(x[1]))
    n3 = layer_N(nb_qubits=2, n_gate = n_gate) 
    return n3 * g2 *n2 * g1 * n1

    'domain'
def real_with_test(x):
    assert np.allclose(np.imag(x), 0.), ""
    return np.real(x)

def _get_some_res(field, list_res, criterion = np.max):
    """ Pick some res amongst a list of res according to some criterion"""
    field_values = np.array([res.get(field) for res in list_res])
    field_best = criterion(field_values)
    return [list_res[n] for n, f in enumerate(field_values) if f == field_best]

def _stats_one_field(field, list_res, dico_output = False):
    field_values = np.array([res.get(field) for res in list_res])
    mask_none = np.array([f is not None for f in field_values])
    f = field_values[mask_none]
    N = len(f)
    if(len(f) > 0):
        field_avg = np.average(f)
        field_std = np.std(f)
        field_min = np.min(f)
        field_max = np.max(f)
        field_median = np.median(f)
    else:
        field_avg = np.nan
        field_std = np.nan
        field_min = np.nan
        field_max = np.nan
        field_median = np.nan
    if dico_output:
        res = {'avg':field_avg, 'median': field_median, 'std': field_std, 'min': field_min, 'max':field_max, 'N':N}
    else:
        res = [field_avg, field_median, field_std, field_min, field_max, N]
    return res

def probit(p):
    return np.clip(np.sqrt(2) * erfinv(2 * p -1), -2.3263478740408408, 2.3263478740408408)

def closest_to(X, x_tgt):
    return np.argsort(dist_to(X, x_tgt))

def dist_to(X, x_tgt):
    return np.linalg.norm(X-x_tgt, axis = 1) / np.shape(X)[1]

def corrupt_readout_qubit(proba, n_ro=0):
    """ classical readout noise with noise[i] = 1-p(i|i) with i in {0,1}"""
    proba_corrupted = np.array(proba)
    if n_ro>0:
        proba_corrupted = proba_corrupted * (1-n_ro) + (1-proba_corrupted) * n_ro
    return proba_corrupted 
    


if __name__ == '__main__':
    # 3 BEHAVIORS DEPENDING ON THE FIRST PARAMETER:
    #   + "gen_configs" generate config files from a metaconfig file
    #   + "gen_configs_custom" generate config files from a metaconfig file (with extra_processing)
    #   + "run_one_config" run cspinoptim based on a config file
    
    if(len(sys.argv) > 5 or len(sys.argv) < 3):
        logger.error("Wrong number of args")
    else:
        type_task = sys.argv[1]
        file_input = sys.argv[2]

        if(type_task == "gen_configs"):
            output_f = str(sys.argv[3]) if(len(sys.argv) == 4) else 'Config'
            update_rules = str(sys.argv[4]) if(len(sys.argv) == 5) else True
            BatchFS.parse_and_save_meta_config(
                    file_input, output_folder = output_f, update_rules = True)

        elif(type_task == "run_one_config"):
            batch = BatchFS(file_input)
            batch.run_procedures(save_freq = 1)

        elif(type_task == "run_meta_config"):
            update_rules = str(sys.argv[3]) if(len(sys.argv) == 4) else True
            batch = BatchFS.from_meta_config(file_input, update_rules = update_rules)
            batch.run_procedures(save_freq = 1)

        else:
            logger.error("first argument not recognized")
   
    # Just for testing purposes
    testing = False 
    if(testing):
        BatchFS.parse_and_save_meta_config(input_file = 'Inputs/model_5_comp3_gaussianpolish_noisy_5s.txt', output_folder = '_tmp/_configs/_tmp', update_rules = True)
        #batch = BatchFS(['_tmp/_configs/_mo5gradient/config_res'+str(i)+'.txt' for i in range(100)])
        batch = BatchFS('_tmp/_configs/_model_4_polish_v3/config_res1.txt')
        batch.run_procedures(save_freq = 1)


        #pulse_grape = np.array([[-1.50799058, -1.76929128, -4.21880315,  0.5965928 ], [-0.56623617,  2.2411309 ,  5.        , -2.8472072 ]])        
        #keys_collect = [['config', 'model', '_FLAG'], ['config', 'optim', '_FLAG']]
        #list_output = ['avg', 'median', 'std', 'min', 'max']
        #c_perfect = BatchFS.collect_res(key_path = keys_collect, folderName = '_tmp/_Output/_mo5_bfgs/', allPrefix = 'res',replace_func=ut.workaroundQObj)
        #x_exp = np.array([c['x_exp'] for c in c_perfect['infs_grad_100']])
        #x_exp_pi = np.round(x_exp/ (np.pi/2)) 
        #x_exp_pi[np.array([np.all([ (xx>=-0.5) and (xx<=2.1) for xx in x]) for x in x_exp_pi])]
        #diff_x = x_exp - x_exp_pi * np.pi/2
        #test_exp = np.array([c['test'] for c in c_perfect['infs_grad_100']])
        perfect_x = np.pi/2 * np.array([1., 3., 2., 3., 1., 3.])
        inperfect_x = perfect_x + 0.5 * np.pi/2 * np.random.normal(size=6) 
        inperfect_x_small = perfect_x + 0.1 * np.pi/2 * np.random.normal(size=6)
        optim = np.array([4.72841866, 4.71646124, 6.28318531, 1.56676997, 1.56157624,4.71475796])
        
        fid_res = [batch.f(perfect_x, aggregate='fid', N=np.inf, n_gate=0.01) for _ in range(10000)]
        fid_res_inper = [batch.f(inperfect_x, aggregate='fid', N=np.inf, n_gate=0.01) for _ in range(1000)]
        fid_res_inper_small = [batch.f(inperfect_x_small, aggregate='fid', N=np.inf, n_gate=0.01) for _ in range(1000)]      
        fid_res_optim = [batch.f(optim, aggregate='fid', N=np.inf, n_gate=0.01) for _ in range(10000)]
        
        l1 = "F=" + str(np.round(batch.f_test(perfect_x) * 100,0)) + "%"
        l2 = "F=" + str(np.round(batch.f_test(inperfect_x) * 100,0)) + "%"
        l3 = "F=" + str(np.round(batch.f_test(inperfect_x_small) * 100,0)) + "%"
        import matplotlib
        import matplotlib.pylab as plt
        matplotlib.rcParams['text.usetex'] = False
        matplotlib.rcParams['text.latex.unicode'] = True
        plt.hist(np.squeeze(fid_res), label = r'$F=100\%$', bins=3, rwidth=100)
        plt.hist(np.squeeze(fid_res_inper_small), label =r'$F=97\%$')
        plt.hist(np.squeeze(fid_res_inper), label = r'$F=73\%$')
        plt.legend()
        plt.title(r'$\varepsilon_n=0.1\%$')
        #plt.savefig("noise01pc.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
        
        print(np.std(np.squeeze(fid_res)))
        print(np.std(np.squeeze(fid_res_inper)))
        print(np.std(np.squeeze(fid_res_inper_small)))
        
        print(np.average(np.squeeze(fid_res)))
        print(np.average(np.squeeze(fid_res_inper)))
        print(np.average(np.squeeze(fid_res_inper_small)))        
        print(np.average(np.squeeze(fid_res_optim)))  
        

        ideal = np.array([4.71238898, 4.71238898, 6.28318531, 1.57079633, 1.57079633,
       4.71238898])
        
        batch.f(optim, aggregate='fid', N=1000000000000000)
        batch.f(ideal, aggregate='fid', N=100000000000)