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
from qutip import sigmax, sigmaz, sigmay, mesolve, Qobj, Options, identity, tensor, basis, cnot
import qutip.logging_utils as logging
logger = logging.get_logger()
from scipy.special import erfinv
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
#import qutip.control.optimizer as optimizer
#import qutip.control.stats as stats
from fidcompnoisy import FidCompUnitNoisy

if __name__ == '__main__':
    sys.path.append('../../../QuantumSimulation')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
else:
    sys.path.append('/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/QuantumSimulation/')
    from QuantumSimulation.Utility.Optim.batch_base import BatchBase
    
class BatchFS(BatchBase):
    """Implement few shots simulations for batching.
    Provides different methods for optimization / estimation
            
    """

    def setup_QTIP_model(self, model_config):
        """ Setup the model in QuTip allowing to compute the dynamics and FoM
        """        
        model = model_config['model']
        verbose = model_config.get('verbose', False)
        
        #define an Hamiltonian H(t) = - Sz - hx(t) Sx
        if(model == 1):
            T = model_config['T']
            lbnd = -4
            ubnd = 4
            H_d = -sigmaz()
            H_c = [-sigmax()]
            H_i = - sigmaz() + 2 * sigmax()
            H_f = - sigmaz() - 2 * sigmax()
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
            def f(x, verbose = verbose, noise=noise_input):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(np.ndim(x_n)>1):
                    res = np.array([f(x_one, verbose,noise=0) for x_one in x_n])
                else:
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
        elif(model==2):

            self.n_params = 2
            self.domain = [(0, 2 * np.pi), (0,1)]
            options_evolve = Options(store_states=True)
            self.phi_0 = Qobj(np.array([1., 0.]))
            all_e = [sigmax(), sigmay(), sigmaz()]
            self.n_meas = model_config.get('noise_n_meas', 1)
            self.n_meas_index = model_config.get('n_meas_index')         
            noise_input = model_config.get('noise_input', 0)
            self.nb_output = 3 if self.n_meas_index is None else 1

            #gen random phi_target and p_tgt associated
            x_tgt = np.array([rdm.uniform(*d) for d in self.domain])
            H_tgt = get_HZY(x_tgt)
            evol_tgt = mesolve(H_tgt, self.phi_0, tlist = [0., 1.], e_ops=all_e, options = options_evolve)
            final_expect_tgt = [e[-1] for e in evol_tgt.expect]
            self.phi_tgt = evol_tgt.states[-1]
            self.p_tgt = np.array([(1 + e)/2 for e in final_expect_tgt])
            #logger.info
            
                
            def f(x, verbose = verbose, noise=noise_input, N = self.n_meas):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(x_n.shape[0] != self.n_params):
                    res = np.array([f(x_one, verbose,noise=0) for x_one in x_n])
                else:
                    self.call_f += 1
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
                    else:
                        res = rdm.binomial(N, proba) / N
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
            """ 2 qubits gate"""
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
            Sx = sigmax()
            Sy = sigmay()
            Sz = sigmaz()
            Si = 0.5*identity(2)
            H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz)) # Drift Hamiltonian
            H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)] # The (four) control Hamiltonians
            
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
            zero = basis(2, 0) 
            one = basis(2, 1)
            basis_init_default = [tensor(zero, zero), tensor(zero, one), tensor(one, zero), tensor(one, one)]
            basis_meas_default = [tensor(zero, zero), tensor(zero, one), tensor(one, one), tensor(one, zero)]
            noise_n_meas = model_config.get('noise_n_meas', 1)
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
            def f(x, verbose = verbose, noise=noise_input):
                x_n = np.clip(x + rdm.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) if noise>0 else x
                if(np.ndim(x_n)>1):
                    res = np.array([f(x_one, verbose,noise=0) for x_one in x_n])
                else:
                    self.call_f += 1
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
            ### OPTIMIZATION            
        else:
            raise NotImplementedError()
        #figure of merit
        self.call_f = 0
        self.call_f_test = 0
        self.fid_zero = None #we are not interested by this figure for this model
        self.f_tgt = None if (self.p_tgt is None) else probit(self.p_tgt)

        return f, f_test


    def run_one_procedure(self, config):
        """ implemention of what is a procedure.
        """
        model_config = config['model']
        optim_config = config['optim']
        if(model_config.get('debug', False) or optim_config.get('debug', False)):
            pdb.set_trace()
        np.random.seed(config['_RDM_SEED'])
        f, f_test = self.setup_QTIP_model(model_config)
        
        #setting up the optimizer
        type_optim = optim_config.get('type_optim', 'BO')
        logger.info('p_tgt: {0}'.format(self.p_tgt))
        logger.info('f_tgt: {0}'.format(self.f_tgt))
        nb_init = optim_config['nb_init']
        nb_iter = optim_config['nb_iter']
        nb_total = nb_init + nb_iter
        
        if(type_optim == 'RANDOM'):
            time_start = time.time()
            X_init = np.transpose([np.atleast_1d(np.random.uniform(*d, nb_total)) for d in self.domain])    
            Y_init = f(X_init)
            if(self.p_tgt is not None):
                assert len(self.p_tgt) == Y_init.shape[1]
                temp = np.abs(Y_init - np.repeat(np.squeeze(self.p_tgt)[np.newaxis, :], len(Y_init), 0))
            else:
                temp = Y_init
            index_best = np.argmin(np.average(temp, 1))
            x_seen = X_init[index_best]
            y_seen = Y_init[index_best]
            x_exp = x_seen
            y_exp = y_seen
            test = f_test(x_exp)
            if(np.ndim(np.squeeze(self.p_tgt)) == 0):
                abs_diff = np.abs(np.squeeze(self.p_tgt) - test)
            else:
                abs_diff = 1 - test
            cum_time = time.time() - time_start
            
            
            
            dico_res = {'test':test, 'p_tgt':self.p_tgt, 'f_tgt':self.f_tgt, 
                'nb_output':self.nb_output, 'x':x_seen[0], 'x_exp':x_exp[0], 
                'abs_diff':abs_diff,'call_f':self.call_f, 'call_f_test': self.call_f_test,
                'fid_zero_field':self.fid_zero, 'phi_0': Qobj2array(self.phi_0), 
                'phi_tgt':Qobj2array(self.phi_tgt), 'time_all':cum_time, 
                'time_fit':0, 'time_suggest':0} 

        if(type_optim == 'CRAB'):
            raise NotImplementedError()
        
        elif(type_optim == 'GRAPE'):
            not NotImplementedError()
        

        #Bayesian Optimization 2 flavors 'BO' and 'BO_NOOPTIM'
        # 'BO' classical bayesian optimization
        # 'BO_NOOPTIM' all the x are randomly generated and GP is fitted 
        #              x_best is decided based on this model
        elif 'BO' in type_optim: 
            type_acq = optim_config['type_acq']
            logger.info('type_acq: {}'.format(type_acq))
            type_lik = optim_config['type_lik']
            mo = optim_config.get('mo')
            nb_anchors = optim_config.get('nb_anchors', 15)
            optim_num_samples= optim_config.get('optim_num_samples', 10000)
            acq_weight = optim_config.get('acq_weight', 4)
            acquisition_weight_lindec = optim_config.get('acquisition_weight_lindec', True)
            model_update_interval= optim_config.get('model_update_interval', 1)
            num_cores = optim_config.get('num_cores', 1)
            max_iters = optim_config.get('max_iters', 1000) # used when updating the hyper-parameters
            optimize_restarts = optim_config.get('optimize_restarts',5)
            
            if(type_optim == 'BO_NOOPTIM'):
                nb_init_bo = nb_init + nb_iter
                nb_iter_bo = 0
                max_time_bo = 0
            else:
                nb_init_bo = nb_init
                nb_iter_bo = nb_iter
                max_time_bo = optim_config.get('max_time', 23.5*3600)
            f_fact = self.n_meas if type_lik == 'binomial' else 1
            if type_acq.find('target') > 0:
                f_wrap = lambda x: f(x) 
            else: 
                f_wrap = lambda x: 1-f(x)
            f_BO = lambda x: f_fact * f_wrap(x)
            
            
            
            X_init = np.transpose([rdm.uniform(*d, nb_init_bo) for d in self.domain]) 
            Y_init = f_BO(X_init)
            bounds_bo = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(self.domain)]

            bo_args = {'model_update_interval':model_update_interval, 'X':X_init,
                    'Y':Y_init, 'domain': bounds_bo, 'optim_num_anchor':nb_anchors, 
                    'optim_num_samples':optim_num_samples, 'num_cores':num_cores, 
                    'max_iters':max_iters, 'optimize_restarts':optimize_restarts}    
            
            if type_acq == 'EI':
                bo_args.update({'acquisition_type':'EI'})
            elif type_acq == 'LCB':
                bo_args.update({'acquisition_type':'LCB', 'acquisition_weight':acq_weight, 
                                'acquisition_weight_lindec':acquisition_weight_lindec})
            elif type_acq == 'EI_target':
                bo_args.update({'acquisition_type':'EI_target', 'acquisition_ftarget': self.p_tgt})
            elif type_acq == 'LCB_target':
                bo_args.update({'acquisition_type':'LCB_target','acquisition_weight':acq_weight, 
                                'acquisition_weight_lindec':acquisition_weight_lindec, 'acquisition_ftarget': self.p_tgt})
            else:
                logger.error('type_acq {} not recognized'.format(type_acq))
  
            if type_lik == 'binomial':
                logger.info('FTARGET is used by BO')
                bo_args.update({ 'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                    'likelihood':'Binomial_' + str(self.n_meas), 'normalize_Y':False,
                    'acquisition_ftarget':self.f_tgt})

            # multioutput
            if mo is not None:
                bo_args.update({'mo':mo})
                nb_output = mo['output_dim']
            else:
                nb_output = 1

            BO = GPyOpt.methods.BayesianOptimization(f_BO, **bo_args)
            BO.run_optimization(max_iter = nb_iter_bo, eps = 0, max_time = max_time_bo)
            (x_seen, y_seen), (x_exp,y_exp) = BO.get_best()
            test = f_test(x_exp)
            #test_exp = f_test(xy_exp[0])
            if(np.ndim(np.squeeze(self.p_tgt)) == 0):
                abs_diff = np.abs(np.squeeze(self.p_tgt) - test)
            else:
                abs_diff = 1 - test
            dico_res = {'test':test, 'params_BO': BO.model.model.param_array, 
                'params_BO_names': BO.model.model.parameter_names(), 
                'p_tgt':self.p_tgt, 'f_tgt':self.f_tgt, 'nb_output':nb_output, 
                'x':x_seen, 'x_exp':x_exp, 'abs_diff':abs_diff,
                'call_f':self.call_f, 'call_f_test': self.call_f_test,
                'fid_zero_field':self.fid_zero,'phi_0': Qobj2array(self.phi_0), 
                'phi_tgt':Qobj2array(self.phi_tgt), 'time_all':BO.cum_time, 
                'time_fit':BO.cum_time_fit, 'time_suggest':BO.cum_time_suggest} 
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
            

#=======================================#
# HELPER FUNCTIONS
#=======================================#
def Qobj2array(qobj):
    if hasattr(qobj,'full'):
        return qobj.full() 
    else:
        return qobj
    
def get_HZY(args):
    """ setup the parametrized Hamiltonian """
    args = args        
    alpha = args[0]
    beta = args[1]
    H = alpha * beta* sigmaz() + sigmay() * alpha * np.sqrt(1. - np.square(beta))
    return H

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
        BatchFS.parse_and_save_meta_config(input_file = 'Inputs/_test_mo_model_3_gaussian.txt', output_folder = '_configs_mo3', update_rules = True)
        batch = BatchFS('_configs_mo3/config_res2.txt')
        batch.run_procedures(save_freq = 1)
        pulse_grape = np.array([[-1.50799058, -1.76929128, -4.21880315,  0.5965928 ],
                                [-0.56623617,  2.2411309 ,  5.        , -2.8472072 ]])
        
