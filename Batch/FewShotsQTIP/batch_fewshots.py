# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import logging 
logger = logging.getLogger(__name__)

import sys
import numpy as np
import pdb
from scipy.special import erfinv
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
from qutip import sigmax, sigmaz
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
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
        n_ts = model_config['n_ts']
        T = model_config['T']
        
        # May need to make the choice of init and tgt states more flexible
        # init = model_config.get('init','eigen0')
        # tgt = model_config.get('tgt','eigen0')
        if(model == 1):
            #define an Hamiltonian H(t) = - Sz - hx(t) Sx
            H_d = -sigmaz()
            H_c = [-sigmax()]
            H_i = - sigmaz() + 2 * sigmax()
            H_f = - sigmaz() - 2 * sigmax()
            n_ctrls = 1
            phi_0 = H_i.eigenstates(eigvals = 1)[1][0]
            phi_tgt = H_f.eigenstates(eigvals = 1)[1][0]

            logger.info("Creating and configuring control optimisation objects")
            # Create the OptimConfig object
            cfg = optimconfig.OptimConfig()
            cfg.log_level = logging.INFO

            # Create the dynamics object
            dyn = dynamics.DynamicsUnitary(cfg)
            dyn.num_tslots = n_ts
            dyn.evo_time = T
            dyn.target = phi_tgt
            dyn.initial = phi_0
            dyn.drift_dyn_gen = H_d
            dyn.ctrl_dyn_gen = H_c
            dyn.params_lbnd = -4
            dyn.params_ubnd = 4
        else:
            raise NotImplementedError()

        # Define the figure of merit
        #noise output
        noise_type = model_config.get('noise_type')        
        noise_n_meas = model_config.get('noise_n_meas', 1)
        noise_mean = model_config.get('noise_mean', 1)
        noise_std = model_config.get('noise_std', 0)
        noise_input = model_config.get('noise_input', 0)
        noise_b_meas = model_config.get('noise_b_meas')

        # CUSTOM FIDELITY COMPUTER
        fidcomp = FidCompUnitNoisy(dyn)
        fidcomp.noise_type = noise_type
        fidcomp.noise_mean = noise_mean
        fidcomp.noise_std = noise_std
        fidcomp.noise_n_meas = noise_n_meas
        fidcomp.noise_b_meas = noise_b_meas
        dyn.fid_computer = fidcomp
        zero_amps = np.zeros([n_ts, n_ctrls])
        dyn.initialize_controls(zero_amps)
        self.fid_zero = dyn.fid_computer.get_fidelity_perfect()
        logger.info("With zero field fid (square): {}".format(model, self.fid_zero))        

        self.dyn = dyn
        self.n_meas = noise_n_meas
        self.n_ctrls = n_ctrls
        self.n_ts = n_ts
        self.p_tgt = self.dyn.fid_computer.get_ptarget()
        if self.dyn.fid_computer.noise_b_meas is None :
            self.meas_basis = None 
        else:
            self.meas_basis = [b.full() for b in self.dyn.fid_computer.noise_b_meas]
        logger.info("Optim model {0} with {1} params between {2}-{3} and T= {4}".format(model, 
                                        n_ts, dyn.params_lbnd, dyn.params_ubnd, T))
        logger.info("dynamics has been created with QuTip and saved self.dyn")
        
        #figure of merit
        self.call_f = 0
        self.call_f_test = 0
        #use during optimization
        def f(x, verbose = verbose, noise=noise_input):
            if noise>0:
                x_n = np.clip(x + np.random.norm(0.0, noise, np.shape(x)), dyn.params_lbnd, dyn.params_ubnd) 
            else:
                x_n = x
            if(x_n.shape[0] != n_ts):
                res = np.array([f(x_one, verbose,noise=0) for x_one in x_n])
            else:
                self.call_f += 1
                amps = np.reshape(x_n, (n_ts, n_ctrls))
                self.dyn.update_ctrl_amps(amps)
                res = self.dyn.fid_computer.get_noisy_fidelity()
                if(verbose):
                    res_perfect = self.dyn.fid_computer.get_fidelity_perfect()
                    print([res, res_perfect])
                    print(np.squeeze(self.dyn.ctrl_amps))
            return np.atleast_1d(res)

        #Use for testing the optimal pulse
        def f_test(x, verbose = verbose):
            if(x.shape[0] != n_ts):
                res = np.array([f_test(x_one) for x_one in x])
            else:
                self.call_f_test += 1
                amps = np.reshape(x, (n_ts, n_ctrls))
                self.dyn.update_ctrl_amps(amps)
                res = self.dyn.fid_computer.get_fidelity_perfect()
                if(verbose):
                    print(res)
                    print(np.squeeze(self.dyn.ctrl_amps))
            return res
        
        return f, f_test


    def run_one_procedure(self, config):
        """ implemention of what is a procedure.
        """
        model_config = config['model']
        optim_config = config['optim']
        if(model_config.get('debug', False) or optim_config.get('debug', False)):
            pdb.set_trace()
        f, f_test = self.setup_QTIP_model(model_config)
        #setting up the optimizer

        type_optim = optim_config.get('type_optim', 'BO')
        domain = (self.dyn.params_lbnd, self.dyn.params_ubnd)
        p_tgt = optim_config.get('p_tgt')
        if p_tgt is None:
            p_tgt = self.p_tgt
        else:
            logger.info('PTARGET has been retrieved from the config')
        f_tgt = probit(p_tgt)
        logger.info('p_tgt: {0}'.format(p_tgt))
        logger.info('f_tgt: {0}'.format(f_tgt))
        np.random.seed(config['_RDM_SEED'])
        #MLE
        if(type_optim == 'CRAB'):
            raise NotImplementedError()
        
        elif(type_optim == 'GRAPE'):
            not NotImplementedError()
        
        #Bayesian Optimization
        elif(type_optim == 'BO'):
            nb_init = optim_config['nb_init']
            type_acq = optim_config['type_acq']
            logger.info('type_acq: {}'.format(type_acq))
            nb_iter = optim_config['nb_iter']
            type_lik = optim_config['type_lik']
            mo = optim_config.get('mo')
            nb_anchors = optim_config.get('nb_anchors', 15)
            acq_weight = optim_config.get('acq_weight', 4)
            
            f_fact = self.n_meas if type_lik == 'binomial' else 1
            if type_acq.find('target') > 0:
                f_wrap = lambda x: f(x) 
            else: 
                f_wrap = lambda x: 1-f(x)
            f_BO = lambda x: f_fact * f_wrap(x)
            
            X_init = np.random.uniform(*domain, (nb_init, self.n_ts))    
            Y_init = f_BO(X_init)
            bounds_bo = [{'name': str(i), 'type': 'continuous', 'domain': domain} for i in range(self.n_ts)]
            
            if type_acq == 'EI':
                bo_args = {'acquisition_type':'EI', 'domain': bounds_bo, 
                           'optim_num_anchor':nb_anchors, 'optim_num_samples':10000} 
            elif type_acq == 'LCB':
                bo_args = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                           'optim_num_anchor':nb_anchors, 'optim_num_samples':10000, 
                            'acquisition_weight':acq_weight, 'acquisition_weight_lindec':True} 
            elif type_acq == 'EI_target':
                bo_args = {'acquisition_type':'EI_target', 'domain': bounds_bo, 
                           'optim_num_anchor':nb_anchors, 'optim_num_samples':10000,
                           'acquisition_ftarget': p_tgt} 
            elif type_acq == 'LCB_target':
                bo_args = {'acquisition_type':'LCB_target', 'domain': bounds_bo, 
                           'optim_num_anchor':nb_anchors, 'optim_num_samples':10000, 
                            'acquisition_weight':acq_weight, 'acquisition_weight_lindec':True,
                            'acquisition_ftarget': p_tgt} 
            else:
                logger.error('type_acq {} not recognized'.format(type_acq))
            
            if type_lik == 'binomial':
                logger.info('FTARGET is used by BO')
                bo_args.update({
                    'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                    'likelihood':'Binomial_' + str(self.n_meas), 'normalize_Y':False,
                    'acquisition_ftarget':f_tgt})

            # multioutput
            if mo is not None:
                bo_args.update({'mo':mo})
                nb_output = mo['output_dim']
            else:
                nb_output = 1

            bo_args['X'] = X_init
            bo_args['Y'] = Y_init
            bo_args['num_cores'] = optim_config.get('num_cores', 1)

            BO = GPyOpt.methods.BayesianOptimization(f_BO, **bo_args)
            BO.run_optimization(max_iter = nb_iter, eps = 0)
            (x_seen, y_seen), (x_exp,y_exp) = BO.get_best()
            test = f_test(x_exp)
            #test_exp = f_test(xy_exp[0])
            if(np.ndim(np.squeeze(p_tgt)) == 0):
                abs_diff = np.abs(np.squeeze(p_tgt) - test)
            else:
                abs_diff = 1 - test
            dico_res = {'test':test, 'params_BO': BO.model.model.param_array, 
                'params_BO_names': BO.model.model.parameter_names(), 
                'p_tgt':p_tgt, 'f_tgt':f_tgt, 'nb_output':nb_output, 
                'x':x_seen[0], 'x_exp':x_exp[0], 'abs_diff':abs_diff,
                'opt_pulse': np.squeeze(self.dyn.ctrl_amps),'call_f':self.call_f,
                'call_f_test': self.call_f_test,'fid_zero_field':self.fid_zero,
                'phi_0': self.dyn.initial, 'phi_tgt':self.dyn.target, 
                'time_all':BO.cum_time, 'time_fit':BO.cum_time_fit, 
                'time_suggest':BO.cum_time_suggest} 
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
            best_res = _get_some_res('test', v, np.max)
            median_res = _get_some_res('test', v, np.median)
            processed.update({k:{'test_exp': test_exp, 'test':test, 
                        'abs_diff':abs_diff, 'time_elapsed':time_elapsed, 
                        'call_f':call_f, 'best_res': best_res, 
                        'median_res':median_res}})
        return processed
            

#=======================================#
# HELPER FUNCTIONS
#=======================================#
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
    return np.sqrt(2) * erfinv(2 * p -1)


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
        BatchFS.parse_and_save_meta_config(input_file = 'Inputs/_test_mo.txt', output_folder = '_configs_mo', update_rules = True)
        batch = BatchFS('_configs/config_res0.txt')
        batch.run_procedures(save_freq = 1)