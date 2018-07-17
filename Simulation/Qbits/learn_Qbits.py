#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
#
#============================================================================== 
import sys
import pdb
import numpy as np
import matplotlib.pylab as plt
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import TwoLevels as tl
    from QuantumSimulation.Utility.Optim import pFunc_base, Learner, Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 
    
else:
    from ...Utility import Helper as ut
    from ...Utility.Optim import pFunc_base, Batch, Learner
    from ...Utility.Optim.RandomGenerator import RandomGenerator 
    from ...Utility.Optim.MP import MPCapability 
    from ...ToyModels import TwoLevels as tl

class learnerQB(Batch.BatchParametrizedControler):
    """
    Should cope with:
        + management of randomGen and mp
        + testing
        + dispatching i.e. more flexibility in creating the controler
    TODO: noise in the test through seed_noise = [(), ..., ()]
    """
    def run_one_procedure(self, config):
        """ 
        """
        self.random_gen = RandomGenerator.init_random_generator(config['_RDM_SEED'])
        self.mp = MPCapability.init_mp(config.get(config['_MP_FLAG']))
        dico_update ={'rdm_obj': self.random_gen, 'mp_obj':self.mp}
        
        model_dico = config.get('model_dico', {})
        if (model_dico not in [None, {}]):
            track_learning = model_dico.pop('track_learning', False)
            model_dico.update(dico_update)
            self._build_control(model_dico)
            model = tl.Qbits(**model_dico)
            
            optim_dico = config['optim_dico']
            optim_dico.update(dico_update)
            optim = Learner.learner_Opt(model = model, **optim_dico)        
            res = optim(track_learning = track_learning)
            control_fun = model.control_fun.clone()
            control_fun.theta = res['params']
            res['func'] = repr(control_fun)
            res['fun_name'] = model_dico['fom']
            del model_dico['rdm_obj']
            del model_dico['mp_obj']
            del optim_dico['rdm_obj']
            del optim_dico['mp_obj']
        else:
            model_dico = {}
            res={'params':[]}
            
        testing_dico = config.get('testing_dico')
        #Testing_dico contains ONLY updates to apply to model_dico
        if(testing_dico is not None):
            testing_updated = ut.merge_dico(model_dico, testing_dico, update_type = 0, copy = True)
            testing_updated.update(dico_update)
            self._build_control(testing_updated)
            model_test = tl.Qbits(**testing_updated)
            model_test.control_fun = model.control_fun
            optim_params = testing_updated.pop('params_force', None)
            if(optim_params is None):
                optim_params = res['params']
            res_test = model_test(optim_params, trunc_res = False)
            res['test_fom'] = res_test
            res['test_fom_names'] = testing_dico['fom']
            res['test_params'] = optim_params
            del testing_updated['rdm_obj']
            del testing_updated['mp_obj']
            config['testing_dico'] = testing_updated
        return res
    
    


    # ----------------------------------------------------------------------- #
    # Reading results aera:
    # ----------------------------------------------------------------------- #
    @classmethod
    def _list_res_get_ideal_evol(cls, list_res):
        name_fom, ideal_nev_foms, dist_params = type(cls).list_res_ideal_learning(list_res)
        ideal_nev_foms_stats = [ut.merge_and_stats_TS(one_fom) for one_fom in ideal_nev_foms]
        dist_params_stats = ut.merge_and_stats_TS(dist_params)
        dico_res = {'ev_distparams':dist_params_stats,'ev_idealfom':ideal_nev_foms_stats[0]}
        
        for n, one_fom_stats in enumerate(ideal_nev_foms_stats[1:]):
            dico_res['ev_ideal'+name_fom[n+1]] = one_fom_stats   
        return dico_res
        
        
    @classmethod
    def ideal_learning(cls, run):
        """ Compute the fom (under testing conditions) for the different functions 
        found along optimization and also the distance between parameters
        should return (names_fom, res, res_params_dist)
        """
        testing_dico = run['testing_dico']
        names_fom = testing_dico['fom']
        model_tmp = tl.Qbits(**testing_dico)
        try:
            evol_params = run['extra_history_nev_params']
            res = [[p[0], model_tmp(p[1])] for p in evol_params]
            res_params = [[p[0], p[1]] for p in evol_params]
            res_params_dist = [np.array([par[0], np.square(par[1] - res_params[n][1])]) for n, par in enumerate(res_params[1:])]
        except:
            print("can't find extra_history_params_fun keys in the res.. no ideal learning possible")
            res = []
        return names_fom, res, res_params_dist
        
    @classmethod
    def list_res_ideal_learning(cls, list_res):
        pdb.set_trace()
        tmp = [type(cls).ideal_learning(res) for res in list_res]
        nb_run = len(list_res)
        nb_fom = len(tmp[0][1])
        res = [[np.array([tmp[r][:,0], tmp[r][:,1+n]]) for r in range(nb_run)] for n in range(nb_fom)]
        return res


    @classmethod
    def one_res_rebuild(cls, res):
        """ Pass it a res object and it should rerun the model"""
        pass

    @classmethod
    def runSimul(cls, dico_simul, params):
        """ from a dico containing the parameters of the simulations and a control 
            function get the results of the simulation"""
        model_tmp = tl.Qbits(**dico_simul)
        res = model_tmp(params)
        return res
        
    

#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    # Testing has been spin off to Test/BH1D
    pass


        