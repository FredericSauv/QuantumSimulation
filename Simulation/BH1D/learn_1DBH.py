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
import copy
import numpy as np
import matplotlib.pylab as plt
from functools import partial
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import pFunc_zoo, pFunc_base, Learner, Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 
    
else:
    from ...Utility import Helper as ut
    from ...Utility.Optim import pFunc_zoo, pFunc_base, Batch, Learner
    from ...Utility.Optim.RandomGenerator import RandomGenerator 
    from ...Utility.Optim.MP import MPCapability 
    from ...ToyModels import BH1D as bh1d

class learner1DBH(Batch.BatchParametrizedControler):
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
            model = bh1d.BH1D(**model_dico)
            
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
            #Options to specify how you want the testing done: **params_force** 
            # allow to use specific parameters values **testing_same_func**
            # force to use the same function as the one in the model (should it be a default, behavior)
            # **use_exp_params** (specific to BO2): use expected best params
            # rather than the seen one 
            testing_updated = ut.merge_dico(model_dico, testing_dico, update_type = 0, copy = True)
            testing_updated.update(dico_update)
            force_params = testing_updated.pop('testing_force_params', None)
            same_func = testing_updated.pop('testing_same_func', True)
            use_exp_params = testing_updated.pop('use_exp_params', 0)

            self._build_control(testing_updated)
            model_test = bh1d.BH1D(**testing_updated)
            
            if(same_func):
                model_test.control_fun = model.control_fun
            
            if(force_params is not None):
                res['test_params'] = force_params
            else:
                if ((use_exp_params == 1) and (res.get('params_exp') is not None)):
                    res['test_params'] = res['params_exp']
                elif ((use_exp_params == 2) and (res.get('params_exp') is not None)):    
                    res['test_params'] = res['params']
                    res['test_params_exp'] = res['params_exp']
                else:
                    res['test_params'] = res['params']
                    
            optim_params = res['test_params']   
            res_test = model_test(optim_params, trunc_res = False)
            res['test_fom'] = res_test
            res['test_fom_names'] = testing_dico['fom']

            if('test_params_exp' in res):
                optim_params_exp = res['test_params_exp']
                res_test_exp = model_test(optim_params_exp, trunc_res = False)
                res['test_fom_exp'] = res_test_exp
                res['test_fom_names_exp'] = testing_dico['fom']
            
            del testing_updated['rdm_obj']
            del testing_updated['mp_obj']
            config['testing_dico'] = testing_updated
        return res
    
    
    def _build_control(self, model_dico):
        """ if the control_object is a string evaluate it if not do nothing"""
        control = model_dico['control_obj']
        if(ut.is_str(control)):
            model_dico['control_obj'] = learner1DBH._build_control_from_string(control,
                      self.random_gen, model_dico)

# ---------------------------
#   COLLECTING RESULTS ## WORKAROUND BUGS ## SHOULDN'T BE USED
# ---------------------------
    @classmethod
    def eval_from_onefile_bug(cls, name):
        """ eval the first element of the first line of a file """
        res = ut.eval_from_file_supercustom(name, evfunc = pFunc_base.eval_with_pFunc)
        return res
    
    @classmethod
    def collect_and_process_res_bug(cls, key_path = [], nameFile = None, allPrefix = 'res_', 
                                folderName = None, printing = False, ideal_evol = False):
        collect = learner1DBH.collect_res_bug(key_path, nameFile, allPrefix, folderName)
        process = learner1DBH.process_collect_res(collect, printing, ideal_evol)
        
        return process
    
    @classmethod
    def read_res_bug(cls, nameFile = None, allPrefix = 'res_', folderName = None):
        """ Extract result(s) stored in a (several) txt file (s) and return them  
        in a (list) of evaluated objects
        Rules: 
            +if nameFile is provided it will try to match it either in folderName
             if provided or  in the current directory
            +if no nameFile is provided it will try to match the allPrefix or 
             fetch everything if None (directory considered follow the same rules 
             based on folderName as before)
        """
        listFileName = ut.findFile(nameFile, allPrefix, folderName)
        results = [learner1DBH.eval_from_onefile_bug(f) for f in listFileName]
        #results = [ut.file_to_dico(f, evfunc = (lambda x: eval(x)) ) for f in listFileName]        
        return results

    @classmethod
    def collect_res_bug(cls, key_path = [], nameFile = None, allPrefix = 'res_', folderName = None):
        """Extract results stored in (a) txt file(s) and group them according to 
        some key values (where key_path provides the path in the potentially 
        nested structure of the results to find the key(s))
        
        Output:
            a dictionary where key is the concatenation of the unique set of 
            keys found and value is the res is a list of all res matching this key
        """
        listRes = cls.read_res_bug(nameFile, allPrefix, folderName)
        res_keys = [tuple([ut.extract_from_nested(res, k) for k in key_path]) 
                    for res in listRes]
        res_keys_unique = list(set(res_keys))
        res = {ut.concat2String(*k_u):[listRes[n] for n, r in enumerate(res_keys) 
                if r == k_u] for k_u in res_keys_unique}
        return res



# ---------------------------
#   END BUG SECTION
# ---------------------------
        
    
# ---------------------------
#   IDEAAL RES -- TO BE USED FOR NOISY SIMULS
# ---------------------------
    @classmethod
    def _list_res_get_ideal_evol(cls, list_res):
        pdb.set_trace()
        tmp = [learner1DBH.ideal_learning(res) for res in list_res]
        nb_run = len(list_res)
        nb_fom = len(tmp[0][1])
        res = [[np.array([tmp[r][:,0], tmp[r][:,1+n]]) for r in range(nb_run)] for n in range(nb_fom)]
        return res
    
    @classmethod
    def ideal_learning(cls, run):
        """ Compute the fom (under testing conditions) for the different functions 
        found along optimization and also the distance between parameters
        """
        testing_dico = run['testing_dico']
        names_fom = testing_dico['fom']
        model_tmp = bh1d.BH1D(**testing_dico)
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
    def rebuild_one_res(cls, res):
        pass

    @classmethod
    def runSimul(cls, dico_simul, params):
        """ from a dico containing the parameters of the simulations and a control 
            function get the results of the simulation"""
        model_tmp = bh1d.BH1D(**dico_simul)
        res = model_tmp(params)
        return res


        
    
def dist(x, y):
    d = np.squeeze(x)-np.squeeze(y)
    return np.dot(d, d)
        
def get_dist_successive(X, n_ev = None):
    distance = [dist(x_n, X[n-1]) for n, x_n in enumerate(X[1:])]
    
    if(n_ev is None):
        n_ev = np.arange(1, len(X)+1)
    return [np.array([0]+n_ev), np.array([0]+distance)]
    
def get_best_so_far(Y, n_ev=None):
    best_tmp = np.Inf
    n_best = []
    y_best = []
    if(n_ev is None):
        n_ev = np.arange(1, len(Y)+1)
    for n, y_n in enumerate(Y):
        if (y_n < best_tmp):
            best_tmp = y_n
            n_best.append(n_ev[n])
            y_best.append(y_n)
    return [np.array(n_best), np.array(y_best)]

def study_convergence(X, Y, end = 0, beg = 0):
    if(len(X) != len(Y)):
        SystemError("X and Y should have the same length")
    nb_obs = len(X)
    nev, dist = get_dist_successive(X)
    nevbest, Ybest = get_best_so_far(Y)
    distbest = dist[np.array([n in nevbest for n in nev])]
    fig, ax = plt.subplots()
    plt.plot(nev, dist, 'b')
    plt.scatter(nevbest, distbest, color = 'b')
    ax.axvspan(0, beg, alpha=0.5, color='grey')
    ax.axvspan(nb_obs-end, nb_obs, alpha=0.5, color='green')

#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    # Testing has been spin off to Test/BH1D
    pass


        