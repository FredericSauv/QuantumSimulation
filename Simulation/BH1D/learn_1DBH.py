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

class learner1DBH(Batch.Batch):
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
            testing_updated = ut.merge_dico(model_dico, testing_dico, update_type = 0, copy = True)
            testing_updated.update(dico_update)
            self._build_control(testing_updated)
            model_test = bh1d.BH1D(**testing_updated)
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

    def extractFromDico(dico, listRes = [], listParams = [], dicoConstraints = {}):
        raise NotImplementedError()
            
    @classmethod
    def _build_control_from_string(cls, control, random_gen=None, context_dico = None):
        if(context_dico is not None):
            context = {k:v for k, v in context_dico.items() if k not in 
                       ['control_obj', 'random_obj', 'mp_obj']}
        else:
            context = None
            
        if(random_gen is None):
            random_gen = RandomGenerator()
        func_factory = pFunc_zoo.pFunc_factory(random_gen, context)
        built_control = func_factory.eval_string(control)
        return built_control
    
    @classmethod
    def _processing_meta_configs(cls, dico):
        """ AdHoc processing rules when dealing with meta configs: 
        helps to create the controler    
        """
        model_dico = dico.get('model_dico')
        if(model_dico is not None):
            dico['model_dico'] = cls._process_controler(model_dico)

        testing_dico = dico.get('testing_dico')
        if(testing_dico is not None):
            dico['testing_dico'] = cls._process_controler(testing_dico)
    
        return dico

    @classmethod
    def _process_controler(cls, dico):
        """ ad-hoc processing to make description of the controller not too long 
        (1) retrieve all the keys starting with ctl_
        (1a) ctl_final is the expression of the controler
        (1b) otherss are the definition of bricks involved in ctl_final
        They are parsed by pFunc_parser
        
        e.g. dico = {'ctl_a':xxx, 'ctl_b':yyy, 'ctl_c':zzz, 'ctl_final':"*(#a, +(#b, #c))"}
        """
        dico_processed = copy.copy(dico)
        
        if('ctl_shortcut' in dico):
            print('use of shortcut')
            shortcut = dico['ctl_shortcut']
            
            # no free params
            ow = "{'name_func':'OwriterYWrap', 'ow':[(-100,0,0),(T,100+T,1)]}"
            bds = "{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}"
            linear = "{'name_func':'LinearFunc', 'bias':0, 'w':1/T}"
            one = "{'name_func':'ConstantFunc', 'c0':[1]}"
            sinpi = "{'name_func':'FourierFunc','A':[0], 'B':[1],'Om':[np.pi/T]}"
            
            #tunable
            sinfour = "{'name_func':'FourierFunc','T':T,'freq_type':'principal','B_bounds':%s,'nb_H':%s}"
            pwc = "{'name_func':'StepFunc','T':T,'F_bounds':%s,'nb_steps':%s}"
            rfour ="{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','A_bounds':%s,'B_bounds':%s,'nb_H':%s}"
            
            if(shortcut[:11] == 'owbds01_pwc'):
                nb_params = int(shortcut[11:])
                dico_atom = {'ow':ow,'bd':bds,'pwc':pwc %('(0,1)',nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwc))'}
             
            elif(shortcut[:12] == 'owbds01_crab'):
                nb_params = int(shortcut[12:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                            'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#rfour)))))'}
            
            elif(shortcut[:14] == 'owbds01_trfour'):
                nb_params = int(shortcut[14:])
                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'sinfour':sinfour%('(-1,1)', nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,#sinfour)))'}
                        
            elif(shortcut[:14] == 'owbds01_linear'):
                dico_atom = {'ow':ow,'bd':bds,'lin':linear}
                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
            
            else:
                 raise SystemError('implement more shotcuts here')
            
            dico_processed['control_obj'] = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
        
        else:
            dico_atom = {}
            dico_expr = {}
            list_keys_to_remove = []
            for k, v in dico_processed.items():
                bits = k.split('_')
                if(bits[0] == 'ctl'):
                    list_keys_to_remove.append(k)
                    if(bits[1] == 'final'):
                        dico_expr.update({bits[1]:v})
                    else:
                        dico_atom.update({bits[1]:v})
                    
    
            for k in list_keys_to_remove:
                del dico_processed[k]
            if('final' in dico_expr):
                dico_processed['control_obj'] = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
        return dico_processed
        
    @classmethod
    def evaluator(expr):
        return pFunc_base.pFunc_base.eval_with_pFunc(expr)
    # ----------------------------------------------------------------------- #
    # Reading results aera:
    # (from parent): read_res // collect_res
    # process_list_res
    #
    # collect = {'collec1': list_res} i.e. a collection is a dict with keys being
    #           
    # list_res =[res1, ..., resN]
    #
    # res is a nested structure containing a lot of info about one run
    # ----------------------------------------------------------------------- #
    @classmethod
    def collect_and_process_res(cls, key_path = [], nameFile = None, allPrefix = 'res_', 
                                folderName = None, printing = False, ideal_evol = False):
        """ Collect a list of res in **folderName** strating with **allPrefix**
        group them by configurations specified by **key_path**"""
        collect = learner1DBH.collect_res(key_path, nameFile, allPrefix, folderName)
        process = learner1DBH.process_collect_res(collect, printing, ideal_evol)
        
        return process
    
    @classmethod
    def process_collect_res(cls, collect_res, printing = False, ideal_evol = False):
        dico_processed = {k: learner1DBH.process_list_res(v) for k, v in collect_res.items()}
        return dico_processed
        
    @classmethod
    def process_list_res(cls, list_res, printing = False, ideal_evol = False):
        """ Take a list of N res(runs) and extract interesting informations (stats)
        packaged as a dico with the following entries:
            + evol_fom_stats: stats (over the N runs) of the fom vs the number 
                                of evaluations
            + evol_ideal_fom_stats x
            + 
            + optimal_functions : 
            + optimal_function_stats: stats of the value of the driving function
                                    over time
            + optimal_fom_stats x
            + optimal_fun1_stats x
        """
        #Path (in the nested structure of the res)
        path_optim_fev_fom = ['extra_history_nev_fun']
        path_optim_func = ['func']
        path_optim_fun = ['fun']
        path_test_fom = ['test_fom']
        path_test_fom_names = ['test_fom_names']
        path_test_t = ['config', 'testing_dico', 'T']
        path_test_t_bis = ['config', 'model_dico', 'T']
        
        
        # evolution of fom (measured) over the aggregated number of func evaluations
        optim_fev_fom = [ut.try_extract_from_nested(run, path_optim_fev_fom) for run in list_res]
        is_none = [o is None for o in optim_fev_fom] 
        if(np.sum(is_none) > 0):
            print(str(np.sum(is_none)) + ' without extra_history_nev_fun')
            optim_fev_fom = [o for o in optim_fev_fom if o is not None]
        if(len(optim_fev_fom) > 0):
            evol_fom_stats = ut.merge_and_stats_TS(optim_fev_fom)
        else:
            evol_fom_stats = None
        # optim_fev_params = [np.array(ut.extract_from_nested(run, path_optim_fev_fom)) for run in list_res]

        #opt function computed between 0,T
        try:
            list_T = [ut.extract_from_nested(run, path_test_t) for run in list_res]
        except:
            list_T = [ut.extract_from_nested(run, path_test_t_bis) for run in list_res]
        list_array_T = [np.arange(0, T, T/100) for T in list_T]
        list_optim_func = [pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(run, path_optim_func)) for run in list_res]
        optim_func_TS = [np.c_[list_array_T[n], func(list_array_T[n])] for n, func in enumerate(list_optim_func)]
        optim_func_stats = ut.merge_and_stats_TS(optim_func_TS)

        # stats (avg, mini, maxi, std, avg_pstd, avg_mstd) of the optimal fom
        test_fom_names = [ut.extract_from_nested(run, path_test_fom_names) for run in list_res]
        test_fom_names_ref = test_fom_names[0]
        assert np.all([(t == test_fom_names_ref) for t in test_fom_names]), "can't mix different fom..."
        test_fom = np.array([ut.extract_from_nested(run, path_test_fom) for run in list_res]).T
        test_fom_stats = [ut.get_stats(l_fom, dico_output = True) for l_fom in test_fom]
        fun = np.array([ut.extract_from_nested(run, path_optim_fun) for run in list_res])
        fun_stats = ut.get_stats(fun, dico_output = True)
        

        if(printing):
            print('FUN:  (avg, min, max)')
            print(fun_stats[:3])
            print('TESTED RES:FUN0  (should be the same as above if no noise)')
            print(test_fom_stats[0, :3])
            print('TESTED RES:FUN1')
            print(test_fom_stats[1, :3])

        #populate dico res
        dico_res = {}
        dico_res['nb_runs'] = len(list_res)
        dico_res['ev_fom'] = evol_fom_stats
        dico_res['funcplot_stats'] = optim_func_stats
        dico_res['funcplot_TS'] = optim_func_TS
        dico_res['best_fom'] = fun_stats
        dico_res['test_fom'] = test_fom_stats[0]
        for n, name_fom in enumerate(test_fom_stats[1:]):
            dico_res['test_' + test_fom_names_ref[n+1]] = test_fom_stats[(n+1)]   
        
        # evol of fom (ideal: testing environment) over number of func evaluation
        if(ideal_evol):
            pdb.set_trace()
            name_fom, ideal_nev_foms, dist_params = learner1DBH.ideal_learning_from_list_res(list_res)
            ideal_nev_foms_stats = [ut.merge_and_stats_TS(one_fom) for one_fom in ideal_nev_foms]
            dist_params_stats = ut.merge_and_stats_TS(dist_params)
            dico_res['ev_distparams'] = dist_params_stats
            dico_res['ev_idealfom'] = ideal_nev_foms_stats[0]
            
            for n, one_fom_stats in enumerate(ideal_nev_foms_stats[1:]):
                dico_res['ev_ideal'+name_fom[n+1]] = one_fom_stats   


        return dico_res
    
    
    
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
    def ideal_learning_from_list_res(cls, list_res):
        pdb.set_trace()
        tmp = [learner1DBH.ideal_learning(res) for res in list_res]
        nb_run = len(list_res)
        nb_fom = len(tmp[0][1])
        res = [[np.array([tmp[r][:,0], tmp[r][:,1+n]]) for r in range(nb_run)] for n in range(nb_fom)]
        return res


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

    @classmethod
    def find_best_run_in_dico_simul(cls, dico_simul, path_criterion, test = 'min', filt = None):
        """ dico_simul = {'name_simul':list_runs}
            list_runs = [run1, ..., run30]
        """
        best = np.inf
        index_best =[None, None]
        if(test == 'max'):
            mult = -1
        else:
            mult = 1
        if(filt is None):
            filt = lambda x:True
        
        for k, res in dico_simul.items():
            if(filt(k)): 
                for n, run in enumerate(res):
                    tmp = mult * ut.extract_from_nested(run, path_criterion)
                    if tmp < best:
                        best = tmp 
                        index_best[0] = k
                        index_best[1] = n

        if(test == 'max'):
            best *= -1
        print(best)
        return copy.copy(dico_simul[index_best[0]][index_best[1]])

    @classmethod     
    def study_convergence_one_res(cls, res):
        path_X = ['test_more', 'X_evol']
        path_Y = ['test_more', 'Y_evol']
        path_nbinit = ['config', 'optim_dico', 'init_obj']

        try:
            X = ut.extract_from_nested(res, path_X)
            Y = ut.extract_from_nested(res, path_Y)
            nbinit = ut.extract_from_nested(res, path_nbinit)
            study_convergence(X, Y, beg = nbinit, end = 15)
        except:
            print("couldn't build the graph probably data missing")
        
        
    
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


        