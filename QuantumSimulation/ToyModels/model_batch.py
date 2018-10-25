class BatchModel(Batch):
    """ Batch class customized for the management of jobs based on models 
    (QuantumSimulation.ToyModels.ModelBase) and Optimization
    

    """

    @classmethod
    def eval_from_onefile(cls, name):
        """ eval the first element of the first line of a file """
        res = ut.eval_from_file(name, evfunc = pFunc_base.eval_with_pFunc)
        return res



    @classmethod
    def _build_underlying_model(cls, model_dico):
        """ to be implemented in the subclass.. 
        e.g. model = tl.Qubits(**model_dico)
            return model
        """
        if(cls.UNDERLYING_MODEL_CONSTRUCTOR is None):
            raise NotImplementedError()
        else:
            model = cls.UNDERLYING_MODEL_CONSTRUCTOR(**model_dico)
        return model

    def _build_control(self,model_dico):
        pass

    def run_one_procedure(self, config):
        """ Iniatilze the model (based on _build_underlying_model), initaialize the optimizer
        run the optimization, save results, build a testing_model, test optimal parameters 
        save testing results
        """
        self.random_gen = RandomGenerator.RandomGenerator.init_random_generator(config['_RDM_SEED'])
        self.mp = MP.MPCapability.init_mp(config.get(config['_MP_FLAG']))
        dico_update ={'rdm_obj': self.random_gen, 'mp_obj':self.mp}
        
        model_dico = config.get('model_dico', {})
        if (model_dico not in [None, {}]):
            track_learning = model_dico.pop('track_learning', False)
            model_dico.update(dico_update)
            self._build_control(model_dico)  ### TOCHECKKK
            model = type(self)._build_underlying_model(model_dico)       
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
            model = None
            model_dico = {}
            res={'params':[]}
            
        testing_dico = config.get('testing_dico')
        #Testing_dico contains ONLY updates to apply to model_dico
        if(testing_dico is not None):
            # Options to specify how you want the testing done: **params_force** 
            # allows to use specific parameters values **testing_same_func**
            # forces to use the same function as the one in the model (default behavior)
            # **use_exp_params** (specific to BO2): use expected best params
            # rather than the seen one 
            testing_updated = ut.merge_dico(model_dico, testing_dico, update_type = 0, copy = True)
            testing_updated.update(dico_update)
            force_params = testing_updated.pop('testing_force_params', None)
            same_func = testing_updated.pop('testing_same_func', True)
            #0 No // 1 Yes instead of params // 2 Test both
            use_exp_params = testing_updated.pop('use_exp_params', 2) 

            self._build_control(testing_updated)
            model_test = type(self)._build_underlying_model(testing_updated)
            
            if(same_func and (model is not None)):
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

    # ---------------------------
    #    DEAL WITH RESULTS
    #    part 2: PROCESS RESULTS
    # ---------------------------    
    # More capabilities to aggregate results
    #
    # STRUCTURE OF THE RESULTS
    # collect = {'collec1': list_res,...} i.e. a collection is a dict with keys 
    #           being the name of the collection of resamd valuesia a list of res 
    #           list_res =[res1, ..., resN]
    #
    # res = corresponds to one run and is a nested structure containing a lot of info
    #
    #
    # stats = corresponds to a configuration where the list of res associated have
    #         been processed and compacted  
    #
    #
    # METHODS (MAIN)
    # ++ collect_and_process_res
    # ++ collection_get_best
    # ++ list_res_get_best
    # ++ one_res_study_convergence
    # ++ list_res_study_convergnce

    #PATH IN RES STRUCTURE
    _P_HISTO_EV_FUN = ['extra_history_nev_fun'] #_path_optim_fev_fom
    _P_HISTO_TM_FUN = ['extra_history_time_fun'] #_path_optim_fev_fom
    _P_BEST_FUNC = ['func'] # path_optim_func
    _P_BEST_FUN = ['fun']   # path_optim_fun 
    _P_BEST_FUN_NAMES = ['fun_name']
    _P_TEST_FOM = ['test_fom'] # path_test_fom 
    _P_TEST_FOM_NAMES = ['test_fom_names'] #path_test_fom_names
    _P_TEST_FOM_EXP = ['test_fom_exp'] # path_test_fom 
    _P_TEST_FOM_EXP_NAMES = ['test_fom_exp_names'] #path_test_fom_names
    _P_TEST_T = ['config', 'testing_dico', 'T'] # path_test_t
    _P_MODEL_T = ['config', 'model_dico', 'T'] #path_test_t_bis
    _P_NAME_RES = ['config', '_RES_NAME'] #path_name
    _P_CONFIG = ['config', 'model_dico'] #path_name         
    _P_CONFIG_OPTIM = ['config', 'optim_dico']
    _P_CONFIG_TESTING = ['config', 'testing_dico']
    _P_X_HIST = ['test_more', 'X_evol']
    _P_Y_HIST = ['test_more', 'Y_evol']
    _P_NBINIT = ['config', 'optim_dico', 'init_obj']
    
    @classmethod
    def _process_collection_res(cls, collection, printing = False, ideal_evol = False):
        dico_processed = {k: cls._process_list_res(v, printing, ideal_evol) for k, v in collection.items()}
        return dico_processed
        
    @classmethod
    def _process_list_res(cls, list_res, printing = False, ideal_evol = False):
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

        Return None for the different fields if it fails
        """
        # evolution of observed FOM over the aggregated number of func evaluations
        evol_fom_stats = cls._list_res_get_evol_fom_stats(list_res)
        
        # evolution of observed FOM over the aggregated number of func evaluations
        evol_nev_time_stats, evol_nev_onetime_stats = cls._list_res_get_evol_nev_time_stats(list_res)
        
        # optimal control functions 
        optim_func_TS = cls._list_res_get_optim_func_TS(list_res)
        if(optim_func_TS is not None):
            optim_func_stats = ut.merge_and_stats_TS(optim_func_TS)
        else:
            optim_func_stats = None
            
        # stats (avg, mini, maxi, std, avg_pstd, avg_mstd) of the tested optimal 
        # FOM ('test_fom') and the optimal FOM found during optim ('fun')
        test_fom_names = cls._list_res_get_names_from(list_res, cls._P_TEST_FOM_NAMES)
        test_fom_stats = cls._list_res_get_stats_from(list_res, cls._P_TEST_FOM)
        fun_stats = cls._list_res_get_stats_from(list_res, cls._P_BEST_FUN)
        fun_names = cls._list_res_get_names_from(list_res, cls._P_BEST_FUN_NAMES)
        test_fom_exp_stats = cls._list_res_get_stats_from(list_res, cls._P_TEST_FOM_EXP)

        # A litlle bit of testing
        if(fun_names is not None):
            name_fun_ref = fun_names[0]
            if(name_fun_ref in test_fom_names):
                try:
                    tmp_index = np.where(test_fom_names == name_fun_ref)[0][0]
                    tmp_fun = test_fom_stats[tmp_index, 0]
                    tmp_fom = fun_stats[0]
                    print("avg over testing: {0}, and over optim: {1}".format(tmp_fom, tmp_fun)) 
                except:
                    pass

        if(printing and fun_stats is not None):
            print('FUN:  (avg, min, max)')
            print(fun_stats[0]['avg'], fun_stats[0]['min'], fun_stats[0]['max'])
            print('TESTED RES:FUN0  (should be the same as above if no noise)')
            print(test_fom_stats[0]['avg'], test_fom_stats[0]['min'], test_fom_stats[0]['max'])

        #populate dico res
        dico_res = {}
        dico_res['nb_runs'] = len(list_res)
        dico_res['ev_fom'] = evol_fom_stats
        dico_res['funcplot_stats'] = optim_func_stats
        dico_res['funcplot_TS'] = optim_func_TS
        dico_res['best_fom'] = fun_stats
        dico_res['nev_time_stats'] = evol_nev_time_stats
        dico_res['nev_onetime_stats'] = evol_nev_onetime_stats
        
        if(test_fom_stats is not None):
            dico_res['test_fom'] = test_fom_stats[0]
            if(len(test_fom_stats) > 1):
                dico_res.update({'test_' + n : v for n, v in zip(test_fom_names[1:], test_fom_stats[1:])})
        
        if(test_fom_exp_stats is not None):
            dico_res['test_fom_exp'] = test_fom_exp_stats[0]
            if(len(test_fom_exp_stats) > 1):
                dico_res.update({'test_' + n +'_exp':v for n, v in zip(test_fom_names[1:], test_fom_exp_stats[1:])})
        
        # evol of fom (ideal: testing environment) over number of func evaluation
        if(ideal_evol):
            dico_ideal = cls._list_res_get_ideal_evol(list_res)
            dico_res.update(dico_ideal)
    
        return dico_res    
    
    @classmethod
    def _list_res_get_evol_fom_stats(cls, list_res):
        optim_fev_fom = [ut.try_extract_from_nested(run, cls._P_HISTO_EV_FUN) for run in list_res]
        is_none = [o is None for o in optim_fev_fom] 
        if(np.sum(is_none) > 0):
            print(str(np.sum(is_none)) + ' without extra_history_nev_fun')
            optim_fev_fom = [o for o in optim_fev_fom if o is not None]
        if(len(optim_fev_fom) > 0):
            evol_fom_stats = ut.merge_and_stats_TS(optim_fev_fom)
        else:
            evol_fom_stats = None
        return evol_fom_stats
    
    

    @classmethod 
    def _list_res_get_evol_nev_time_stats(cls, list_res):
        list_nev_time = [cls._one_res_get_nev_time(r) for r in list_res]
        is_none = [o is None for o in list_nev_time] 
        if(np.sum(is_none) > 0):
            print(str(np.sum(is_none)) + ' without extra_history_nev_fun')
            list_nev_time = [o for o in list_nev_time if o is not None]
        if(len(list_nev_time) > 0):
            merged_TS = ut.merge_TS(list_nev_time, rule='linear')            
            merged_array = np.array([ [ts[0]]+list(ts[1]) for ts in merged_TS])
            stats = np.array([np.concatenate(([ind], ut.get_stats(ts))) for ind, ts in merged_TS])
            dico_stats = ut.stats_array2dico(stats)
            merged_diff = [[m[0], np.array(m[1:]-merged_array[n, 1:])] for n, m in enumerate(merged_array[1:])]
            stats_diff = np.array([np.concatenate(([ind], ut.get_stats(ts))) for ind, ts in merged_diff])
            dico_stats_diff = ut.stats_array2dico(stats_diff)
        else:
            dico_stats = None
            dico_stats_diff = None
        return (dico_stats, dico_stats_diff)
    
    @classmethod
    def _get_nev_onetime(cls, nev_time):
        nev_onetime = np.c_[nev_time[1:,0], np.diff(nev_time[:,1])]
        return nev_onetime
    
    @classmethod
    def _one_res_get_nev_time(cls, one_res):
        nev = ut.try_extract_from_nested(one_res, cls._P_HISTO_EV_FUN)
        time = ut.try_extract_from_nested(one_res, cls._P_HISTO_TM_FUN)
        if((nev is not None) and (time is not None)):
            nev_time = np.array(nev)
            nev_time[:,1] = np.array(time)[:,0]        
        else:
            nev_time = None            
        return nev_time

    @classmethod
    def _list_res_get_optim_func_TS(cls, list_res):
        try:
            try:
                list_T = [ut.extract_from_nested(run, cls._P_TEST_T) for run in list_res]
            except:
                list_T = [ut.extract_from_nested(run, cls._P_MODEL_T) for run in list_res]
            list_array_T = [np.arange(0, T, T/1000) for T in list_T]
            list_optim_func = [pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(run, cls._P_BEST_FUNC)) for run in list_res]
            optim_func_TS = [np.c_[list_array_T[n], func(list_array_T[n])] for n, func in enumerate(list_optim_func)]
        except:
            optim_func_TS = None
        return optim_func_TS

    @classmethod
    def _list_res_get_stats_from(cls, list_res, path):
        """ Stats on the values of the best value of fom found during optim """
        vals = [ut.try_extract_from_nested(run, path, True) for run in list_res]
        is_not_none = np.array([v is not None for v in vals])
        if(np.sum(is_not_none) == 0):
            res = None
        else:
            vals = np.array([l for l, b in zip(vals, is_not_none) if b]).T
            res = [ut.get_stats(v, dico_output = True) for v in vals]
        return res

    @classmethod
    def _list_res_get_names_from(cls, list_res, path):
        names = [ut.try_extract_from_nested(run, path, True) for run in list_res]
        is_not_none = np.array([n is not None for n in names])
        if(np.sum(is_not_none) == 0):
            res = None
        else:
            names = [l for l, b in zip(names, is_not_none) if b]
            names_ref = names[0]
            if not(np.all([(t == names_ref) for t in names])):
                raise SystemError("can't mix different fom...")
            res = names_ref
        return res
    
    # ---------------------------
    #    DEAL WITH RESULTS
    #    part 2b: IDEAL LEARNING (and as a bonus dist betweem parameters tried)
    # ---------------------------    
    @classmethod
    def _list_res_get_ideal_evol(cls, list_res):
        try:
            list_ideal =  cls._list_res_ideal_learning(list_res)
            if(list_ideal is None):
                dico_res = {}
            else:
                name_fom, ideal_nev_foms, dist_params = list_ideal
                ideal_nev_foms_stats = [ut.merge_and_stats_TS(one_fom) for one_fom in ideal_nev_foms]
                dist_params_stats = ut.merge_and_stats_TS(dist_params)
                dico_res = {'ev_distparams':dist_params_stats,'ev_idealfom':ideal_nev_foms_stats[0]}
                for n, one_fom_stats in enumerate(ideal_nev_foms_stats[1:]):
                    dico_res['ev_ideal'+name_fom[n+1]] = one_fom_stats   
        except:
            dico_res ={}
        return dico_res
        
        
    @classmethod
    def ideal_learning(cls, res):
        """ Compute the fom (under testing conditions) for the different functions 
        found along the optimization procedure 

        and also the distance between parameters
        should return (names_fom, res, res_params_dist)
        """
        try:
            evol_params = res.get('extra_history_nev_params')
            if(evol_params is None):
                return None
            else:
                testing_dico = ut.extract_from_nested(res, cls._P_CONFIG_TESTING)
                testing_dico['fom_print'] = False
                names_fom = testing_dico['fom']
                model_tmp = cls._build_underlying_model(testing_dico)
                evol_ideal_fom = np.array([np.r_[np.array([p[0]]), np.array(model_tmp(p[1], trunc_res = False))] for p in evol_params])
                res_params_dist = np.array([[par[0], np.linalg.norm(par[1] - evol_params[n][1])] for n, par in enumerate(evol_params[1:])])

                return names_fom, evol_ideal_fom, res_params_dist
        except:

            print("can't find extra_history_params_fun keys in the res.. no ideal learning possible")
            return None

        
        
    @classmethod
    def _list_res_ideal_learning(cls, list_res):
        tmp = [cls.ideal_learning(res) for res in list_res]
        # [[names, fom_evol, dist_evol]]
        is_not_none = np.array([t is not None for t in tmp])
        #pdb.set_trace()
        if(np.sum(is_not_none) == 0):
            res = None
        else:
            tmp = [t for t, b in zip(tmp, is_not_none) if b]
            nb_run = len(tmp)
            names_fom = [t[0] for t in tmp]
            names_fom_ref = names_fom[0]
            assert np.all([n == names_fom_ref for n in names_fom]), "names test fom don't match between runs.. can't average them"
            nb_fom = len(names_fom_ref)
            evol_fom = [[tmp[r][1][:,[0, n+1]] for r in range(nb_run)] for n in range(nb_fom)]
            evol_dist = [tmp[r][2] for r in range(nb_run)]
            res = (names_fom_ref, evol_fom, evol_dist)
        return res

    


    # ---------------------------
    #    DEAL WITH RESULTS
    #    part 3: Extract best accoring to some criterion
    # ---------------------------
    @classmethod
    def collection_get_best(cls, dico_simul, path_criterion, test = 'min', filt = None, return_value = False):
        """ dico_simul = {'name_simul':list_runs}
            list_runs = [run1, ..., run30]
        """
        best = np.inf
        res_best = None
        mult_coeff = -1 if (test == 'max') else 1

        for k, list_res in dico_simul.items():
            if(filt(k)): 
                res_tmp, val = cls.list_res_get_best(list_res, path_criterion, test = 'min', filt = None, return_value = False)
                tmp = val * mult_coeff
                if(tmp < best):
                    res_best = res_tmp
                    best = tmp
        best *= mult_coeff
        print(best)
        if return_value:
            return (res_best, best)
        else:
            return res_best

    @classmethod
    def list_res_get_best(cls, list_res, path_criterion, test = 'min', filt = None, return_value = False):
        """ Find and return the best res in a list of res """
        best = np.inf
        index_best =[None]
        mult_coeff = -1 if (test == 'max') else 1
        for n, run in enumerate(list_res):
            tmp = mult_coeff * ut.extract_from_nested(run, path_criterion)
            if tmp < best:
                best = tmp 
                index_best[1] = n
        
        best *= mult_coeff  
        res_best = copy.copy(list_res[index_best[0]])
        if return_value:
            return (res_best, best)
        else:
            return res_best
    
    # ---------------------------
    #    DEAL WITH RESULTS
    #    part -1: utility
    # ---------------------------
    @classmethod
    def one_res_get_func(cls, res):
        return pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(res, cls._P_BEST_FUNC))

    @classmethod
    def one_res_plot_func(cls, res):
        T = ut.extract_from_nested(res, cls._P_MODEL_T)
        tt = np.linalg(-0.1, T +0.1, 10000)
        fun = pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(res, cls._P_BEST_FUNC))
        fun.plot_function(tt)
        
    @classmethod     
    def one_res_study_convergence(cls, res):
        """  works only when res contains"""
        try:
            X = ut.extract_from_nested(res, cls._P_X_HIST)
            Y = ut.extract_from_nested(res, cls._P_Y_HIST)
            nbinit = ut.extract_from_nested(res, cls._P_NBINIT)
            if(np.ndim(nbinit) > 1):
                nbinit = np.shape(nbinit)[0]
            study_convergence(X, Y, beg = nbinit, end = 15)
        except:
            print("couldn't build the graph the underlying data are probably missing")
        
        
    @classmethod     
    def list_res_study_convergence(cls, res):
        """  works only when res contains"""
        pass
        
        
# ---------------------------
# To be implemented in the subclass // TOFINISH
# ---------------------------            
    @classmethod
    def _processing_meta_configs(cls, dico):
        """ 
        """
        raise NotImplementedError()


    @classmethod
    def one_res_rebuild(cls, res):
        """ Pass it a res object and it should rerun the model"""
        pass

    @classmethod
    def runSimul(cls, dico_simul, params):
        """ from a dico containing the parameters of the simulations and a control 
            function get the results of the simulation"""
        model_tmp = cls.UNDERLYING_MODEL_CONSTRUCTOR(**dico_simul)
        res = model_tmp(params)
        return res
               

        
def dist(x, y):
    """ Compute distances between two vectors (or two list of vectors)
    """
    xx, yy = np.squeeze(x), np.squeeze(y)
    shape_xx, shape_yy = np.shape(xx), np.shape(yy)
    if (shape_xx != shape_yy):
        raise SystemError("shape  of x {0} different from shape of y {1}".format(shape_xx, shape_yy))
    diff = xx - yy
    if(len(shape_xx)==1):
        res = np.square(np.dot(diff, diff))
    elif(len(shape_xx)==2):
        res = np.array([dist(d) for d in diff])
    else:
        raise NotImplementedError
    return res
        
def get_dist_successive(X, n_ev = None):
    """ For a list of X compute the successive distances btween Xs """
    distance = [dist(x_n, X[n-1]) for n, x_n in enumerate(X[1:])]
    
    if(n_ev is None):
        n_ev = np.arange(1, len(X)+1)
    return [np.array([0]+n_ev), np.array([0]+distance)]
    
def get_best_so_far(Y, n_ev=None):
    """ aggregated MIN value of Y (list)"""
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
    """ X and Y are respectively a list of parameters and values 
    plot 
    """
    if(len(X) != len(Y)):
        SystemError("X and Y should have the same length")
    nb_obs = len(X)
    nev, dist = get_dist_successive(X)
    nevbest, Ybest = get_best_so_far(Y)
    distbest = dist[np.array([n in nevbest for n in nev])]
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(nev, dist, 'b')
    ax1.scatter(nevbest, distbest, color = 'b')
    ax1.axvspan(0, beg, alpha=0.5, color='grey')
    ax1.axvspan(nb_obs-end, nb_obs, alpha=0.5, color='green')        
    ax2.plot(nevbest, Ybest)  
    ax2.axvspan(0, beg, alpha=0.5, color='grey')
    ax2.axvspan(nb_obs-end, nb_obs, alpha=0.5, color='green')
        
        




#==============================================================================
#                   BATCH CLASS
# Add some ad-hoc capabilities to deal with parametrized functions for the 
# controls of the model
#==============================================================================
class BatchParametrizedControler(Batch):
    """
    Batch:
        * parse an input file to create a list of configurations to run
        * run it According to some procedures (funcProc)
        * save results 
    """
    def __init__(self, config_object = None, rdm_object = None, debug = False):
        """ Init the batch with a <dic> (it represents one config) OR <list<dico>>
        (several configs) OR <str:file_path> (onefile = one config) OR
        <list<str:file_path>> (several configs)
        """
        Batch.__init__(self, config_object, rdm_object, debug)

# ---------------------------
#   ADHOC METHODS TO DEAL WITH THE PARAMETRIZED FUNCTIONS
# ---------------------------    
    def _build_control(self, model_dico):
        """ if the control_object is a string evaluate it if not do nothing"""
        control = model_dico['control_obj']
        if(ut.is_str(control)):
            model_dico['control_obj'] = type(self)._build_control_from_string(control,
                      self.random_gen, model_dico)



    @classmethod
    def _build_control_from_string(cls, control, random_gen=None, context_dico = None):
        if(context_dico is not None):
            context = {k:v for k, v in context_dico.items() if k not in 
                       ['control_obj', 'random_obj', 'mp_obj']}
        else:
            context = None
    
        if(random_gen is None):
            random_gen = RandomGenerator.RandomGenerator()
        
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
        #TODO: NEED TO BE REFACTORED // PUT SOMEWHERE ELSE
        """
        dico_processed = copy.copy(dico)
        
        if('ctl_shortcut' in dico):
            logger.info('use of shortcut')
            shortcut = dico['ctl_shortcut']
            
            # no free params
            ow = "{'name_func':'OwriterYWrap', 'ow':[(-inf,0,0),(T,inf,1)]}"
            ow_r = "{'name_func':'OwriterYWrap', 'ow':[(-inf,0,1),(T,inf,0)]}"
            bds = "{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}"
            linear = "{'name_func':'LinearFunc', 'bias':0, 'w':1/T}"
            linear_r = "{'name_func':'LinearFunc', 'bias':1, 'w':-1/T}"
            one = "{'name_func':'ConstantFunc', 'c0':[1]}"
            half = "{'name_func':'ConstantFunc', 'c0':[0.5]}"
            mone = "{'name_func':'ConstantFunc', 'c0':[-1]}"
            sinpi = "{'name_func':'FourierFunc','A':[0], 'B':[1],'Om':[np.pi/T]}"
            pow15 = "{'name_func':'PowerFunc','power':1.5}"
            sqrt  = "{'name_func':'PowerFunc','power':0.5}"
            # square = "{'name_func':'PowerFunc','power':2}"
    
            
            #tunable
            grbf = "{'name_func':'GRBFFunc','A':%s, 'x0':%s,'l':%s,'A_bounds':%s}"
            rfour ="{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','A_bounds':%s,'B_bounds':%s,'nb_H':%s}"
            rsinfour = "{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','B_bounds':%s,'nb_H':%s}"
            four = "{'name_func':'FourierFunc','T':T,'freq_type':'principal','A_bounds':%s,'B_bounds':%s,'nb_H':%s}"
            sinfour = "{'name_func':'FourierFunc','T':T,'freq_type':'principal','B_bounds':%s,'nb_H':%s}"
            omfour = "{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','A_bounds':%s,'B_bounds':%s,'nb_H':%s}"
            omsinfour = "{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','B_bounds':%s,'nb_H':%s}"
            pwc = "{'name_func':'StepFunc','T':T,'F_bounds':%s,'nb_steps':%s}"
            pwl = "{'name_func':'PWL','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}"
            pwlr = "{'name_func':'PWL','TLast':T,'T0':0,'F0':1,'FLast':0,'F_bounds':%s,'nb_steps':%s}"
            logis = "{'name_func':'LogisticFunc','L':2,'k':%s,'x0':0}"
            logisflex = "{'name_func':'LogisticFunc','L':%s,'k':%s,'x0':%s}"

            
            if(shortcut[:11] == 'owbds01_pwc'):
                nb_params = int(shortcut[11:])
                dico_atom = {'ow':ow,'bd':bds,'pwc':pwc %('(0,1)',nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwc))'}

            elif(shortcut[:12] == 'owbds01r_pwc'):
                nb_params = int(shortcut[12:])
                dico_atom = {'ow':ow_r,'bd':bds,'pwc':pwc %('(0,1)',nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwc))'}
                        
            elif(shortcut[:11] == 'owbds01_pwl'):
                nb_params = int(shortcut[11:])
                dico_atom = {'ow':ow,'bd':bds,'pwl':pwl %('(0,1)',nb_params+1)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwl))'}

            elif(shortcut[:13] == 'owbds01r_pwlr'):
                nb_params = int(shortcut[13:])
                dico_atom = {'ow':ow_r,'bd':bds,'pwlr':pwlr %('(0,1)',nb_params+1)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwlr))'}

            ### RDMIZED FREQ
            elif(shortcut[:13] == 'owbds01_1crab'):
                # Custom Crab parametrization f(t) = g(t) * (1 + alpha(t)* erf((four series)))
                # slightly different from the normal one (cf. before)
                # additional erf function (logistic function such that the four 
                # series is bounded) alpha(t) is sine ** 1.5
                nb_params = int(shortcut[13:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                k = 4 /nb_params
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                             'ctm':mone,'logis': logis%(str(k)),
                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,**(+(#logis,#ctm),#rfour))))))'}
             
            elif(shortcut[:13] == 'owbds01_2crab'):
                # alpha(t) is sine ** 0.5
                nb_params = int(shortcut[13:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                k = 4 /nb_params
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                             'powscale':sqrt, 'ctm':mone,'logis': logis%(str(k)),
                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(**(#powscale,#scale),**(+(#logis,#ctm),#rfour))))))'}
                        
            elif(shortcut[:13] == 'owbds01_3crab'):
                # Custom Crab parametrization f(t) = g(t) * (1 + alpha(t)* erf((four series)))
                # slightly different from the normal one (cf. before)
                # additional erf function (logistic function such that the four 
                # series is bounded) alpha(t) is sine ** 1.5
                nb_params = int(shortcut[13:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                k = 4 /nb_params
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': linear_r, 'ct': one,
                             'ctm':mone,'logis': logis%(str(k)),
                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                
                dico_expr = {'final':'**(#ow,**(#bd,+(#guess,*(#scale,**(+(#logis,#ctm),#rfour)))))'}

            elif(shortcut[:13] == 'owbds01_4crab'):
                # AANOTHER Custom Crab parametrization f(t) = g(t) + erf((sin four series)))
                nb_params = int(shortcut[13:])
                k = 4 /nb_params
                x0 = '0.1*T'
                k2 = '60/T'          
                L = '1'
                
                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone, 'mask':logisflex%(L,k2,x0),
                             'logis': logis%(str(k)), 'sinfour':rsinfour%('(-1,1)', nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,*(#mask,**(+(#logis,#ctm),#sinfour)))))'}


            elif(shortcut[:12] == 'owbds01_crab'):
                # Crab parametrization f(t) = g(t) * (1+alpha(t)*(four series))
                # with g(t) a linear guess, alpha(t) a sine s.t alpha(0) = alpha(T) = 0
                # and the four series used randomized frequencies
                nb_params = int(shortcut[12:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                            'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#rfour)))))'}
                        
                        
            ##########  WITH FREE OMEGAS  ##########
            elif(shortcut[:15] == 'owbds01_Om4crab'):
                # AANOTHER Custom Crab parametrization f(t) = g(t) + erf((sin four series)))
                nb_params = int(shortcut[15:])
                k = 4 /nb_params
                x0 = '0.1*T'
                k2 = '60/T'          
                L = '1'
                if(nb_params % 2 != 0):
                    SystemError('nb_params = {} while it should be 2n'.format(nb_params))
                nbH = int(nb_params/2)
                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone, 'mask':logisflex%(L,k2,x0),
                             'logis': logis%(str(k)), 'sinfour':omsinfour%('(-1,1)', nb_params-1)}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,*(#mask,**(+(#logis,#ctm),#sinfour)))))'}


            elif(shortcut[:14] == 'owbds01_Omcrab'):
                # f(t) = g(t) * (1+alpha(t)*(four series))
                nb_params = int(shortcut[14:])
                if(nb_params % 3 != 0):
                    SystemError('nb_params = {} while it should be 3n'.format(nb_params))
                nbH = int(nb_params/3)
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                            'rfour':omfour%('(-1,1)', '(-1,1)', str(nbH))}
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#rfour)))))'}           
            

            ### NOMORERANDOMIZATION BUT GRBF INSTEAD
            elif(shortcut[:13] == 'owbds01_1grbf'):
                #pdb.set_trace()
                # (t) = g(t) * (1 + alpha(t)* RBF)
                nb_params = int(shortcut[13:])
                #RBF
                a_scale = np.sqrt(8* np.log(2))
                b_scale = np.sqrt(8* np.log(4))
                cc = (2 * b_scale + (nb_params - 1) * a_scale)
                sigma_str = 'T/' + str(cc)
                sigma = [sigma_str for _ in range(nb_params)]
                l = '[' + ",".join(sigma) + "]"
                A = str([0.0 for _ in range(nb_params)]) #np.repeat(1, nb_P)
                x0_list = [str(b_scale) +'*'+ sigma_str + "+" + str(a_scale) + "*" + sigma_str + "*" + str(p) for p in np.arange(nb_params)]  
                x0 = "[" + ",".join(x0_list)+"]"
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                             'ctm':mone, 'grbf':grbf%(A, x0, l, (-1,1))}
                
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#grbf)))))'}


            ### W/O RANDOMIZED FREQ    
            elif(shortcut[:14] == 'owbds01_crfour'):
                # Crab parametrization w/o randomized freq
                nb_params = int(shortcut[14:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                            'four':four%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#four)))))'}
             
            elif(shortcut[:15] == 'owbds01_Ccrfour'):
                # Custom Crab parametrization w/o randomized freq
                nb_params = int(shortcut[15:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                k = 4 /nb_params
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                             'powscale':pow15, 'ctm':mone,'logis': logis%(str(k)),
                             'four':four%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                
                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(**(#powscale,#scale),**(+(#logis,#ctm),#four))))))'}
                        
            
            elif(shortcut[:14] == 'owbds01_trevfour'):
                #trend and fourier (sine part only)
                nb_params = int(shortcut[14:])
                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'sinfour':sinfour%('(-1,1)', nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,#sinfour)))'}
                        
                        
            elif(shortcut[:13] == 'owbds01_trsin'):
                #f(t) = g(t) + erf((sin four series)))
                nb_params = int(shortcut[13:])
                k = 4 /nb_params
                x0 = '0.1*T'
                k2 = '60/T'          
                L = '1'
                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone,
                             'logis': logis%(str(k)), 'sinfour':sinfour%('(-1,1)', nb_params)}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,**(+(#logis,#ctm),#sinfour))))'}
            
            
            # LINEAR RAMP
            elif(shortcut[:14] == 'owbds01_linear'):
                dico_atom = {'ow':ow,'bd':bds,'lin':linear}
                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
            
            #LINEAR INVERTED
            elif(shortcut[:16] == 'owbds01r_linearr'):
                dico_atom = {'ow':ow_r,'bd':bds,'lin':linear_r}
                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
            
            elif(shortcut[:14] == 'owbds01_wrfour'):
                #wrapped fourier
                dico_atom = {'ow':ow,'bd':bds,'lin':linear}
                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
                        
            elif(shortcut[:13] == 'owbds01_cfred'):
                # Custom parametrization f(t) = g(t)  + alpha(t)* erf((four series)))
                # slightly different from the normal one (cf. before)
                # additional erf function (logistic function such that the four 
                # series is bounded) alpha(t) = sine ** 0.5
                nb_params = int(shortcut[13:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                k = 4 /nb_params
                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
                             'powscale':sqrt, 'ctm':mone,'logis': logis%(str(k)), 'half':half,
                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
                
                dico_expr = {'final':'**(#ow,**(#bd,+(#guess,*(*(#half,**(#powscale,#scale)),**(+(#logis,#ctm),#rfour)))))'}
                        
            
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
