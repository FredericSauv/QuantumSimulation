# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
#from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel
import csv, os, pathlib, pdb, copy, sys, time
from ast import literal_eval as ev
import matplotlib.pylab as plt
import numpy as np
import logging 
logger = logging.getLogger(__name__)
def log_uncaught_exceptions(*exc_info): 
    logging.critical('Unhandled exception:', exc_info=exc_info) 
sys.excepthook = log_uncaught_exceptions

if __name__ == '__main__':
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import pFunc_base, pFunc_zoo, Learner
    from QuantumSimulation.Utility.Misc import RandomGenerator, MP

else:
    from .. import Helper as ut
    from . import pFunc_base, pFunc_zoo, Learner
    from ..Misc import RandomGenerator, MP


#==============================================================================
#                   BATCH CLASS
# Generate a batch of simulations based on a text file containing the configurations
# on which run the procedure and SAVE the results
#
# Methods:
#  - parse_input_file: 
#  - run_procedures: ok (call wrap_proceduresr)
#  - save_res:
#  - read_res: read file stored in 
#
# NEW (19Apr)    
#    - Rdmruns / start to prepare to accomodate for different input type
#    -
# ThinkAbout:
#  - Management of random generator state // seeds number are gen but not used 
#  - remove procToRun
#==============================================================================
class Batch:
    """
    Batch:
        * parse an input file to create a list of configurations to run
        * run it According to some procedures (funcProc)
        * save results 
    """
    UNDERLYING_MODEL_CONSTRUCTOR = None ## TO IMPLEMENT IN THE SUBCLASSES
    EMPTY_LINE = [[], None]
    LEX_NA = list(['(*', ''])
    _DEF_RES_NAME = "res_default"
    
    # METAPARAMETERS OF THE BATCH
    # INFO {'Name':(def_value, type)}, type is not used atm
    METAPARAMS_FIRST_CHAR = "_"
    METAPARAMS_INFO = {'_RDM_RUNS': (1, 'int'), 
                       '_RDM_FIXSEED': (False, 'bool'),
                      '_OUT_PREFIX': ('res_','str'),
                      '_OUT_FOLDER': (None, 'str'),
                      '_OUT_COUNTER': (None, 'int'), 
                      '_OUT_NAME': (None, 'list'),
                      '_OUT_STORE_CONFIG': (True, 'bool'), 
                      '_MP_FLAG':(False, 'bool'),
                      '_WALL_TIME': (None, 'float')}
    METAPARAMS_NAME = METAPARAMS_INFO.keys()
    

    def __init__(self, config_object = None, rdm_object = None, debug = False):
        """ Init the batch with a <dic> (it represents one config) OR <list<dico>>
        (several configs) OR <str:file_path> (onefile = one config) OR
        <list<str:file_path>> (several configs)
        """
        if(debug):  pdb.set_trace()
        self.listConfigs = self._read_configs(config_object) 


    @classmethod
    def from_meta_config(cls, metaFile = None, rdm_object = None, procToRun = None, 
                         debug = False, extra_processing = False, update_rules = False):
        """ init the config with a meta_configuration where a metaconfig is 
        understood as a file from which can be generated a list of configs.
        If extra_processing = True it will use some treatment to create the configsRandomGenerator.
        (it should be implemented in _dispatch_configs implemented)
        
        """
        if(debug): pdb.set_trace()
        list_configs = cls.parse_meta_config(metaFile, extra_processing = extra_processing, 
                                             update_rules = update_rules)
        obj = Batch(list_configs, rdm_object, procToRun)
        return obj

# ---------------------------
# MAIN FUNCTIONS
#   - run_procedures
#   - save_res
# ---------------------------
    @classmethod
    def _build_underlying_model(cls, model_dico):
        """ to be implemented in the subclass """
        if(cls.UNDERLYING_MODEL_CONSTRUCTOR is None):
            raise NotImplementedError()
        else:
            model = cls.UNDERLYING_MODEL_CONSTRUCTOR(**model_dico)
        return model

    def _build_control(self,model_dico):
        """ to be implemented in the subclass """
        pass

    def run_procedures(self, config = None, saveFreq = 0, splitRes = False, printInfo = False, debug=False):
        """ For each element of listConfigs it will run one procedure (via 
        self.run_one_procedure which should be implemented) and save the reults
        
        Arguments:
        + saveFreq: (-1) save at the end / (0) don't save anything/ 
                    (1) save each time / (N)save each N res
        +splitRes: (False) only one file /(True):generate a file for each simulations 
        """
        if(debug): pdb.set_trace()
        list_configs = self.listConfigs if(config is None) else self._read_configs(config) 

        self.listRes = []
        listConfTmp = []
        listResTmp = []
        i_config = 0

        for conf in list_configs:
            _time_zero = time.time()
            tmp = self.run_one_procedure(conf) # run the simulation for one config
            tmp['time_elapsed'] = time.time() - _time_zero
            i_config +=1
            if(printInfo): logger.info(i_config)
            listResTmp.append(tmp)
            listConfTmp.append(conf)
            
            if ((saveFreq > 0) and (i_config%saveFreq == 0)):
                self.save_res(listResTmp, listConfTmp, splitRes)
                listResTmp = []
                listConfTmp = []
                        
        #Save all configs in one (separate) file maybe???
        if((saveFreq != 0) and (len(listResTmp) !=0)):
            self.save_res(listResTmp, listConfTmp, splitRes)
  

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


    def save_res(self, resToStore = None, confToStore=None, split = False):
        """ Store res/configs as a (several) text files
        Arguments:
            + resRoStore - a LIST[dicoRes] where dicoRes is a dictionnary containing the results
                    e.g. dicoRes = {'name': 'test1', 'optimCoeffs':list([a,b])}
                             / None (use self.listRes)
            + confToStore - a LIST[configs] / None (use self.listConfigs)
                            / False(don't save the configs in this case)
            + split - {True/False}: generate a file for each simulations or only one file
            + prefix - a string, to add before the name of the result
            + folder - {None, string}: {will be stored in the current folder, 
                folder in which it will be stored}
            + counter - if None nothing /else use the value to start this counter 
                            incremented at each write
        """
        if(resToStore is None):
            resToStore = self.listRes 
        if(confToStore is None):
            confToStore = self.listConfigs         
            
        if(split):
             for i in range(len(resToStore)):
                 oneRes = resToStore[i]
                 if(confToStore in [None, []]):
                     raise NotImplementedError()
                 else:
                     oneConf = confToStore[i]
                     folder = oneConf.get('_OUT_FOLDER', None)
                     name_tmp = oneConf.get('_RES_NAME')
                 oneRes['config'] = oneConf 
                 self.write_one_res(oneRes, name_tmp, folder=folder)

        else:
            resAll = ut.concat_dico(resToStore)
            if(isinstance(confToStore, list)):
                confAll = ut.concat_dico(confToStore)
            else:
                confAll = None
            #TODO: Probably to think about
            name_tmp = self._gen_name_res(confAll)
            
            self.write_one_res(resAll, name_tmp, folder)
            self.write_one_res(confAll, 'configs_' + name_tmp, folder)
                
# ---------------------------
# Dealing with the generation of configs and reading config_object
# ---------------------------
    @classmethod
    def _read_configs(cls, config_object, list_output = True):
        """ Allow for different type of config input (dico, list of dicos, fileName, 
        list of file Names)
        """
        if(ut.is_dico(config_object)):
            if(list_output):
                configs = [config_object]
            else:
                configs = config_object

        elif(ut.is_list(config_object)):
            configs = [cls._read_configs(conf, list_output = False) for conf in config_object]

        elif(ut.is_str(config_object)):
            configs = ut.file_to_dico(config_object)
            if(list_output):
                configs = [configs]
        
        else:
            raise NotImplementedError()

        return configs
    
    @classmethod
    def parse_and_save_meta_config(cls, input_file = 'inputfile.txt', 
            output_folder = 'Config', extra_processing = False, 
            update_rules=False, debug = False):
        """ parse an input file containing a meta-config generate the differnt 
        configs and write a file for each of them"""
        list_configs = cls.parse_meta_config(input_file, extra_processing, update_rules, debug=debug)
        for conf in list_configs:
            name_conf = 'config_' + conf['_RES_NAME']
            if(not(output_folder is None)):
                pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 
                name_conf = os.path.join(output_folder, name_conf)
            ut.dico_to_text_rep(conf, fileName = name_conf, typeWrite = 'w')

    @classmethod
    def parse_meta_config(cls, inputFile = 'inputfile.txt', extra_processing = False, 
                          update_rules = False, debug = False):
        """Parse an input file containing a meta-config and generate a 
        <list<dict:config>> where config is a dict containing a configuration 
        which can be directly used to run a procedure
        extra_processing >>
        update_rules >> for a key with several elements the first one serve as
        a reference and the other elements contain only updates of the reference 
        (works only if elements are dict)
        
        TODO: better docstring
        TODO:JSON files
        """
        if(debug):
            pdb.set_trace()
        use_context = False
        with open(inputFile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ' ')
            list_values = list([])
            list_keys = list([])
            nbline = 0
            dico_METAPARAMS = {}
            for line in reader:
                nbline += 1
                if(line in cls.EMPTY_LINE) or (line[0] in cls.LEX_NA):
                    pass
                #New feature to use a context for the evaluation of the subsequent line
                #
                elif(line[0] == '_CONTEXT'):
                    context = ev(line[1])
                    if(ut.is_dico(context) and np.product([c[0] == '_' for c in context])):
                        use_context = True
                elif(line[0] in cls.METAPARAMS_NAME):
                    
                    assert(len(line) == 2), 'batch input file: 1 arg expected in l.' + str(nbline)+ ' (' +str(line[0]) + ')'
                    dico_METAPARAMS[line[0]] = ev(line[1])                                    
                else:
                    assert (len(line)>=2), 'batch input file: not enough args in l.' + str(nbline) 
                    list_keys.append(line[0])
                    
                    if(use_context):
                        ### use eval instead of ast.literal_eval
                        line_with_context = [apply_context(line[i], context) for i in range(1,len(line))]
                        ev_tmp = [eval(lwc) for lwc in line_with_context]
                    else:
                        ev_tmp = [ev(line[i]) for i in range(1,len(line))]

                    if(update_rules):
                        ref_value = copy.copy(ev_tmp[0])
                        ev_tmp_new = []
                        for ev_i in ev_tmp:
                            in_progress = ut.merge_dico(ref_value, ev_i, update_type = 0, copy = True)
                            ev_tmp_new.append(copy.copy(in_progress))
                        ev_tmp = ev_tmp_new
                    list_values = ut.cartesianProduct(list_values, ev_tmp, ut.appendList) 

            list_configs = [ut.lists2dico(list_keys, l) for l in list_values]           
            list_configs = cls._apply_metaparams(list_configs, dico_METAPARAMS)

        if(extra_processing):
            list_configs = [cls._processing_meta_configs(c) for c in list_configs]

        return list_configs


    @classmethod        
    def _apply_metaparams(cls, list_configs, dico_meta):
        """ deal with randomness (start with _RDM) // management of the output
        (start with OUT)
        """
        #Parse Metaparameters cast them to the right type and use def value
        dico_meta_filled = cls._parse_metaparams(dico_meta)
        #MANAGEMENT OF RANDOM RUNS
        nb_rdm_run = dico_meta_filled.get('_RDM_RUNS', 1)
        fix_seed = dico_meta_filled.get('_RDM_FIXSEED', False)
        runs = range(nb_rdm_run)
        if(fix_seed):
            seeds = RandomGenerator.RandomGenerator.gen_seed(nb = nb_rdm_run)
            if(nb_rdm_run == 1):
                seeds = [seeds]
        else:
            seeds = [None for _ in runs]
        random_bits = [{'_RDM_RUN':n, '_RDM_SEED': seeds[n]} for n in runs]
        for conf in list_configs:
            conf['_ID_DET'] = RandomGenerator.RandomGenerator.gen_seed()
        list_configs = ut.cartesianProduct(list_configs, random_bits, ut.add_dico)
        
        #GENERATE RESULTS NAME AND INCLUDE THEM IN CONFIGS DICO
        name_res_list = []
        for conf in list_configs:
            name_res_tmp, dico_meta_filled = cls._gen_name_res(conf, dico_meta_filled)
            conf['_RES_NAME'] = name_res_tmp
            assert (name_res_tmp not in name_res_list), 'conflicts in the name res'
            name_res_list.append(name_res_tmp)
            conf['_OUT_FOLDER'] = dico_meta_filled['_OUT_FOLDER']
            conf['_OUT_STORE_CONFIG'] = dico_meta_filled['_OUT_STORE_CONFIG']
            conf['_MP_FLAG'] = dico_meta_filled['_MP_FLAG']
        return list_configs
    
    @classmethod
    def _parse_metaparams(cls, dico):
        """ fill the dico with default values if key is not found
        Either to move to helper or add more rules (if needed)
        """
        dico_parsed = {}
        for k, v in cls.METAPARAMS_INFO.items():
            def_val, _ = v
            val_tmp = dico.get(k, def_val)
            dico_parsed[k] = val_tmp
        return dico_parsed
            

    @classmethod
    def _gen_name_res(cls, config, metadico = {}, type_output = '.txt'):
        """ Generate the name of the simulation for saving purposes
        Ad-hoc can be changed reimplemented etc...
        Should be carefull as results will be overwritten
        """
        res_name = ''
        if(metadico.get('_OUT_NAME') is not None):
            name_rules = metadico['_OUT_NAME']
            if(ut.is_dico(name_rules)):
                for k, v in name_rules.items():
                    res_name += (k + "_" + config[k][v])
            else:
                raise NotImplementedError()
                
            if((config.get("_RDM_RUN", None) is not None)):
                res_name += "_" 
                res_name += str(config["_RDM_RUN"])
            
        elif(metadico.get('_OUT_COUNTER') is not None):
            res_name += str(metadico['_OUT_COUNTER'])
            metadico['_OUT_COUNTER'] +=1
        
        if(res_name == ''):
            #just in case it will generate a long random number
            res_name = str(RandomGenerator.RandomGenerator.gen_seed()) 
        
        prefix = metadico.get('_OUT_PREFIX', '')
        res_name = prefix + res_name + type_output

        return res_name, metadico
        
# ---------------------------
#   WRITTING RESULTS
# ---------------------------
    @classmethod
    def write_one_res(cls, resToWrite, forceName = None, folder = None):
        """ Write a res as a text file
        Parameters:
            + resToWrite - a dictionnary
            + folder - in which folder to write the results
            + forceName - Force the name given to the file (if None look for 
                         a key name in the results, if none use default name)
        """
        #Create name of the file
        if(forceName is None):
            if("_RES_NAME" in resToWrite):
                name = resToWrite['_RES_NAME']
            else:
                name = cls._DEF_RES_NAME
        else:
            name = forceName

        if(not(folder is None)):
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
            name = os.path.join(folder, name)     
        
        #Write
        ut.dico_to_text_rep(resToWrite, fileName = name, typeWrite = 'w')
    
    # ---------------------------
    #    DEAL WITH RESULTS
    #    part 0: MAIN FUNCTIONS
    # ---------------------------

    @classmethod
    def collect_and_process_res(cls, key_path = [], nameFile = None, allPrefix = 'res_', 
                                folderName = None, printing = False, ideal_evol = False):
        """ Collect a list of res in **folderName** strating with **allPrefix**
        group them by configurations specified by **key_path** """
        collection = cls.collect_res(key_path, nameFile, allPrefix, folderName)
        collection_stats = cls._process_collection_res(collection, printing, ideal_evol)
        
        return collection_stats


    # ---------------------------
    #    DEAL WITH RESULTS
    #    part 1: COLLECTING RESULTS
    # ---------------------------
    @classmethod
    def eval_from_onefile(cls, name):
        """ eval the first element of the first line of a file """
        res = ut.eval_from_file(name, evfunc = pFunc_base.eval_with_pFunc)
        return res
    

    @classmethod
    def read_res(cls, nameFile = None, allPrefix = 'res_', folderName = None):
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
        results = [Batch.eval_from_onefile(f) for f in listFileName]
        #results = [ut.file_to_dico(f, evfunc = (lambda x: eval(x)) ) for f in listFileName]        
        return results

    @classmethod
    def collect_res(cls, key_path = [], nameFile = None, allPrefix = 'res_', folderName = None):
        """Extract results stored in (a) txt file(s) and group them according to 
        some key values (where key_path provides the path in the potentially 
        nested structure of the results to find the key(s))
        
        Output:
            a dictionary where key is the concatenation of the unique set of 
            keys found and value is the res is a list of all res matching this key
        """
        listRes = cls.read_res(nameFile, allPrefix, folderName)
        res_keys = [tuple([ut.extract_from_nested(res, k) for k in key_path]) 
                    for res in listRes]
        res_keys_unique = list(set(res_keys))
        res = {ut.concat2String(*k_u):[listRes[n] for n, r in enumerate(res_keys) 
                if r == k_u] for k_u in res_keys_unique}
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
               
def apply_context(string, dico_context):
    for k,v in dico_context.items():
        string = string.replace(k, str(v))
    return string
        
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


#==============================================================================
#                   Testing
# 
#==============================================================================
if __name__ == "__main__": 
    import numpy.random as rdm
    test1 = False #
    test2 = False
    test3 = True
    
    if(test1):
        batchTest = Batch.from_meta_config('test_batch_1.txt', debug = True)
        def funcDummy(x):
            rdstate = rdm.RandomState(x.get('_RDM_SEED'))
            return {'res': rdstate.uniform(size=10)}
        batchTest.attach_proc(funcDummy)
        batchTest.run_procedures(config = None, saveFreq = 1, splitRes = True, printInfo = False)
        
    if(test2):
        Batch.parse_and_save_meta_config('test_batch_1.txt', output_folder = 'TestConfig')
        batchTest = Batch('TestConfig/config_res_algo_den_mot_a3_2.txt')
        def funcDummy(x):
            rdstate = rdm.RandomState(x.get('_RDM_SEED'))
            return {'res': rdstate.uniform(size=10)}
        batchTest.attach_proc(funcDummy)
        batchTest.run_procedures(saveFreq = 1, splitRes = True)
    

    if(test3):
        Batch.parse_and_save_meta_config('test_meta_config.txt', output_folder = '_Test', debug = True)