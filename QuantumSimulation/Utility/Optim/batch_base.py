# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import logging
import sys
logger = logging.getLogger(__name__)
def log_uncaught_exceptions(*exc_info): 
    logging.critical('Unhandled exception:', exc_info=exc_info) 
sys.excepthook = log_uncaught_exceptions

import time
import csv
import os
import pathlib
import pdb
import copy
from ast import literal_eval
import numpy as np


if __name__ == '__main__':
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Misc import RandomGenerator
    from QuantumSimulation.Utility.Optim import pFunc_base, pFunc_zoo
else:
    from .. import Helper as ut
    from ..Misc import RandomGenerator
    from . import pFunc_base, pFunc_zoo

class BatchBase:
    """
    BatchBase serves the purpose of managing several runs of the same job 
    (called a procedure) with different configurations 
    It allows :
        - creating and saving configurations based on some meta configuration files
        - run jobs / save results based on these configurations (either individually
        or a bunch of them)
        - loading and collect results generated 

    Some constraints are put on the input output: config and the output of one procedure 
    are <dic>. config should contain a field with '_RES_NAME'
    
    MAIN FUNCTIONS
    ---------------
    run_procedures
    run_one_procedure
    save_res
    parse_and_save_meta_config
    parse_meta_config
    """


    # Syntax of the meta config file
    EMPTY_LINE = [[], None]
    LEX_NA = list(['(*', '', '#'])
    
    # meta parameters of the batch object
    _DEF_RES_NAME = "res_default"
    METAPARAMS_INFO = {'_RDM_RUNS': (1, 'int'), #Nb of random runs of each configuration  
        '_RDM_FIXSEED': (False, 'bool'), #Fix the random seed for different configurations
        '_OUT_PREFIX': ('res_','str'), #Prefix to append to the result names
        '_OUT_FOLDER': (None, 'str'), #Folder in which to put the results
        '_OUT_COUNTER': (None, 'int'), #Append a number to the name 
        '_OUT_NAME': (None, 'list'), #More a template to build a name based on config
        '_OUT_STORE_CONFIG': (True, 'bool'), #Store or not the config
        '_MP_FLAG':(False, 'bool')} #Use of multiprocessing
    METAPARAMS_NAME = METAPARAMS_INFO.keys()
    

    def __init__(self, config_object = None, debug = False):
        """ 

        Arguments
        ----------
        config_object: <dic> one config given as a dict
                       <list<dico>> several configs 
                       <str> file path containing one config 
                       <list<str:file_path>> list of file paths
        rdm_object:
                        a random state
        """
        if(debug): pdb.set_trace()
        self.listConfigs = self._read_configs(config_object) 


    @classmethod
    def from_meta_config(cls, meta_file = None, update_rules = False, debug = False):
        """ Initialize a Batch object from a meta_configuration. A meta-config is understood
        as a file from which can be generated a list of configs """
        if(debug): pdb.set_trace()
        list_configs = cls.parse_meta_config(meta_file, update_rules = update_rules)
        obj = BatchBase(list_configs)
        return obj



    def run_procedures(self, config = None, save_freq = 1, debug=False):
        """ For each element of self.listConfigs it will run one procedure (via 
        self.run_one_procedure which should be implemented) and save the reults
        
        Arguments:
        ----------
        config: <list<dic>> 
                Configs on which run the jobs. By default it will take the configs in  
                self.listConfigs
        save_freq: int
                 Frequency at which the results are saved 
                 (-1) save at the end (i.e. after running jobs on all the configs) 
                 (0) don't save anything 
                 (1) save each time (after running each config)
                 (N) save each N config
        """
        if(debug): pdb.set_trace()
        list_configs = self.listConfigs if(config is None) else self._read_configs(config) 

        conf_list_tmp = []
        res_list_tmp = []

        for i_config, conf in enumerate(list_configs):
            _time_zero = time.time()
            res_tmp = self.run_one_procedure(conf) 
            res_tmp['time_elapsed'] = time.time() - _time_zero
            logger.info("{0}th config ran ".format(i_config))
            res_list_tmp.append(res_tmp)
            conf_list_tmp.append(conf)
            
            if ((save_freq > 0) and (i_config % save_freq == 0)):
                self.save_res(res_list_tmp, conf_list_tmp)
                res_list_tmp = []
                conf_list_tmp = []
                        
        if((save_freq != 0) and (len(res_list_tmp) !=0)):
            self.save_res(res_list_tmp, conf_list_tmp)
  
    def run_one_procedure(self, config):
        """ Define what it means ro tun a job. Should be implemented in subclasses.
        It should take a config as an input an return a res <dic>"""
        raise NotImplementedError

    def save_res(self, list_res , list_configs):
        """ Save results and configuration associated as several text file(s)
        
        Arguments:
        ---------
            list_res: <list<dic>>
                each dic represents a result
            list_configs: <list<dic>>
                Each dic represents a configuration with the following keys:
                '_OUT_FOLDER': folder name to store the results () 
                '_RES_NAME': name of the result (IT HAS TO BE PART OF THE DICT)
            store_config: bool
                If True config is added in res

        When writing the file it uses repr()
        """ 
        for one_res, one_conf in zip(list_res, list_configs):
            folder = one_conf['_OUT_FOLDER']
            name_tmp = one_conf['_RES_NAME']
            store_config = one_conf['_OUT_STORE_CONFIG']
            if(store_config): one_res['config'] = one_conf 
            self.write_one_res(one_res, name_tmp, folder=folder)

    @classmethod
    def _read_configs(cls, config_object, list_output = True):
        """ Allow for different type of configs to be passed (<dic>, <list<dic>>, 
        <str>, <list<str>>)
        """
        if(ut.is_dico(config_object)):
            configs = [config_object] if(list_output) else config_object
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
    def parse_and_save_meta_config(cls, input_file='inputfile.txt', 
            output_folder='Config', update_rules=False, debug=False):
        """ Parse an input file containing a meta-configuration, generate the differnt 
        configs, and save them as files.

        Parameters
        ----------
            input_file: str
                where the file containing the meta configuration is
            output_folder: str, none
                In which folder to store the configs
            update_rules: bool
                rules to apply when generating the configs
            debug: bool
                debug mode
        """
        list_configs = cls.parse_meta_config(input_file, update_rules, debug=debug)
        for conf in list_configs:
            name_conf = 'config_' + conf['_RES_NAME']
            if(not(output_folder is None)):
                pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 
                name_conf = os.path.join(output_folder, name_conf)
            ut.dico_to_text_rep(conf, fileName = name_conf, typeWrite = 'w')


    @classmethod
    def parse_meta_config(cls, input_file = 'inputfile.txt', update_rules = False, debug = False):
        """ Parsing an input file containing a meta-config into a list of configs.
        It relies on generating the cartesian product of the elements 
        
        
        Example (txt file containing a meta config)
        -------
        ###
        _CONTEXT {'_T':5}
        key1 Val1 Val2
        key2 Val3
        ###
        --> [{key1: eval(Val1), key2: eval(Val3)}, {key1: eval(Val2), key2: eval(Val3)}]

        Syntax of the meta-config file:
        ---------------------------------
        Keys starting with a _ and in self.METAPARAMS_INFO.keys() are metaparameters
        of the parser. o.w. they will be keys of the final config dico.
        
        If the first line key = _CONTEXT the Val associated should be of the form 
        {'_contextvar1':val,'_contextvar2':val1} i.e. a dico with keys starting with a '_'          
        
        Arguments
        ---------
        input_file: str
            path of the meta-config file
        update_rules : 
            When building the configs should the first value seen (e.g. Val1) be used 
            as a reference. Only works if Vals are <dic>. If True exampel's output becomes: 
            [{key1: eval(Val1), key2: eval(Val3)}, {key1: eval(Val2).update(eval(Val2)), key2: eval(Val3)}]
        debug : bool
            debug mode

        Output
        ------
        list_configs_processing_meta_configs: <list<dic>>

        """
        if(debug): pdb.set_trace()
        
        use_context = False
        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ' ')
            list_values = list([])
            list_keys = list([])
            nbline = 0
            dico_METAPARAMS = {}
            for line in reader:
                nbline += 1
                if(line in cls.EMPTY_LINE) or (line[0] in cls.LEX_NA): pass

                elif(line[0] == '_CONTEXT'):
                    context = literal_eval(line[1])
                    if(ut.is_dico(context) and np.product([c[0] == '_' for c in context])):
                        use_context = True

                elif(line[0] in cls.METAPARAMS_NAME):
                    assert(len(line) == 2), 'batch input file: 1 arg expected in l.' + str(nbline)+ ' (' +str(line[0]) + ')'
                    dico_METAPARAMS[line[0]] = literal_eval(line[1])                                    

                else:
                    assert (len(line)>=2), 'batch input file: not enough args in l.' + str(nbline) 
                    list_keys.append(line[0])
                    if(use_context):
                        line_with_context = [cls._apply_context(line[i], context) for i in range(1,len(line))]
                        ev_tmp = [eval(lwc) for lwc in line_with_context]
                    else:
                        ev_tmp = [literal_eval(line[i]) for i in range(1,len(line))]

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
            list_configs = [cls._processing_meta_configs(c) for c in list_configs]


        return list_configs

    @classmethod
    def _processing_meta_configs(cls, dico):
        """ AdHoc processing rules when dealing with meta configs: 
        helps to create the controler 
        """
        return dico

    def _apply_context(string, dico_context):
        """ replace patterns in string based on dico_context """
        for k,v in dico_context.items():
            string = string.replace(k, str(v))
        return string

    @classmethod        
    def _apply_metaparams(cls, list_configs, dico_meta):
        """ Deal with genearting random runs, names of the simulation

        METAPARAMETERS: RANDOM RUNS
        ---------------------------
        '_RDM_RUNS': default = 1 type = 'int': 
            Number of random runs for each config 
        '_RDM_FIXSEED': default = False type =  'bool'
            If True all the configurations for the same random run will have 
            the same random seed (new key, value added in config '_RDM_SEED' != None)
        To keep track of which configurations map to the same initial configuration 
        (i.e. without taking care of rdm runs) a flag 'ID_DET' is added 
        

        METAPARAMETERS: OUTPUT
        ----------------------
        '_OUT_PREFIX': ('res_','str'),
        '_OUT_FOLDER': (None, 'str'),
        '_OUT_COUNTER': (None, 'int'), 
        '_OUT_NAME': (None, 'list'),
        '_OUT_STORE_CONFIG': (True, 'bool'),

         METAPARAMETERS: MISC
        ------------------------
        '_MP_FLAG':(False, 'bool')}
        
        """
        dico_meta_filled = cls._parse_metaparams(dico_meta)
        
        #Management of the random runs
        nb_rdm_run = dico_meta_filled['_RDM_RUNS']
        fix_seed = dico_meta_filled['_RDM_FIXSEED']
        runs = range(nb_rdm_run)
        if(fix_seed):
            seeds = RandomGenerator.RandomGenerator.gen_seed(nb = nb_rdm_run)
            if(nb_rdm_run == 1): seeds = [seeds]
        else:
            seeds = [None for _ in runs]
        random_bits = [{'_RDM_RUN':n, '_RDM_SEED': seeds[n]} for n in runs]
        for conf in list_configs:
            conf['_ID_DET'] = RandomGenerator.RandomGenerator.gen_seed()
        list_configs = ut.cartesianProduct(list_configs, random_bits, ut.add_dico)
        
        #Management of the output
        name_res_list = []
        for conf in list_configs:
            name_res_tmp, dico_meta_filled = cls._gen_name_res(conf, dico_meta_filled)
            conf['_RES_NAME'] = name_res_tmp
            assert (name_res_tmp not in name_res_list), "Duplicated name: {0}".format(name_res_tmp)
            name_res_list.append(name_res_tmp)
            conf['_OUT_FOLDER'] = dico_meta_filled['_OUT_FOLDER']
            conf['_OUT_STORE_CONFIG'] = dico_meta_filled['_OUT_STORE_CONFIG']
            conf['_MP_FLAG'] = dico_meta_filled['_MP_FLAG']
        return list_configs
    
    @classmethod
    def _parse_metaparams(cls, dico):
        """ fill the dico with default values if key is not found """
        dico_parsed = {}
        for k, v in cls.METAPARAMS_INFO.items():
            def_val, _ = v
            val_tmp = dico.get(k, def_val)
            dico_parsed[k] = val_tmp
        return dico_parsed
            

    @classmethod
    def _gen_name_res(cls, config, metadico = {}, type_output = '.txt'):
        """ Generate a name associated to a config

        Rules
        -----
        if _OUT_NAME is not None and has a <dic> type:
            {'k':v} --> 'k_XXX' where XXX has been found in the config structure
            following path given by v


        elif _OUT_COUNTER is not None increment this counter for each new config

        
        add _OUT_PREFIX

        """
        res_name = ''
        if(metadico.get('_OUT_NAME') is not None):
            name_rules = metadico['_OUT_NAME']
            if(ut.is_dico(name_rules)):
                for k, v in name_rules.items():
                    res_name += (k + "_" + str(ut.extract_from_nested(config, v)))
            else:
                raise NotImplementedError()
                
            if((config.get("_RDM_RUN") is not None)):
                res_name += "_" 
                res_name += str(config["_RDM_RUN"])
            
        elif(metadico.get('_OUT_COUNTER') is not None):
            res_name += str(metadico['_OUT_COUNTER'])
            metadico['_OUT_COUNTER'] +=1
        
        if(res_name == ''):
            res_name = str(RandomGenerator.RandomGenerator.gen_seed()) 
        
        prefix = metadico.get('_OUT_PREFIX', '')
        res_name = prefix + res_name + type_output

        return res_name, metadico
        
    # ---------------------------
    #   WRITTING RESULTS
    # ---------------------------
    @classmethod
    def write_one_res(cls, res, name_res = None, folder = None):
        """ Write a res as a text file. 

        Arguments:
            + res - a dictionnary
            + folder - in which folder to write the results
            + forceName - Force the name given to the file (if None look for 
                         a key name in the results, if none use default name)
        """
        #Create name of the file
        if(name_res is None):
            if("_RES_NAME" in res):
                name = res['_RES_NAME']
            else:
                name = cls._DEF_RES_NAME
        else:
            name = name_res

        if(not(folder is None)):
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
            name = os.path.join(folder, name)     
        
        #Write
        ut.dico_to_text_rep(res, fileName = name, typeWrite = 'w')
    

    # ---------------------------
    #    DEAL WITH RESULTS
    # ---------------------------
    @classmethod
    def collect_and_process_res(cls, key_path = [], nameFile = None, allPrefix = 'res_', 
                                folderName = None, replace_func=None, **args):
        """ Collect and process a list of results """
        collection = cls.collect_res(key_path, nameFile, allPrefix, folderName, 
                                     replace_func=replace_func)
        collection_stats = cls._process_collection_res(collection, **args)
        
        return collection_stats

    @classmethod
    def _process_collection_res(cls, collection_res, **xargs):
        """ Process the collected res. As it is custom to the type of results
        it has to be implemented in the subclasses"""
        raise NotImplementedError()


    @classmethod
    def eval_from_onefile(cls, name, replace_func = None):
        """ Get results from a file. Open a file and evaluate (with eval) its first element"""
        res = ut.eval_from_file(name, evfunc = eval,replace_func = replace_func)
        return res
    

    @classmethod
    def read_res(cls, nameFile = None, allPrefix = 'res_', folderName = None, 
                 returnName = False, replace_func = None):
        """ Extract result(s) stored in a (several) txt file (s) and return them  
        as a (list) of evaluated objects

        Arguments
        ---------
        nameFile: None or <str>
            If not None only read the file with this name
            If None will rely on folderName and allPrefix 
        folderName: None or <str>
            In which folder should we look for the files
        allPrefix: None or <str>
            If not None enforce collecting only the results starting with this prefix
        returnName: bool
            if True returns also the name of the file(s)
        replace_func: <func> take the string extracted from the file and transform it

        Output
        ------
        results: <list>
            each element is the evaluated string contained i the relevant file


        """
        listFileName = ut.findFile(nameFile, allPrefix, folderName)
        results = [cls.eval_from_onefile(f, replace_func = replace_func) for f in listFileName]
        filter_none = [r is None for r in results]
        if np.any(filter_none):
            logger.warning('{} empty files'.format(np.sum(filter_none)))
            results = [r for r, f in zip(results, filter_none) if not(f) ]
        if returnName:
            return results, listFileName
        else:
            return results

    @classmethod
    def collect_res(cls, key_path = [], nameFile = None, allPrefix = 'res_', 
                    folderName = None, replace_func = None):
        """ collect results and group them according to some key values 

        Arguments
        ---------
        key_path: <list>
            if not empty defines how to group results: provides a path in the 
            dict structure of the results to get a value used for the grouping 
        nameFile, allPrefix, folderName: cf. read_res()
        
        Output:
        -------
            a dictionary where key is the concatenation of the unique set of 
            keys found and value is the res is a list of all res matching this key
        """
        listRes, listNames = cls.read_res(nameFile, allPrefix, folderName, 
                                    returnName=True, replace_func=replace_func)
        if(len(key_path) == 0):
            res = {k:v for k,v in zip(listNames, listRes)}

        else:
            res_keys = [tuple([ut.extract_from_nested(res, k) for k in key_path]) 
                        for res in listRes]
            res_keys_unique = list(set(res_keys))
            res = {ut.concat2String(*k_u):[listRes[n] for n, r in enumerate(res_keys) 
                if r == k_u] for k_u in res_keys_unique}
        return res



#==============================================================================
#                   BATCH CLASS
# Add some ad-hoc capabilities to deal with parametrized functions for the 
# controls of the model
#==============================================================================
class BatchBaseParamControl(BatchBase):
    """
    Batch:
        * parse an input file to create a list of configurations to run
        * run it According to some procedures (funcProc)
        * save results 
    """
    def __init__(self, config_object = None, debug = False):
        """ Init the batch with a <dic> (it represents one config) OR <list<dico>>
        (several configs) OR <str:file_path> (onefile = one config) OR
        <list<str:file_path>> (several configs)
        """
        super().__init__(config_object, debug)

# ---------------------------
#   ADHOC METHODS TO DEAL WITH THE PARAMETRIZED FUNCTIONS
# ---------------------------    
    def _build_control(self, model_dico):
        """ if the control_object is a string evaluate it if not do nothing"""
        control = model_dico.get('control_obj', None)
        if(ut.is_str(control)):
            model_dico['control_obj'] = type(self)._build_control_from_string(control,
                      None, model_dico)


    @classmethod
    def _build_control_from_string(cls, control, random_gen=None, context_dico = None):
        if(context_dico is not None):
            context = {k:v for k, v in context_dico.items() if k not in 
                       ['control_obj', 'random_obj', 'mp_obj']}
        else:
            context = None
    
        #if(random_gen is None):
        #   random_gen = RandomGenerator.RandomGenerator()
        
        func_factory = pFunc_zoo.pFunc_factory(rdm_obj=None, context=context)
        built_control = func_factory.eval_string(control)
        return built_control
    
        
    @classmethod
    def _processing_meta_configs(cls, dico):
        """ AdHoc processing rules when dealing with meta configs: 
        helps to create the controler    
        """
        np.random.seed(dico.get('_RDM_SEED', None))
        dico_new = {k: cls._process_controler(v) for k, v in dico.items()}
        return dico_new

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
        if type(dico) != dict:
            return dico
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
            intquad = "{'name_func':'InterpQuad','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}"
            intcub = "{'name_func':'InterpCub','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}"
            
            
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

            elif(shortcut[:15] == 'owbds01_intquad'):
                nb_params = int(shortcut[15:])
                dico_atom = {'ow':ow,'bd':bds,'pwl':intquad %('(0,1)',nb_params+1)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwl))'}
                        
            elif(shortcut[:14] == 'owbds01_intcub'):
                nb_params = int(shortcut[14:])
                dico_atom = {'ow':ow,'bd':bds,'pwl':intcub %('(0,1)',nb_params+1)}
                dico_expr = {'final':'**(#ow,**(#bd,#pwl))'}

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

    @classmethod
    def eval_from_onefile(cls, name, replace_func = None):
        """ Get results from a file. Open a file and evaluate (with eval) its first element"""
        res = ut.eval_from_file(name, evfunc = pFunc_base.eval_with_pFunc,replace_func = replace_func)
        return res
    

#==============================================================================
#                   Testing
# 
#==============================================================================
if __name__ == "__main__": 
    import numpy.random as rdm
    test1 = False #
    
    class BatchDummy(BatchBaseParamControl):
        def run_one_procedure(self, config):
            rdstate = rdm.RandomState(config.get('_RDM_SEED'))
            return {'res': rdstate.uniform(size=10)}
    
    
    if(test1):
        BatchDummy.parse_and_save_meta_config(
                input_file = '_test_meta_config.txt', update_rules = True,
                output_folder = 'TestConfig')
        batchTest = BatchDummy('TestConfig/config_res_0.txt')
        batchTest.run_procedures(save_freq = 1)
    
