# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import csv 
import os
import pathlib
import pdb
from ast import literal_eval as ev
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel


if __name__ == '__main__':
    import sys
    sys.path.append("../../")
    from Utility import Helper as ut
    from Utility.Optim import RandomGenerator as rdgen
    from Utility.Optim.ParametrizedFunctionFactory import *    
else:
    from .. import Helper as ut
    from . import RandomGenerator as rdgen
    from .ParametrizedFunctionFactory import * 
    
import importlib as ilib
ilib.reload(ut)

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
#  -
#==============================================================================
class Batch:
    """
    Batch:
        * parse an input file to create a list of configurations to run
        * run it According to some procedures (funcProc)
        * save results 
    """
    EMPTY_LINE = [[], None]
    LEX_NA = list(['(*', ''])
    _DEF_RES_NAME = "res_default"
    
    # METAPARAMETERS OF THE BATCH
    # INFO {'Name':(def_value, type)}.
    # type is not used atm
    METAPARAMS_FIRST_CHAR = "_"
    METAPARAMS_INFO = {'_RDM_RUNS': (1, 'int'), '_RDM_FIXSEED': (False, 'bool'),
                      '_OUT_PREFIX': ('res_','str'),'_OUT_FOLDER': (None, 'str'),
                      '_OUT_COUNTER': (None, 'int'), '_OUT_NAME': (None, 'list'),
                      '_OUT_STORE_CONFIG': (True, 'bool')}
    METAPARAMS_NAME = METAPARAMS_INFO.keys()
    
    def __init__(self, config_object = None, rdm_object = None, procToRun = None, debug = False):
        """ Init the batch with a config(dico, list of dicos, file or list of files containing
        the configs )
        """
        if(debug):
            pdb.set_trace()
        self._RDMGEN = rdgen.RandomGenerator.init_random_generator(rdm_object) # not used atm
        self.listConfigs = self._read_configs(config_object) 
        self._init_proc_to_run(procToRun)


    @classmethod
    def from_meta_config(cls, metaFile = None, rdm_object = None, procToRun = None, debug = False):
        """ init the config with a meta_configuration (instead of directly a
        /several configs). metaconfig is a file on which can be generated 
        list of configs
        """
        if(debug):
            pdb.set_trace()
        list_configs = cls.parse_meta_config(metaFile)
        obj = Batch(list_configs, rdm_object, procToRun)
        return obj

    @classmethod
    def from_meta_with_dispatch(cls, inputFile = None, rdm_object = None, procToRun = None, debug = False):
        """ Init the Batch object from a meta-config file and some extra processing
        of the file (._dispatch_configs), it should be implemented in the subclass
        or add a mechanism to hook it up
        """
        obj = cls.from_meta_config(inputFile, rdm_object, procToRun, debug)
        obj.listConfigs = [obj._dispatch_configs(conf) for conf in obj.listConfigs]
        return obj

    def _dispatch_configs(self, dico):
        """ 
        """
        raise NotImplementedError()

# ---------------------------
# MAIN FUNCTIONS
#   - run_procedures
#   - save_res
# ---------------------------
    def run_procedures(self, config = None, saveFreq = 0, splitRes = False, printInfo = False):
        """ For each element of listConfigs run procedures (procToRun) and save reults
        Arguments:
        saveFreq:
            (-1) save at the end / (0) don't save anything/ (1) save each time 
            / (N)save each N res
        splitRes
             (False) only one file /(True):generate a file for each simulations 
             TODO: add gather all the runs for the same setup
        """
        #pdb.set_trace()
        if(config is None):
            list_configs = self.listConfigs
        else:
            list_configs = self._read_configs(config)

        self.listRes = []
        listConfTmp = []
        listResTmp = []
        i_config = 0

        for conf in list_configs:
            #conf_filt = ut.filter_dico_first_char(conf, self.METAPARAMS_FIRST_CHAR, keep = False)
            tmp = self.run_one_procedure(conf) # run the simulation for one config
            i_config +=1
            if(printInfo):
                print(i_config)
            #self.listRes.append(tmp)
            listResTmp.append(tmp)
            #if(conf['_OUT_STORE_CONFIG']): 
            listConfTmp.append(conf)
            
            if ((saveFreq > 0) and (i_config%saveFreq == 0)):
                self.save_res(listResTmp, listConfTmp, splitRes)
                listResTmp = []
                listConfTmp = []
                        
        #Save all configs in one (separate) file maybe???
        if((saveFreq != 0) and (len(listResTmp) !=0)):
            self.save_res(listResTmp, listConfTmp, splitRes)
  
    
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
    def parse_and_save_meta_config(cls, input_file = 'inputfile.txt', output_folder = 'Config'):
        """ parse an input file generate teh configs and write a file for each of them
        """
        list_configs = cls.parse_meta_config(input_file)
        for conf in list_configs:
            name_conf = 'config_' + conf['_RES_NAME']
            if(not(output_folder is None)):
                pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 
                name_conf = os.path.join(output_folder, name_conf)
            ut.dico_to_text_rep(conf, fileName = name_conf, typeWrite = 'w')

    @classmethod
    def parse_meta_config(cls, inputFile = 'inputfile.txt'):
        """Parse an input file and generate a list[dicoConfigs] where dicoConfigs are 
        dictionaries containing a configuration which can be directly used to run the proc
        """
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
                elif(line[0] in cls.METAPARAMS_NAME):
                    assert (len(line) == 2), 'batch input file: 1 arg expected in l.' + str(nbline)+ ' (' +str(line[0]) + ')'
                    dico_METAPARAMS[line[0]] = ev(line[1])                                    
                else:
                    assert (len(line)>=2), 'batch input file: not enough args in l.' + str(nbline) 
                    list_keys.append(line[0]) 
                    ev_tmp = [ev(line[i]) for i in range(1,len(line))]
                    list_values = ut.cartesianProduct(list_values, ev_tmp, ut.appendList) 

            list_configs = [ut.lists2dico(list_keys, l) for l in list_values]           
            list_configs = cls._apply_metaparams(list_configs, dico_METAPARAMS)

        return list_configs


    @classmethod        
    def _apply_metaparams(cls, list_configs, dico_meta):
        """ deal with randomness (start with _RDM) // management of the output
        (start with OUT)
        """
        #Parse Metaparameters cast them to the right type and use def value
        dico_meta_filled = cls._parse_metaparams(dico_meta)
                     
        #MANAGEMENT OF RANDOM RUNS
        nb_rdm_run, fix_seed = dico_meta_filled['_RDM_RUNS'], dico_meta_filled['_RDM_FIXSEED']
        if(nb_rdm_run > 1):
            nb_run = range(nb_rdm_run)
            if(fix_seed):
                seeds = rdgen.RandomGenerator.gen_seed(nb = nb_rdm_run)
            else:
                seeds = [None for _ in range(nb_rdm_run)]
            random_bits = [{'_RDM_RUN':n, '_RDM_SEED': seeds[n]} for n in nb_run]
            for conf in list_configs:
                conf['_ID_DET'] = rdgen.RandomGenerator.gen_seed()
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
            res_name = str(rdgen.RandomGenerator.gen_seed()) 
        
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
#   COLLECTING RESULTS
# ---------------------------
    
    @classmethod
    def read_res(cls, nameFile = None, allPrefix = 'res_', folderName = None):
        """
        What it does:
            Extract results stored in a txt file and transform them in a (list) of dictionnaries
        Arguments:
            nameFile {None, string} = {just look for the file fileName, 
                    else look for the files starting by prefix in folderName}
        """
        listFileName = ut.findFile(nameFile, allPrefix, folderName)
        results = [ut.file_to_dico(f, evfunc = (lambda x: eval(x)) ) for f in listFileName]        
        return results

    @classmethod
    def collect_res(cls, requRes = [], params= {}, filters = {}, nameFile = None, allPrefix = 'res_', folderName = None):
        """Extract results stored in a txt files and rearange them as 
        Arguments:
            nameFile {None, string} = {just look for the file fileName, 
                    else look for the files starting by prefix in folderName}
        Output:
            a dictionary where each key is an entry in reqRes
            each value is requested Results for the parameters in
        """
        raise NotImplementedError()
        #listRes = cls.read_res(nameFile = None, allPrefix = 'res_', folderName = None)
        #listResProcessed = [ for k in resKeys]
        results = {}
        return results 

    def extractFromDico(dico, listRes = [], listParams = [], dicoConstraints = {}):
        raise NotImplementedError()
        
    
# ---------------------------
# To be implemented in the subclass
# ---------------------------
    def run_one_procedure(self, conf):
        # should take a config(dico) and return results (packed as a dico too)
        # results can contain 
        res = self._procToRun(conf)
        return res

    def _init_proc_to_run(self, proc = None):
        """If None return Id function
        """
        if proc is None:
            self._procToRun = ut.idFun1V # idfunction
        else:
            self.attach_proc(proc)
            
    def attach_proc(self, proc):
        """ Update the procedure to run
        """
        self._procToRun = proc
        



#==============================================================================
#                   Testing
# 
#==============================================================================
if __name__ == "__main__": 
    import numpy.random as rdm
    test1 = True #
    test2 = True
    
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
    
