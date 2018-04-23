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

if __name__ == '__main__':
    import sys
    sys.path.append("../../")
    from Utility import Helper as ut
    from Utility.Optim import RandomGenerator as rdgen
    
else:
    from .. import Helper as ut
    from . import RandomGenerator as rdgen
    
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
#  -
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
    DEF_RES_NAME = "default"
    
    # METAPARAMETERS INFO {'Name':(def_value, type)}.
    # bool is not used atm
    METAPARAMS_INFO = {'_RDM_RUNS': (1, 'int'), '_RDM_FIXSEED': (False, 'bool'),
                      '_OUT_PREFIX': ('res_','str'),'_OUT_FOLDER': (None, 'str'),
                      '_OUT_COUNTER': (None, 'int'), '_OUT_NAME': (None, 'list'),
                      '_OUT_STORE_CONFIG': (True, 'bool')}
    
    METAPARAMS_NAME = METAPARAMS_INFO.keys()
    
    
    def __init__(self, inputFile = None, rdm_object = None, procToRun = None, debug = False):
        if(debug):
            pdb.set_trace()
        self._RDMGEN = rdgen.RandomGenerator.init_random_generator(rdm_object)        
        self.listConfigs = self._parse_input(inputFile) #Generate the different configurations
        self._init_proc_to_run(procToRun)

    @classmethod
    def init_with_dispatch(cls, inputFile = None, rdm_object = None, procToRun = None):
        """ To think about goal is to init with some extra rules on low to deal
        with the dispatching of the input files configurations
        """
        obj = cls.__init__(inputFile, rdm_object, procToRun)
        obj.listConfigs = [obj._dispatch_configs(conf) for conf in obj.listConfigs]
        return obj

    def _dispatch_configs(self, dico):
        """ 
        """
        raise NotImplementedError()

# ---------------------------
# MAIN FUNCTIONS
#   -run_procedures
#   - save_res
# ---------------------------
    def run_procedures(self, saveFreq = 0, splitRes = False, printInfo = False):
        """ For each element of listConfigs run procedures (procToRun) and save reults
        Arguments:
        saveFreq:
            (-1) save at the end / (0) don't save anything/ (1) save each time 
            / (N)save each N res
        splitRes
             (False) only one file /(True):generate a file for each simulations 
             TODO: add gather all the runs for the same setup
        """
        self.listRes = []
        listResTmp = []
        if(self._out['store_config']):
            listConfTmp = []
        else:
            listConfTmp = None
        i_config = 0

        for conf in self.listConfigs:
            tmp = self.run_one_procedure(conf) # run the simulation for one config
            i_config +=1
            if(printInfo):
                print(i_config)
            self.listRes.append(tmp)
            listResTmp.append(tmp)
            if(self._out['store_config']):
                listConfTmp.append(conf)
            
            if ((saveFreq > 0) and (i_config%saveFreq == 0)):
                self.save_res(listResTmp, listConfTmp, splitRes)
                listResTmp = []
                listConfTmp = []
                        
        #Save all configs in one (separate) file maybe???
        if((saveFreq != 0) and (len(listResTmp) !=0)):
            self.save_res(listResTmp, listConfTmp, splitRes)
  
    
    def save_res(self, resToStore = None, confToStore=None, split = False, prefix = None, folder = None):
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
        if(prefix is None):
            prefix = self._out['prefix']
        if(folder is None):
            folder = self._out['folder']            
            
        if(split):
             for i in range(len(resToStore)):
                 oneRes = resToStore[i]
                 if(confToStore is None):
                     oneConf = None
                 else:
                     oneConf = confToStore[i]
                 oneRes['config'] = oneConf 
                 name_tmp= self._gen_name_res(oneConf)
                 self.write_one_res(oneRes, name_tmp, prefix=prefix, folder=folder)

        else:
            resAll = ut.concat_dico(resToStore)
            if(isinstance(confToStore, list)):
                confAll = ut.concat_dico(confToStore)
            else:
                confAll = None
            name_tmp = self._gen_name_res(confAll)
            
            self.write_one_res(resAll, name_tmp, prefix, folder)
            self.write_one_res(confAll, 'configs_' + name_tmp, prefix, folder)
                
# ---------------------------
#   PARSING
# ---------------------------
    def _parse_input(self, input_object):
        """ Allow for different type of input (and different behavior associated)
        (use a decorator??)        
        """
        if(ut.is_dico(input_object)):
            list_configs = self._parse_input_dico()
        
        elif(ut.is_list(input_object)):
            list_configs = self._parse_input_list()

        else:
            list_configs = self._parse_input_file(input_object)            
        
        return list_configs
                
    def _parse_input_file(self, inputFile = 'inputfile'):
        """Parse an input file and generate a list[dicoConfigs] where dicoConfigs are 
            dictionaries containing a configuration
        """
        with open(inputFile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ' ')
            list_values = list([])
            list_keys = list([])
            nbline = 0
            dico_METAPARAMS = {}

            for line in reader:
                nbline += 1
                if(line in self.EMPTY_LINE) or (line[0] in self.LEX_NA):
                    pass
                elif(line[0] in self.METAPARAMS_NAME):
                    assert (len(line) == 2), 'batch input file: 1 arg expected in l.' + str(nbline)+ ' (' +str(line[0]) + ')'
                    dico_METAPARAMS[line[0]] = ev(line[1])                                    
                else:
                    assert (len(line)>=2), 'batch input file: not enough args in l.' + str(nbline) 
                    list_keys.append(line[0]) 
                    ev_tmp = [ev(line[i]) for i in range(1,len(line))]
                    list_values = ut.cartesianProduct(list_values, ev_tmp, ut.appendList) 

            listConfigs = [ut.lists2dico(list_keys, l) for l in list_values]            
            listConfigs = self.apply_metaparams(listConfigs, dico_METAPARAMS)

        return listConfigs

    def _parse_input_dico(self, input_dico):
        raise NotImplementedError()

    def _parse_input_list(self, input_dico):
        raise NotImplementedError()
        
    def apply_metaparams(self, list_configs, dico_meta):
        """ deal with randomness (start with _RDM) // management of the output
        (start with OUT)
        """
        #Parse Metaparameters cast them to the right tye and use def value
        dico_meta_parsed = self.parse_metaparams(dico_meta)
        
        #RANDOM RUNS
        nb_rdm_run, fix_seed = dico_meta_parsed['_RDM_RUNS'], dico_meta_parsed['_RDM_FIXSEED']
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
        
        #OUTPUT METAPARAMS
        output_keys = ['_OUT_PREFIX', '_OUT_FOLDER', '_OUT_COUNTER', '_OUT_NAME', '_OUT_STORE_CONFIG']
        pref, fold, counter, name , store = [dico_meta_parsed[k] for k in output_keys]
        self._out = {'prefix':pref, 'folder': fold, 'counter': counter, 
                     'name': name, 'store_config':store}
        
        return list_configs
    
    def parse_metaparams(self, dico):
        """ Either to move to helper or add more rules (if needed)
        """
        dico_parsed = {}
        for k, v in self.METAPARAMS_INFO.items():
            def_val, _ = v
            val_tmp = dico.get(k, def_val)
            dico_parsed[k] = val_tmp
        return dico_parsed
            

# ---------------------------
#   WRITTING RESULTS
# ---------------------------
    def write_one_res(self, resToWrite, forceName = None, prefix = "res_", folder = None):
        """ Write a res as a text file
        Parameters:
            + resToWrite - a dictionnary
            + confToWrite - a dictionnary
            + prefix - to prepend to the name of the file
            + folder - in which folder to write the results
            + forceName - Force the name given to the file (if None look for 
                         a key name in the results, if none use default name)
        """
        #Create name of the file
        if(forceName is None):
            if("name" in resToWrite):
                name = prefix + resToWrite["name"] + ".txt"
            else:
                name = prefix + self.DEF_RES_NAME + ".txt" 
        else:
            name = prefix + forceName + ".txt"

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
        results = [ut.file_to_dico(f) for f in listFileName]        
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
        
    def _gen_name_res(self, configs):
        """ Generate the name of the simulation for saving purposes
        Ad-hoc can be changed reimplemented etc...
        Should be carefull as results will be overwritten
        """
        res_name = ''
        if(self._out.get('name') is not None):
            name = self._out.get('name')
            bits = [str(name[i])+"="+str(configs.get(name[i])) for i in range(len(name))]    
            res_name += ut.concat2String(bits)
            
        if(self._out.get('counter') is not None):
            res_name += str(self._out['counter'])
            self._out['counter'] +=1
        
        if(res_name == ''):
            #just in case it will generate a long random number
            res_name = str(rdgen.RandomGenerator.gen_seed()) 
        
        return res_name


#==============================================================================
#                   Testing
# 
#==============================================================================
if __name__ == "__main__": 
    import numpy.random as rdm
    batchTest = Batch('test_batch.txt')
    def funcDummy(x):
        rdstate = rdm.RandomState(x.get('_RDM_SEED'))
        return {'res': rdstate.uniform(size=10)}
    batchTest.attach_proc(funcDummy)
    batchTest.run_procedures(1, True, False)



    
