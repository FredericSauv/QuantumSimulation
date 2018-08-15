#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
#
#============================================================================== 
import sys
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels.BH import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import pFunc_base, Batch 
    
else:
    from . import BH1D as bh1d
    from ...Utility import Helper as ut
    from ...Utility.Optim import pFunc_base, Batch

class learner1DBH(Batch.BatchParametrizedControler):
    """ Simply define the underlying model constructor and some custom functions to cope
    with a bug"""
    UNDERLYING_MODEL_CONSTRUCTOR = bh1d.BH1D


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


        
#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    # Testing has been spin off to Test/BH1D
    pass


        