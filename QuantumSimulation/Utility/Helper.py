# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""
import csv
csv.field_size_limit(100000000)
import logging 
logger = logging.getLogger(__name__)
import os 
import pdb
import numpy as np
from functools import wraps
import matplotlib.pylab as plt
#import numpy.random as rdm
#from ast import literal_eval as ev
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
#==============================================================================
#                   I/O (write read)
#==============================================================================
## SHOULD BE DEPRECIATED
def dico2text(dico, fileName = None, typeWrite = 'w', beg = None, end = None):
    """ Transform a dico into a text file. First element of the line is the key, 
        and the rest is the value
    Parameters:
        + dico - dictionnary to write
        + filename - name of the file / if None will look for the key 'name' in dico 
        + typeWrite - 'w': create a new file/erase, 'a' append to a file
        + beg/end - if not None write at the beginning/end of the file
    """
    if (fileName is None):
        assert hasattr(dico, 'name'), 'Utility.dico2text: need either a fileName of a name key in the dico'
        fileName = str(dico['name']) + '.txt'
    with open(fileName, typeWrite) as file:
        if(beg is not None):
            file.write(beg)
            file.write("\n")
        for keys, val in dico.items():
            file.write(str(keys))
            file.write(' ')
            file.write(repr(val).replace(" ",""))
            file.write("\n")
        if(end is not None):
            file.write(end)
            file.write("\n")
            

def text2dico(file, delimiter = ' ', skip=['']):
    """ Transform a txt file into a dico. First element of the line is the key, 
        and the rest is the value
    """
    dicoRes = {}
    nbline = 0
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = delimiter)
        for line in reader:
            if (line[0] not in skip):
                nbline += 1
                assert (len(line)>=2), 'batch input file: not enough args in l.' + str(nbline)                
                dicoRes[line[0]] = line[1:]
    return dicoRes

def dico_to_text_rep(dico, fileName = None, typeWrite = 'w', beg = None, end = None):
    """
    Parameters:
        + dico - dictionnary to write
        + filename - name of the file / if None will look for the key 'name' in dico 
        + typeWrite - 'w': create a new file/erase, 'a' append to a file
        + beg/end - if not None write at the beginning/end of the file
    """
    if (fileName is None):
        assert hasattr(dico, 'name'), 'Utility.dico2text: need either a fileName of a name key in the dico'
        fileName = str(dico['name']) + '.txt'
    with open(fileName, typeWrite, newline=None) as file:
        if(beg is not None):
            file.write(beg)
        file.write(repr(dico).replace(" ","").replace("\n","").replace("array", "np.array"))
        if(end is not None):
            file.write(end)
            
        
# SHOULD BE USED 
def write_to_file_for_eval(obj, fileName, typeWrite = 'w'):
    """ write the representation of an object (after some small alterations) to
    a file """
    with open(fileName, typeWrite, newline=None) as file:
        file.write(custom_repr(obj))

def eval_from_file(file, evfunc = eval, replace_func = None):
    """ open a txt file grasp the first element transform it and eval it"""
    if replace_func is None:
        replace_func = lambda x:x
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        line = reader.__next__()
        assert (len(line)==1), 'not the right format/size' 
        transformed = replace_func(line[0])
        evaluated = evfunc(transformed)
    return evaluated

def eval_from_file_supercustom(file, evfunc = eval):
    """ workaround for a bug"""
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        line = reader.__next__()
        assert (len(line)==1), 'not the right format/size' 
        string = parse_supercustom(line[0])
        evaluated = evfunc(string)
    return evaluated

def parse_supercustom(string):
    if(string.find("'name_algo':'BO2'") == -1):
        beg, tmp = string.split(",'opt_more'")
        garbage, end = tmp.split(",'init':")
        res = beg + ",'init':" + end
    else:
        res = string
    return res

def custom_repr(obj):
    """ Represent and remove unwanted spaces, line jump, transform array in np.array"""
    res = repr(obj).replace(" ","").replace("\n","").replace("array", "np.array")
    return res          

#change name
def findFile(nameFile = None, prefix = '', folderName = None):
    """Find files either with exact name or starting with a prefix, either in 
    the current directory or in a specified folder. Return a list with the names 
    of the files found (taking into account the path if a folder was specified)
    """
    if (folderName is None):
        list_all_files = os.listdir()
        if(nameFile is None):
            list_files = [f for f in list_all_files if f.startswith(prefix)]
        else:
            list_files = [f for f in list_all_files if (f == nameFile)]
    else:
        list_all_files = os.listdir(folderName)
        if(nameFile is None):
            if(prefix in [None, '']):
                list_files = [os.path.join(folderName, f) for f in list_all_files]
            else:
                len_pref = len(prefix)
                list_files = [os.path.join(folderName, f) for f in list_all_files if (f[:len_pref] == prefix)]
        else:
            list_files = [os.path.join(folderName, f) for f in list_all_files if (f == nameFile)]

    
    return list_files
  
#==============================================================================
#                   MANIP STRUCTURES (list / dico / text files)
#==============================================================================
#def CountElementsNestedList(l, ninit = 0):
#    ntotal = ninit
#    nlist = []
#    for el in l:
#        if(isinstance(el, list)):
#            ntotal_nested, nlist_ntested = CountElementsNestedList(el)
#            ntotal += ntotal_nested
#            nlist.append(nlist_ntested)
#        else:
#            ntotal += 1
#            nlist.append(1)
#    return ntotal, nlist

# test_nested = {'a':{'aa':5, 'bb':7}, 'b':22}
def print_nested_keys(nested_struct, padding = '+', nesting_max = None):
    try:
        if((nesting_max is None) or(nesting_max > 0)):
            keys = nested_struct.keys()
            if(nesting_max is not None):
                nesting_max -= 1
            for k in keys:
                print(padding+k)
                print_nested_keys(nested_struct[k], padding + padding, nesting_max)
    except:
        pass






def save_str(string, file_name, type_write = 'w'):
    """ save a string <str> to a file <filename>"""
    with open(file_name, type_write, newline=None) as file:
        file.write(string)

            
def file_to_dico(file, nested = False, delimiter = ' ', na_character = [''], evfunc = eval):
    """ Transform a txt file into a dico. Two behaviors (nested=True): 
        line after line take the first elem as a key and eval the second one 
        to get the value (nested = False) just eval first line"""
    nbline = 0
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = delimiter)
        if(nested):
            dicoRes = {}
            for line in reader:
                if (line[0] not in na_character):
                    nbline += 1
                    assert (len(line)>=1), 'batch input file: not enough args in l.' + str(nbline)                
                    dicoRes[line[0]] = [evfunc(el) for el in line[1:]]
        else:
            line = reader.__next__()
            assert (len(line)==1), 'not the right format/size' 
            dicoRes = evfunc(line[0])
    return dicoRes



            

def lists2dico(list_keys, list_values):
    """
    Purpose:
        form a dictionnary with a list of keys and a list of values (should have
        the same length)
    """
    assert (len(list_keys) == len(list_values)), 'lists2dico lengths differ'
    dicoRes = {}
    for ind in range(len(list_keys)):
         dicoRes[list_keys[ind]] = list_values[ind]
    
    return dicoRes





def getStringOrDico(strOrDico):
    """
    What it does:
        If it is a string (assume to be the path to a txt file) extract dico from the textfile 
        If it is a dico return the dico
        Else not expected
    """
    if isinstance(strOrDico, str):
        dico = text2dico(strOrDico)
    elif isinstance(strOrDico, dict):
        dico = strOrDico
    else:
        raise NotImplementedError()
    return dico

def get_from_dico_def(key, dico, dico_default):
    """ Provide one ref dictionary (in which to look first) and a default 
    dictionary if the key can't be found in dico
    Will raise an error if the key is neither in dico nor dico_default
    """
    if(is_list(key)):
        res = [get_from_dico_def(k, dico, dico_default) for k in key]
    else:
        try:
            res = dico[key]
        except:
            res = dico_default[key]

    return res

def printDicoLineByLine(dico):
    """
    Purpose:
        Print the key/values of a dico jumping lines between each pair
    """
    for key, vals in dico.items():
        str_tmp = "'" + key +"': " + str(vals) 
        print(str_tmp)


def concat_dico(listDico, global_name='Concatenated Results', suffix = ''):
    """
    Purpose:
        transform a list of dictionaries containing results in one big dictionnary
        a global name is added and we append to the keys of each of the dictionary  
        a counting integer (to avoid clashes between same keys)
    """
    dicoOut = {}
    if(global_name is not None):
        dicoOut['name'] = global_name

    counter = 0
    for dico in listDico:
        counter += 1
        for keys in dico:
            dicoOut[keys+'_'+ suffix + str(counter)] = dico[keys]
            
    return dicoOut

def merge_dico(dico_init, dico_new, update_type = 0, copy = True):
    """ Update entries of (a copy if <copy>) <dico_init> based on 
    entries in <dico_new> according to <update_type> which defines 
    the update rules to apply
    >> update_type:
        0 if key doesn't exist CREATE  // if exist UPDATE
        1                      CREATE  //          FAIL 
        -1                     FAIL    //          UPDATE
        2                      CREATE  //          SUM NEW VALUE TO THE OLD ONE 
        3                      CREATE  //          FAIL IF IT IS NOT THE SAME VALUE
        4                      PASS    //          UPDATE
    """

    if(copy):
        dico_final = dico_init.copy() 
    else:
        dico_final = dico_init

    if(dico_new is None):
        return dico_final
    
    for k, v in dico_new.items():
        if (update_type == 0):
           dico_final[k] = v
        elif (update_type == 1):
            assert (k not in dico_final), '1 : entry already existing'
            dico_final[k] = v
        elif (update_type == -1):
            assert (k in dico_final), '-1 : entry doesnt exist'
            dico_final[k] = v
        elif (update_type == 2):
            if (k in dico_final):
                dico_final[k] += v
            else:
                dico_final[k] = v
        elif (update_type == 3):
            if(k in dico_final):
                assert (dico_final[k] == dico_new[k]), '3: entry already exists, but not the same value'
            else:
                dico_final[k] = v
        elif(update_type == 4):
            if(k in dico_final):
                dico_final[k] = v
        else:
            raise NotImplementedError()
            
    return dico_final
        
def add_dico(dico_init, dico_new, copy = True):
    """ Specific implemenetation of the function above
    """
    res = merge_dico(dico_init, dico_new, update_type = 1, copy = copy)
    return res 
    

def merge_N_dico(conflict = 0, *args):
    """
    Purpose:
        Extend update_dico to work with an arbitrary number of dictionnaries
    """
    dico_final = {}
    for dico in args:
        dico_final = merge_dico(dico_final, dico, conflict)
    return dico_final



def filter_dico(dico, keys_filter, keep = True):
    """ Create a new dico based on an initial dico and a list of a keys
    to keep or remove depending on parameters keep
    """
    if(keep):
        dico_final = {k: v for k, v in dico.items() if k in keys_filter}
    else:
        dico_final = {k: v for k, v in dico.items() if k not in keys_filter}
    return dico_final    

def filter_dico_first_char(dico, keys_first_char, keep = True):
    """ Create a new dico based on an initial dico and a first char value for the keys
    """
    if(keep):
        dico_final = {k: v for k, v in dico.items() if k[0] == keys_first_char}
    else:
        dico_final = {k: v for k, v in dico.items() if k[0] != keys_first_char}
    return dico_final      

def find_duplicates(array, decimals=None):
    """ find duplicates (up to some rounding)
    return list of list of indices (corresponding to unique values) and a boolean 
    (True if some values weren't unique)
    """
    indices = np.arange(array.shape[0])
    if (decimals is not None):
        a = np.round(array, decimals=decimals)
    else:
        a = array
    unique_values, count = np.unique(a, return_counts = True)
    list_indices = [[i for i in indices if a[i]== u]for u in unique_values]
    is_notunique = np.sum(count>1)

    return list_indices, is_notunique

def extract_from_nested(nested_structure, path = [], list_output = False):
    """ Extract an elemenet from a nested structure following a path
    if path = [a, b, c] >> return nested_sturcture[a][b][c]
    """
    res = nested_structure
    for p in path:
        res = res[p]

    if(list_output):
        res = as_list(res)
    return res

def try_extract_from_nested(nested_structure, path = [], list_output = False):
    try:
        res = extract_from_nested(nested_structure, path, list_output)
    except:
        res = None
    return res

def is_x_in_y(x, y, tol=0):
    """ depending on the shape of test if x == y, x in [y[0], y[1]] """
    res = False
    if is_iter(y):
        if (len(y) >=2):
            res = (x >= y[0] - tol) * (x <= y[1] + tol)
        else:
            res = (np.abs(x-y[0]) <= tol)   
    else:
        res = (np.abs(x-y) <= tol)
    return res

def is_x_not_in_y(x, y, tol=0):
    """ depending on the shape of test if x == y, x in [y[0], y[1]] """
    return not(is_x_in_y(x, y, tol))
#==============================================================================
#                   FUNCTIONS
# Not really used.. May be deleted
#==============================================================================
def idFun(*args):
    """identitity function
    """        
    return args   

def idFun1V(x):
    """ identitity function with one var
    """        
    return x

def zeroFun(*args):
    """ identitity function with one var
    """        
    return 0

def compoFunctions(listFun, initValue = 0, order = 0):
    """ compoFunction([f,g,h], initValue = 15, order = 0) >> f(g(h(15)))
    """
    res = initValue
    if (order == 0):
        f2apply = listFun
    elif(order == -1):
        f2apply = reversed(listFun)
    else:
        raise NotImplementedError()

    for f in f2apply:
        res = f(res)
    
    return res

def reduceFunToFirstOutput(fun):
    """ Turn a function returning (potentially) a list to a function returning a
        scalar (first element of the list) 
    """
    def funWrapped(x):
        res = fun(x)
        if isinstance(res, list):
            res = res[0]
        return res
    return funWrapped


def ArrayToDicoInputWrapper(fun, index2NameFun = None, **args_call):
    """ Transform a function taking array as input to a function taking named 
    args as input
    i.e. cost('0'=2, '1'=5, '2'=1) = fun([2,5,1])
    !!! Used to interface the bayesianOptimization Library
    Horrible??
    """
    if(index2NameFun is None):
        def cost(**args):
            list_args = [args[str(i)] for i in range(len(args))]
            res = fun(np.array(list_args), **args_call)
            if(isinstance(res, list)):
                res = res[0]
            return -res
    else:
        def cost(**args):
            list_args = [args[index2NameFun(i)] for i in range(len(args))]
            res = fun(np.array(list_args), **args_call)
            if(isinstance(res, list)):
                res = res[0]
            return -res        
    return cost

#==============================================================================
#                   SETS
#==============================================================================
def cartesianProduct(list1, list2, productFun, axis=0):
    """ Generate Cartesian products of elements of two sets (here lists)        
        [a b] x [d e] -> [a*d, a*e, b* d, b*e] (if axis == 0)
        With the product a * b defined as productFun(a, b)
    Seems to bealready implemented 
    """ 
    if(axis == 0):
        if(list1 == []):
            # For initialisation (may have to change it as strange)
            resList = [productFun([], l2) for l2 in list2]
        elif(list2 == []):
            resList == list1
        else:
            resList = [productFun(l1, l2) for l1 in list1 for l2 in list2]
    
    ##WTF???
    elif(axis == 1):
        if(list1 == []):
            # For initialisation (may have to change it as strange)
            resList = [productFun([], l2) for l2 in list2]
        elif(list2 == []):
            resList == list1
        else:
            resList = [productFun(l1, l2) for l2 in list2 for l1 in list1]

    return resList


def appendList(x,y):
    return x + [y]



#==============================================================================
#                   PARSING STRINGS
# getRecasted
# concat2String
# splitString
# removeFromString
# recastString
#==============================================================================
def as_list(l):
    try:
        res = list(l)
    except:
        res = list([l])
    return res

def parse_enclose_with_counter(string, **extra_args):
    """ {dico1} + {dico2} >>> str_func({dico1}}"""
    before = extra_args.get('before', '(') 
    outer = extra_args.get('nested', True)
    after = extra_args.get('after', ')')   
    flag = extra_args.get('flag', '{}')
    flag_bef = flag[0]
    flag_after = flag[1]

    counter = 0
    str_out = ''

    for char in string:
        if(char == flag_bef):
            if(outer):
                if(counter == 0):
                    str_out += before
                    str_out += char
                counter += 1
            else:    
                str_out += before
                str_out += char
                
        elif(char == flag_after):
            if(outer):
                counter -= 1
                if(counter == 0):
                    str_out += char
                    str_out += after    

            else:
                str_out += char
                str_out += after

            assert (counter >= 0), "pb when parsing"
        
        else:
            str_out += char
    return str_out    

def custom_cast(string_val, type_casting):
    """ cast string_val as type_casting
    casting_allowed = ['int', 'str', 'bool'], should be augmented with new types    
    should it deal with other types of input (i.e. not string???)
    """
    if(type_casting == 'str'):
        return str(string_val)
    elif(type_casting == 'int'):
        return int(string_val)
    elif(type_casting == 'float'):
        return float(string_val)
    elif(type_casting == 'bool'):
        return str2boolean(string_val)
    else:
        raise NotImplementedError()
        
        
def getRecasted(dico, key, default = None):
    """
    Purpose:
        get() behavior on a dictionnary + the returned value is recasted if possible

    """    
    if(isinstance(key, list)):
        recasted_value = [getRecasted(k) for k in key]
    else:
        string_value = dico.get(key,default)
        if(isinstance(string_value, str)):
            recasted_value = string_value
        else:
            recasted_value = recastString(string_value)
    return recasted_value

def concat2String(*args, symbol = '_'):
    """ Create a string from n args using a delimiter between each args
        >> concat2String('A', 'B', 'CD') -> 'A_B_CD'
    """
    res = str(args[0])
    if len(args) > 1:
        for i in range(len(args)-1):
            res = res + symbol + str(args[i+1])
    return res


def splitString(string, symbol='_'):
    """ split a string according to the delimitator symbol return a list 
    containing the different bits
    >> splitString('A_B_CD', '_') -> ['A', 'B', 'CD']

    """
    res = []
    index_symb = string.find(symbol)
    while (index_symb>=0):
        res.append(string[:index_symb])
        string = string[(index_symb+1):]
        index_symb = string.find(symbol)
    res.append(string)
    return res

def removeFromString(string, toremove = ''):
    """ Remove toremove from a string (first  occurence only)
    """
    len_toremove = len(toremove)
    index_toremove = string.find(toremove)
    if(index_toremove < 0):
        res = string
    else:
        res = string[:index_toremove] + string[(index_toremove + len_toremove):]
    return res

def recastString(string):
    """ Recast a string in (in this order) int / float / bool / list[float] / 
    matrixFloat / list[str] if possible 
    """
    if(is_int(string)):
        return int(string)

    elif(is_float(string)):
        return float(string)

    elif(is_boolean(string)):
        return str2boolean(string)
    
    elif(is_list_float(string)):
        return listFloatFromString(string)

    elif(is_matrix_float(string)):
        return matrixFloatFromString(string)

    elif(is_list_string(string)):
        return listStringFromString(string)
    
    else:
        return string


def listFloatFromString(string):
    """
    Purpose:
        convert a string containing a list of floats to an actual list of Float
        Fail in case it is not a list of Float
    """
    if((string[0] == '[') and (string[-1] == ']')):
        stringTmp = string[1 : (len(string)-1)]
        bits = splitString(stringTmp, ',')
        res = [float(b) for b in bits]
    else:
        assert True, 'Not a List of Float'
    return res

def matrixFloatFromString(string):
    """
    Purpose:
        convert a string containing a list of floats to an actual list of Float
        Fail in case it is not a list of Float
    """
    if((string[0] == '[') and (string[-1] == ']')):
        stringTmp = string[1 : (len(string)-1)]
        bits = splitString(stringTmp, ',')        
        res = [float(b) for b in bits]
    else:
        assert True, 'Not a List of Float'
    return res


def listStringFromString(string):
    """
    Purpose:
        convert a string containing a list of floats to an actual list of Float
    """
    if((string[0] == '[') and (string[-1] == ']')):
        stringTmp = string[1 : (len(string)-1)]
        res = splitString(stringTmp, ',')
    else:
        assert True, 'Not a List of String'
    return res




def is_float(string):
    """
    Purpose:
        Chef if it is an int
    """
    try:
        float(string)
        return True
    except:
        return False


def is_boolean(string):
    """
    Purpose:
        Chef if it is an int
    """
    bool_list = ["True", "False"]
    return (string in bool_list)


def str2boolean(string):
    """ convert string to boolean, and fail if it can't"""
    true_list, false_list  = ["True", "1"], ["False", "0"]
    if (string in true_list):
        return True
    elif(string in false_list):
        return False
    else:
        raise TypeError()
    
    
def is_list_float(string):
    """
    Purpose:
        Chef if it is an int
    """
    try:
        listFloatFromString(string)
        return True
    except:
        return False        

def is_matrix_float(string):
    """
    Purpose:
        Chef if it is an int
    """
    try:
        matrixFloatFromString(string)
        return True
    except:
        return False    


def is_list_string(string):
    try:
        listStringFromString(string)
        return True
    except:
        return False     
    

def is_int(string):
    """
    Purpose:
        Chef if it is an int
    """
    try:
        int(string)
        return True
    except:
        return False



#==============================================================================
#                       Matrix Functions
#  
# dagger
# isHermitian
# isUnit
# expOperator
# trimEigenSystem
# sortEigenSystem
# sortedEigenSystem
# degenEV
# gram_schmidt
#
#
#==============================================================================
#def dagger(H):
#    return np.conj(np.transpose(H))
#
#def isHermitian(H):
#    return np.allclose(H, dagger(H))
#
#def isUnit(H):
#    test = False
#    shape = H.shape
#    if shape[0] == shape[1]:
#         test = np.allclose(np.dot(H, dagger(H)), np.eye(shape[0]))
#    return test


def expOperator(H, coeff = 1.0):
    """ 
        Compute U = e^{coeffs * H}
    """
    dim = H.shape
    eH = np.zeros(dim, dtype='complex128')
    hvals, hvects = degenEV(H, rounding = None)
    for i in range(dim[0]):
        eH += np.exp(coeff * hvals[i]) * np.outer(hvects[:,i], np.conj(hvects[:,i]))
    return eH

def trimEigenSystem(evals, evecs, min_cutoff=None, max_cutoff=None, order = None):
    """
    Purpose:
        Sort and trim eigenvalues and eigenvectors
    """
    evals, evecs = sortEigenSystem(evals, evecs, order)
    mask = np.ones(len(evals))

    if(min_cutoff is not None):
        mask = mask * (evals > min_cutoff) 
        if(mask[0]):
            print('TrimES: has retained the first eigen value...')


    if(max_cutoff is not None):
        mask = mask  * (evals <=  max_cutoff)
        if(not(mask[-1])):
            print('TrimES: has retained the last eigen value...')

    cut_vals = evals[mask]
    cut_vecs = evecs[:, mask]

    return cut_vals, cut_vecs


def sortAccordingToFirst(*args, **kwargs):
    """ Sort arrays according to the first one provided
    
    """
    idx = args[0].argsort()
    if(('order' in kwargs.keys()) and (kwargs['order'] =='desc')):
        idx = idx[::-1]
    test_len = np.array([len(array) for array in args])
    assert np.any(test_len == len(idx)), "problem size of the arrays"
    if (len(args) == 1):
        res = args[0][idx]
    else:    
        res = [array[idx] for array in args]

    return res

# TODO: Replace by sortAccordingToFirst
def sortEigenSystem (evals, evecs, order = None):
    """ Sort eigenvectors accoring to their eigenvalues
    """
    idx = evals.argsort()
    if(order in [None, 'asc']):
        idx = idx
    elif(order == 'desc'):
        idx = idx[::-1]
    else:
        raise NotImplementedError()

    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def sortedEigenSystem (H, order = None):
    """ Return the sorted eigensystem
    """    
    evals, evecs = np.linalg.eig(H) 
    evals, evecs = sortEigenSystem (evals, evecs, order)
    return evals, evecs


def degenEV(H, rounding=10):
    """
    Purpose:
        Return evals, evecs after applying a Graham Schmidt procedure when evalues
        are degenerate (up to some rounding)
        
    ## FIX 12Apr
    In the event the degenerate evectors are not independent AND there is only one
    degenerate eigen subspace return  (some basis of) the orthogonal of the Non-
    degenerate subspaces
    
    It was a problem for 
    array([[  0.00000000e+00+0.j,   1.45232839e-98+0.j,   1.13836027e-99+0.j],
       [  1.45232839e-98+0.j,   0.00000000e+00+0.j,   0.00000000e+00+0.j],
       [  1.13836027e-99+0.j,   0.00000000e+00+0.j,   1.00000000e-01+0.j]])
    
    """
    # Check degeneracy of the quasienergies and take action if necessary
    evals, evecs = sortedEigenSystem(H)    
    deg_indices, is_uniquepb = find_duplicates(evals, rounding)
    if is_uniquepb:
        for index in deg_indices:
            if(len(index)>1):
                try:
                    to_ortho = evecs[:, index]
                    ortho = gram_schmidt(to_ortho)
                    evecs[:,index] = ortho
                    
                except ArithmeticError:
                    pdb.set_trace()
                    nb_degen_subs = np.sum([len(ind)>1 for ind in deg_indices])
                    assert nb_degen_subs == 1, "can't deal with this case"
                    assert H.shape[0] == H.shape[1], "can't deal with this case"
                    dim = H.shape[1]
                    a_basis = np.zeros_like(H)
                    np.fill_diagonal(a_basis,1)
                    index_non_degen = [i for i in range(dim) if i not in index]
                    non_degen = evecs[:, index_non_degen]
                    ortho = get_orthospace(non_degen, a_basis)
                    evecs[:, index] = ortho
                    
    return evals, evecs

def get_orthospace(subsp, sp):
    """
    Convention vectors are column
    Purpose:
        return the orthogonal subspace of subsp
    
    
    >> subspace should be given as an array of (possibly non linear indep) m 
    D-dim vectors (i.e. m x D array)
    
    >>sp should be given as a list of D vectors spanning the space
    """
    subsp_indep = gram_schmidt(subsp, trunc = True)
    dim_subsp = subsp_indep.shape[1]
    
    full_sp_dep = np.concatenate((subsp_indep, sp), axis = 1)
    full_sp = gram_schmidt(full_sp_dep, trunc = True)
    
    ortho_subsp = full_sp[:,dim_subsp:]
    return ortho_subsp

def gram_schmidt(vecs, trunc = False):
    """
    Purpose:
        apply a Graham Schmidt procedure on a set of vectors (each column of vecs
        is a vector)
        
    New Behavior implemented: trunc if true and if the initial set of vectors 
    is not linearly independent: return a truncated set of vectors
    """
    result = np.zeros_like(vecs)
    n = vecs.shape[1]

    if(not(trunc)):
        r = np.dot(np.conj(vecs[:,0]), vecs[:,0])
        if r == 0.0:
            raise ArithmeticError("Vector with norm 0 occured.")
        else:
            result[:, 0] = vecs[:, 0]/r

        for j in range(1, n):
            q = vecs[:, j]
    
            for i in range(j):
                rij = np.dot(np.conj(result[:,i]), q)
                q = q-rij*result[:, i]
    
            rjj = np.dot(np.conj(q), q)
    
            if rjj == 0.0:
                raise ArithmeticError("Vector with norm 0 occured.")
            else:
                result[:,j] = q/np.sqrt(rjj)
                
    else:
        index_to_keep = []
        cutoff = 1e-12
        r = np.dot(np.conj(vecs[:,0]), vecs[:,0])
        if np.abs(r) > cutoff:
            result[:, 0] = vecs[:, 0]/r
            index_to_keep.append(0)
            
        for j in range(1, n):
            q = vecs[:, j]
    
            for i in range(j):
                rij = np.dot(np.conj(result[:,i]), q)
                q = q-rij*result[:, i]
    
            rjj = np.dot(np.conj(q), q)
    
            if np.abs(rjj) > cutoff:
                result[:,j] = q/np.sqrt(rjj)
                index_to_keep.append(j)
        result = result[:, index_to_keep]

    return result

#==============================================================================
#                   TS: 
# TS are understood as arrays of Nx2 float where 1st column is a list of indices
# 
#==============================================================================
def sort_TS(TS):
    """ sort a Nx2 array according to its first column
    """
    if(is_list(TS)):
        return [sort_TS(el) for el in TS]
    
    elif(len(TS.shape) == 2):
        TS_out = np.array(TS)
        sort_ind = np.argsort(TS[:,0])
        return TS_out[sort_ind,:]
    else:
        raise NotImplementedError()

def merge_TS(listTS, rule = 'lower'):
    """take a list of TS compute the union of their indices and extend each TS 
    such that 
    """
    list_index = [np.array(el)[:,0] for el in listTS]
    unique_index = np.unique([el for l in list_index for el in l])
    res = [[ind, np.array([find_from_index_TS(ind, np.array(ts), rule=rule) for ts in listTS])] for ind in unique_index]
    return res

def merge_and_stats_TS(listTS, dico=True, rule = 'lower'):
    """
    """
    merged_TS = merge_TS(listTS, rule=rule)
    stats = np.array([np.concatenate(([ind], get_stats(ts))) for ind, ts in merged_TS])
    if(dico):
        res = stats_array2dico(stats)
    else:
        res = stats
    return res

def stats_array2dico(stats):
    res = {'index':stats[:,0], 'avg': stats[:,1],'min':stats[:,2], 'max': stats[:,3]
        ,'std':stats[:,4], 'avg_pstd': stats[:,5],'avg_mstd':stats[:,6], 'n': stats[:,7]}
    return res

def get_stats(list_val, dico_output = False):
    """ from a list of value get the stats defined in this order:
        [avg, mini, maxi, std, avg_pstd, avg_mstd, n]
    """
    avg = np.average(list_val)
    mini = np.min(list_val)
    maxi = np.max(list_val)
    std = np.std(list_val)
    avg_pstd = avg + std
    avg_mstd = avg - std
    n = len(list_val)
    if(dico_output):
        res = {'avg':avg,'min':mini,'max':maxi,'std':std,'avg_pstd':avg_pstd,'avg_mstd':avg_mstd, 'n':n, 'index':None}
    else:
        res = [avg, mini, maxi, std, avg_pstd, avg_mstd, n]
    return res
    
    
def find_from_index_TS(index, TS, rule='lower'):
    """ from a TS get the 
    """
    if(is_iter(index)):
        return [find_from_index_TS(ind, TS, rule) for ind in index]    
    else:
        if(rule == 'lower'):
            flag = np.diff(index < TS[:,0])
            if(index<TS[0,0]):
                flag[0] = True
            if (np.sum(flag) == 0):
                mask = np.concatenate((flag, [True]))
            else:
                mask = np.concatenate((flag, [False]))
            res = TS[mask,1][0]
        elif(rule == 'linear'):
            if(index<=TS[0,0]):
                res = TS[0,1]
            elif(index>=TS[-1,0]):
                res = TS[-1,1]
            else:
                first = np.arange(len(TS)-1)[np.diff(index < TS[:,0])][0]
                res = TS[first,1] + (TS[first+1,1]-TS[first,1]) * (index - TS[first,0]) / (TS[first+1,0]-TS[first,0])
        else:
            raise NotImplementedError()
        return res
    
    

def plot_from_stats(stats, dico_plot = {}, plot = None, return_plot = False, save_fig = False):
    """ from a stats create some plots
    stats = {'index':, 'avg', 'mini', 'maxi', 'std', 'avg_pstd', 'avg_mstd', 'n'}
    """
    # Get attributes of the graph
    if(plot is None):
        fig, ax_plt = plt.subplots()
    else:
        fig, ax_plt = plot
    color = dico_plot.get('color', 'b')
    shape = dico_plot.get('shape', 's')
    legend = dico_plot.get('legend')
    component = dico_plot.get('component', 'avg')
    func_wrap = dico_plot.get('func_wrap')

    # Get data
    indices = stats['index']
    try:
        nb_TS_used = stats['n']
        print(str(np.min(nb_TS_used)) + ' dataset were used to get the stats')
    except:
        pass
    if(func_wrap is None):
        func_wrap = idFun1V
    ymin = func_wrap(stats['min'])
    ymax = func_wrap(stats['max'])
    yavg = func_wrap(stats['avg'])

    # Different plot types
    if(component  == 'minmax'):
        ax_plt.plot(indices, ymin, color = color, label=legend)
        ax_plt.plot(indices, ymax, color = color)
        ax_plt.fill_between(indices, ymin, ymax,alpha=0.2, color = color)
    
    elif(component  == 'avgminmax'):
        ax_plt.plot(indices, ymin, color = color, label=legend)
        ax_plt.plot(indices, ymax, color = color)
        ax_plt.fill_between(indices, ymin, ymax,alpha=0.2, color = color)
        ax_plt.plot(indices, yavg, dashes=[6, 2], color = color)

    elif(component == 'finalerror'):
        if(len(np.shape(yavg)) == 0):
            avg = yavg
        else:
            avg = yavg[-1]
        if(len(np.shape(ymin)) == 0):
            mini = ymin
        else:
            mini = yavg[-1]
        if(len(np.shape(ymax)) == 0):
            maxi = ymax
        else:
            maxi = ymax[-1]

        err_minus = np.array([avg - mini])
        err_plus = np.array([maxi - avg])
        yavg = np.array([avg])
        x = np.array([stats.get('label_nb', 1)])
        ax_plt.errorbar(x, yavg, yerr=[err_minus, err_plus], fmt=shape, color = color, ecolor=color, capthick=2, label = legend)

    elif(component == 'finalerrorstd'):
        m = func_wrap(stats['avg_mstd'])
        p = func_wrap(stats['avg_pstd'])
        if(len(np.shape(yavg)) == 0):
            avg = yavg
        else:
            avg = yavg[-1]
        if(len(np.shape(m)) == 0):
            mini = m
        else:
            mini = m[-1]
        if(len(np.shape(p)) == 0):
            maxi = p
        else:
            maxi = p[-1]

        err_minus = np.array([avg - mini])
        err_plus = np.array([maxi - avg])
        yavg = np.array([avg])
        x = np.array([stats.get('label_nb', 1)])
        ax_plt.errorbar(x, yavg, yerr=[err_minus, err_plus], fmt=shape, color = color, ecolor=color, capthick=2, label = legend)
                                
    elif(component == 'pm1sd'):
        m = func_wrap(stats['avg_mstd'])
        p = func_wrap(stats['avg_pstd'])
        ax_plt.plot(indices, m, color = color, label=legend)
        ax_plt.plot(indices, p, color = color)
        ax_plt.fill_between(indices, m, p, color = color, alpha = 0.2)
        
    elif(component == 'avgpm1sd'):
        m = func_wrap(stats['avg_mstd'])
        p = func_wrap(stats['avg_pstd'])
        ax_plt.plot(indices, yavg, dashes=[6, 2], color = color, label=legend)
        ax_plt.plot(indices, m, color = color)
        ax_plt.plot(indices, p, color = color)
        ax_plt.fill_between(indices, m, p, color = color, alpha = 0.2)
    
    else:
        comp = splitString(component)
        for c in comp:
            if(legend is None):
                ax_plt.plot(indices, func_wrap(stats[c]), label = str(c))
            else:
                ax_plt.plot(indices, func_wrap(stats[c]))

    apply_dico_plot(None, ax_plt, dico_plot)
    if(is_str(save_fig)):
        fig.savefig(save_fig)
    
    if return_plot:
        return (fig, ax_plt)

    
    
def plot_from_list_stats(list_stats, dico_main, dico_inset = None, save_fig = False, **args_extra):
    """ from a LIST of stats create some custom plots
    stats = [[index, avg, mini, maxi, std, avg_pstd, avg_mstd],...]
    """
    # pdb.set_trace()
    # Prepare main graphs
    listColors = dico_main.get('colors', ['b', 'r', 'g', 'orange'])
    listShapes = dico_main.get('shapes', ['s', 'o', 'v', 'p'])
    legend = dico_main.get('legend')
    component = dico_main['component']
    func_wrap = dico_main.get('func_wrap')
    dico_one_plot = {'component': component, 'func_wrap':func_wrap}
    fig, ax_plt = plt.subplots()

    # Prepare inset graph
    if(dico_inset not in [None, {}]):
        legend_inset = dico_inset.get('legend')
        inset_pos = dico_inset.get('inset', [0.4, 0.4, 0.46, 0.46])
        size = dico_inset.get('inset_size', 12)
        dico_inset['prop'] = {'size': size}
        ax_plt2 = fig.add_axes(inset_pos)
        component_inset = dico_inset['component']
        func_wrap_inset = dico_inset.get('func_wrap')
        dico_one_plot_inset = {'component': component_inset, 'func_wrap':func_wrap_inset}
        inset_flag = True
    else:
        inset_flag = False

    # plot the different stats series
    for n, stats in enumerate(list_stats):
        #plot main
        dico_one_plot['color'] = listColors[n]
        dico_one_plot['shape'] = listShapes[n]
        if (legend is not None):
            if(n < len(legend)):
               dico_one_plot['legend'] = legend[n]
            else:
                dico_one_plot['legend'] = None
        stats['label_nb'] = n+1
        plot_from_stats(stats, plot = (fig,ax_plt), dico_plot = dico_one_plot)
        
        #plot inset
        if(inset_flag):
            dico_one_plot_inset['color'] = listColors[n]
            dico_one_plot_inset['shape'] = listShapes[n]
            if (legend_inset is not None):
                if(n < len(legend)):
                   dico_one_plot_inset['legend'] = legend_inset[n]
                else:
                    dico_one_plot_inset['legend'] = None
            plot_from_stats(stats, plot = (fig,ax_plt2), dico_plot = dico_one_plot_inset)
        
    # Format / save
    apply_dico_plot(fig, ax_plt, dico_main)
    if(inset_flag):
        apply_dico_plot(fig, ax_plt2, dico_inset)
    
    if('hline' in args_extra):        
        dico_line = args_extra['hline']
        x_add = dico_line.pop('x_add', 1)
        x_pad = dico_line.pop('x_pad', 0.3)
        y_pad = dico_line.pop('y_pad', -0.01)

        xmin, xmax = ax_plt.get_xlim()
        ax_plt.set_xlim((xmin, xmax + x_add))
        font = {'family': 'serif', 'color':'darkred','weight': 'normal', 'size': 10}
        for k, v in dico_line.items():
            ax_plt.hlines(v, xmin, xmax, colors='darkred', linestyles='dotted')
            ax_plt.text(xmax + x_pad, v +y_pad , k, fontdict = font)
    
    if(is_str(save_fig)):
        fig.savefig(save_fig)

def apply_dico_plot(fig, axis, dico_plot):
    ylim = dico_plot.get('ylim')
    xlim = dico_plot.get('xlim')
    suptitle = dico_plot.get('suptitle')
    size_label = dico_plot.get('size_label', 10)
    ylabel = dico_plot.get('ylabel')
    xlabel = dico_plot.get('xlabel')
    legend = dico_plot.get('legend')
    legend_prop = dico_plot.get('prop')
    legend_loc = dico_plot.get('legend_loc')
    xticks = dico_plot.get('xticks')
    xtick_label = dico_plot.get('xtick_label')
    size_ticks = dico_plot.get('xtick_size')

    if(legend is not None):
        if(legend_prop is None):
            axis.legend()
        else:
            axis.legend(prop = legend_prop)
    if(legend_loc is not None):
        axis.legend(loc = legend_loc)
    if(ylim is not None):
        axis.set_ylim(ylim[0], ylim[1])
    if(xlim is not None):
        axis.set_xlim(xlim[0], xlim[1])
    if(suptitle is not None):
        fig.suptitle(suptitle)
    if(ylabel is not None):
        axis.set_ylabel(ylabel, fontsize=size_label)
    if(xlabel is not None):
        axis.set_xlabel(xlabel, fontsize=size_label)
    if(xticks is not None):
        axis.set_xticks(xticks)
    if(xtick_label is not None):
        axis.set_xticklabels(xtick_label, fontsize=size_label)
    if(size_ticks is not None):
        axis.tick_params(labelsize = size_ticks)
        
def save_plot(fig, save = None):
    if(is_dico(save)):
        fig.savefig(**save)
    elif(is_str(save)):
        fig.savefig(save, format = 'pdf')
        #bbox_inches='tight'
        


#==============================================================================
#                   is_XXX functions
#==============================================================================
def is_even(value):
    return ((value % 2) == 0)

def is_odd(value):
    return ((value % 2) == 1)

def is_dico(obj):
    return isinstance(obj, dict)

def is_iter(obj):
    return hasattr(obj, '__iter__')

def is_callable(obj):
    return hasattr(obj, '__call__')

def is_str(obj):
    return isinstance(obj, str)

def is_list(obj):
    return isinstance(obj, list)

#==============================================================================
#                   Decorators
# Build generators with arguments
#==============================================================================
#TODO: replace everything with extend_method_to_higher_dim
def vectorise_method(f):
    @wraps(f)
    def vectorized_method(self, arg, **kwargs):
        if hasattr(arg, '__iter__'):
            return np.array([ f(self, x, **kwargs) for x in arg ])
        else:
            return f(self, arg, **kwargs)
    return vectorized_method

#TODO: Decorator debug
def vectorise_method_2D(f):
    @wraps(f)
    def vectorized_method(self, arg, **kwargs):
        if len(np.array(arg).shape)==2:
            return np.array([f(self, x, **kwargs) for x in arg ])
        elif len(np.array(arg).shape)==1:
            return f(self, arg, **kwargs)
        else:
            raise NotImplementedError()
    return vectorized_method

def vectorise_method_ndim(n_dim):
    """ for a method taking an np.array with dim d return a vectorized
    version accepting d+1 dim input
    """
    def vectorize_impl(f):
        @wraps(f)
        def vectorized_method(self, args, **kwargs):
            if len(np.array(args).shape) == n_dim:
                return f(self, args, **kwargs)
            elif len(np.array(args).shape) == (n_dim+1):
                return np.array([f(self, x, **kwargs) for x in args])
            else:
                raise NotImplementedError()
        return vectorized_method
    return vectorize_impl

def extend_dim_method(n_dim=0, array_output = False):
    """ for a method taking as its first positional argunent 
    an object of nb_dim = n_dim extend it s.t. it accepts
    """
    def extend_impl(f):
        @wraps(f)
        def extended_method(self, arg, *args, **kwargs):
            n_dim_args = len(np.shape(arg))
            if (n_dim_args == n_dim):
                res = f(self, arg, *args, **kwargs)
            elif (n_dim_args == (n_dim + 1)):
                res = [f(self, x, *args, **kwargs) for x in arg]
                if array_output:
                    res = np.array(res)
            else:
                raise SystemError(" wrong arg dimensionnality: {}".format(np.shape(arg)))
            return res
        return extended_method
    return extend_impl

def extend_dim_function(n_dim=0, array_output = False):
    """ for a method taking as its first positional argunent 
    an object of nb_dim = n_dim extend it s.t. it accepts
    """
    def extend_impl(f):
        @wraps(f)
        def extended_function(arg, *args, **kwargs):
            n_dim_args = len(np.shape(arg))
            if (n_dim_args == n_dim):
                res = f(arg, *args, **kwargs)
            elif (n_dim_args == (n_dim + 1)):
                res = [f(x, *args, **kwargs) for x in arg]
                if array_output:
                    res = np.array(res)
            else:
                raise SystemError(" wrong arg dimensionnality: {}".format(np.shape(arg)))
            return res
        return extended_function
    return extend_impl


#
@extend_dim_function(1, True)
def qb_get_sph_coord(state):
    r = np.linalg.norm(state)
    if(not(np.allclose(r, 1))):
        logger.error('state is not normalized')
    a, b = state
    mod_a = np.abs(a) 
    arg_a = np.angle(a)
    theta = 2 * np.arccos(mod_a)
    arg_b = np.angle(b)
    phi = arg_b - arg_a
    return (r, theta, phi)


@extend_dim_function(1, True)    
def qb_get_cart_coord(state):
    sph_c = qb_get_sph_coord(state)
    cart_c = sph2cart_coord(sph_c)
    return cart_c


@extend_dim_function(1, True)
def sph2cart_coord(sph_coord):
    r, theta, phi = sph_coord
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return (x,y,z)





#==============================================================================
#                   Decorators
#==============================================================================
if __name__ == '__main__':
    state = np.array([1.0, 1.j]) /np.sqrt(2)

    test_sp = qb_get_sph_coord(state)
    test_eu = qb_get_cart_coord(state)

    
