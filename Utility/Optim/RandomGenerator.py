#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:58:12 2018

@author: fred
"""
import numpy as np
import numpy.random as rdm
import os
import pdb

if(__name__ == '__main__'):
    import sys
    sys.path.append("../")
    import Helper as ut
else:
    import Helper as ut
#==============================================================================
#                   MANIP STRUCTURES (list / dico / text files)
#==============================================================================
class RandomGenerator(rdm.RandomState):
    """
    Purpose:
        Allow the management of the random generation (wrapper of numpy random)
    """
    def __init__(self, seed = None):
        """
        Purpose:
            Initialize a numpy.random.RandomState
        """
        rdm.RandomState.__init__(self, seed)


    def reseed(self, seed = None):
        """ Re-seed the generator
        """
        rdm.seed(seed)
    
    @classmethod
    def init_random_generator(cls, rdm_object = None):
        """ Return a RandomGenerator. Can deal with multiple input: Seed/
        RandomGenerator/None
        """
        if(rdm_object is None):
            return RandomGenerator() 
            
        elif(ut.is_int(rdm_object)):
            return RandomGenerator(seed = rdm_object)
            
        elif(rdm_object.__class__ == cls):
            return rdm_object
        
        else:
            raise NotImplementedError()
        
    @classmethod
    def gen_seed(cls, nb = 1):
        """ Return a RandomGenerator. Can deal with multiple input: Seed/
        RandomGenerator/None
        """
        if(nb>1):
            seed = [cls.gen_seed() for _ in range(nb)]
        else:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
        return seed

      
    def gen_rdmnb_from_string(self, method_rdm, dim = 1):
        """
        #TODO: change name gen_rdm_XXX // use genrdmdunction // 
        #Old name = GenRdmNumbersFromStr
        Purpose:
            Generate Random Sequences based on a string or list of string
            
            {X0,   XN-1} is represented as a N row 
                                             NxD matrix if each RandomVar has dim D
            More generally dim = [dim_pop, dim_RV]
        """ 
        functmp = self.gen_rdmfunc_from_string(method_rdm, dim)
        if(ut.is_list(functmp)):
            res = [f() for f in functmp]
        else:
            res = functmp()
            
        return res



    def gen_rdmfunc_from_string(self, method_rdm, dim = 1):
        """return a function that when called return random variables according
        to the distribution described by the string, with specific dim
        convention: dim[-1] correspond to the dim of the RV while dim[:-1] to 
        the size of the population
        TODO: works only for 1D dim
        """ 
        if ut.is_list(method_rdm):
            # return a lst of function
            # should it be a function returning a list??
            func = [self.gen_rdmfunc_from_string(meth, dim) for meth in method_rdm]
        else:
            
            args = ut.splitString(method_rdm)
            if(ut.is_int(dim)):
                dimRV = dim
            else:
                dimRV = dim[-1]
            if(args[0] in ['uniform', 'normal']):
                if(len(args) == 1):
                    first_args = np.repeat(0, dimRV)
                    second_args = np.repeat(1, dimRV)
                    
                elif(len(args) == 3):
                    first_args = np.repeat(float(args[1]), dimRV)
                    second_args = np.repeat(float(args[2]), dimRV)       
                
                elif(len(args) == (1 + 2 * dimRV)):
                    first_args = np.array([float(args[1+2*d]) for d in range(dimRV)])
                    second_args = np.array([float(args[2+2*d]) for d in range(dimRV)])
                else:
                    raise NotImplementedError()
                
                if(dim == 1):
                    # such that function return a value instead of an array
                    # may change
                    first_args, second_args = first_args[0], second_args[0]
                    dim = None
                if args[0] == 'normal':
                    def func():
                        return self.normal(first_args, second_args, size = dim)
                else:
                    def func():
                        return self.uniform(first_args, second_args, size = dim)
            
            elif(args[0] == 'determ'):
                if(len(args) == 1):
                    constant = np.repeat(0, dim)
                elif(len(args) == 2):
                    constant = np.repeat(float(args[1]), dim)
                elif(len(args) == (1 + dimRV)):
                    constant = np.array(args[1:])
                    if(ut.is_list(dim)):
                        dim_pop = np.product(dim[:-1])
                        constant = np.tile(constant, dim_pop).reshape(dim)
                else:
                    raise NotImplementedError()
                def func():
                    return constant
            else:
                raise NotImplementedError()
        return func

    def init_population_lhs(self, population_shape, limits = None):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        FROM SCIPY.DIFFERENTIAL.EVOLUTION
        """
        if (limits is None):
            shift = 0
            scale = 1            
        else:
            shift = 0.5 * (limits[0] + limits[1] - 1)
            scale = np.fabs(limits[0] - limits[1])
        
        nb_params = population_shape[1]
        num_population_members = population_shape[0]
        segsize = 1.0 / num_population_members
        samples = (segsize * self.random_sample(population_shape)
                   + np.linspace(0., 1., num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        population = np.zeros_like(samples)

        for j in range(nb_params):
            order = self.permutation(range(num_population_members))
            population[:, j] = samples[order, j]
        population = population * scale + shift
        
        return population



# =========================================================================== #
        
if __name__ == "__main__":
    rgen = RandomGenerator()    
    func = rgen.gen_rdmfunc_from_string('uniform_0_1_0_10', dim = [5,2])
    res = func()
    print(res)
    func = rgen.gen_rdmfunc_from_string('normal_0_1_0_10', dim = [2,5,2])
    res = func()
    print(res[:,:,0])
    print(res[:,:,1])
    func = rgen.gen_rdmfunc_from_string('determ_0_10', dim = [2,5,2])
    res = func()
    print(res[:,:,0])
    print(res[:,:,1])
    func = rgen.gen_rdmfunc_from_string('normal', dim = 2)
    res = func()
    print(res)  
    func = rgen.gen_rdmfunc_from_string('normal', dim = 1)
    res = func()
    print(res)  
    