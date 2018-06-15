#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:58:12 2018

@author: fred
"""
import multiprocessing as mp
if(__name__ == '__main__'):
    import sys
    sys.path.append("../")
    import Helper as ut
else:
    from .. import Helper as ut

#==============================================================================
#                   management of multiprocessing objects
#==============================================================================
class MPCapability():
    """ Allow the management of MP pool  """
    def __init__(self, **args_MP):
        """ """
        flagMP = args_MP.get('flag_MP')
        
        if(isinstance(flagMP, bool)):
            if(flagMP):
                self._flag_MP = True
                self.n_workers = max(1, self.n_cpus -1)
                self.pool = mp.Pool(self.n_workers)
                
            else:
                self._flag_MP = False
                self.n_workers = 1
                self.pool = None
                
        
        elif(isinstance(flagMP, int)):
            self._nb_cpus = mp.cpu_count()
            self._nb_workers = int(flagMP)
            self._Pool = mp.Pool(self._nb_workers)
            self._flag_MP = True
        
        else:
            raise NotImplementedError()
    
    def close_mp(self):
        """ Close the pool if it exists"""
        if(self.pool is not None):
            self.pool.close()
    
    @property
    def pool(self):
        return self._pool
    
    @property.setter
    def pool(self, p):
        self._pool = p
   
    @property
    def n_workers(self):
        return self._nb_workers
    
    @property.setter
    def n_workers(self, n):
        self._nb_workers = n

    @property
    def n_cpus(self):
        if(self._flag_MP):
            n = mp.cpu_count()
        else
            n =0
        return n
    
    @classmethod
    def init_mp(cls, mp_object = None):
        """ Return a MPGenar. Can deal with multiple input
        TODO: extend at some point"""
        if(mp_object.__class__ == cls):
            return rdm_object
        else:
            return MP(flag_MP = mp_object)
        



# =========================================================================== #
# TODO: testing mp capabilities
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
    