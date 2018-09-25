#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:58:12 2018

@author: fred
"""
import multiprocessing as mp
import logging
logger = logging.getLogger(__name__)
import pdb
if(__name__ == '__main__'):
    import sys
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
else:
    from .. import Helper as ut

#==============================================================================
#                   management of multiprocessing objects
#==============================================================================
class MPCapability():
    """ Allow the management of MP pool  """
    def __init__(self, **args_MP):
        """ """
        if(args_MP.get('debug')):
            pdb.set_trace()
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
            self.pool = mp.Pool(self._nb_workers)
            self._flag_MP = True
        
        
        else:
            self._flag_MP = False
            self.n_workers = 1
            self.pool = None
    
    def close_mp(self):
        """ Close the pool if it exists"""
        if(self.pool is not None):
            self.pool.close()
    
    @property
    def pool(self):
        return self._pool
    
    @pool.setter
    def pool(self, p):
        self._pool = p
   
    @property
    def n_workers(self):
        return self._nb_workers
    
    @n_workers.setter
    def n_workers(self, n):
        self._nb_workers = n

    @property
    def n_cpus(self):
        if(self._flag_MP):
            n = mp.cpu_count()
        else:
            n =0
        return n
    
    @classmethod
    def init_mp(cls, mp_object = None):
        """ Return a MPGenar. Can deal with multiple input
        TODO: extend at some point"""
        if(mp_object.__class__ == cls):
            return mp_object
        else:
            return MPCapability(flag_MP = mp_object)
        

    def map_custom(self, f, list_args):
        if(self.pool is None):
            res = [f(a) for a in list_args]
        else:
            try:
                res = self.pool.starmap(f, list_args)
            except Exception as e:
                logging.error('Error occurred : {0}... closing mp.pool'.format(e))
                self.close_mp()
                self.pool = None
                self._flag_MP = False
                self.n_workers = 1
                res = [f(a) for a in list_args]
        return res

# =========================================================================== #
# TODO: testing mp capabilities
# =========================================================================== #        
if __name__ == "__main__":
    pass
    