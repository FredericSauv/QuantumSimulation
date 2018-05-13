#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:31:54 2018

@author: fred
"""

import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import matplotlib.pylab as plt
#

##Create Hamiltonian
L, J, U, mu =5, 1, 50, 0.0 # system size
basis = boson_basis_1d(L,Nb=L,sps=3)

hop=[[-J,i,(i+1)%L] for i in range(L)] #PBC
interact=[[0.5*U,i,i] for i in range(L)] # U/2 \sum_j n_j n_j
pot=[[-mu-0.5*U,i] for i in range(L)] # -(\mu + U/2) \sum_j j_n
static=[['+-',hop],['-+',hop],['n',pot],['nn',interact]]
dynamic=[]
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)



## Compute variance of the number particle for site 0 for the GS
E,V=H.eigsh(k=1,which='SA',maxiter=1E10) # only GS

n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis,dtype=np.float64) for i in range(L)]
n_var_sites = [variance(op, V) for op in n_sites]
measure_n = np.average(n_var_sites)

def variance(O, V):
    OV = O.dot(V)
    VOOV = np.asscalar(O.matrix_ele(V, OV))
    VOV2 = O.expt_value(V) ** 2
    var = VOOV -VOV2
    return var








