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

## ================================== ##
## System meta-parameters
## ================================== ##
L, mu =5, 0.0 # system size


#basis = boson_basis_1d(L,Nb=L, sps=3)
basis = boson_basis_1d(L,Nb=L, sps=3, kblock=0, pblock=1)


## ================================== ##
## Some ad-hoc functions
## ================================== ##
def variance(O, V):
    OV = O.dot(V)
    VOOV = np.asscalar(O.matrix_ele(V, OV))
    VOV2 = O.expt_value(V) ** 2
    var = VOOV -VOV2
    return var

def gen_linear_ramp(v, T = 1, ymin=0, ymax=1):
    ramp = (lambda t: ymin + v * t)
    T = (ymax - ymin)/v
    return ramp, T

def ip(V1, V2):
    return np.dot(np.conj(np.squeeze(V1)), np.squeeze(V2))

def fid(V1, V2):
    return np.square(np.abs(ip(V1, V2)))

## ================================== ##
## Hamiltonian
## ================================== ##
v = 0.1
U, T = gen_linear_ramp(v)
J = (lambda t: 1 - U(t))
args_U, args_V = [], []
args_J = []

hop = [[-1,i,(i+1)%L] for i in range(L)] #PBC
dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]



inter_nn = [[0.5, i, i] for i in range(L)]
inter_n = [[-0.5, i] for i in range(L)]
dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]

dynamic = dynamic_inter + dynamic_hop

pot_n =  [[mu, i] for i in range(L)]
static = [['n', pot_n]]

H=hamiltonian(static, dynamic, basis=basis, dtype=np.float64)


## ================================== ##
## Look at some instantaneous properties of the ESpectrun
## Look at the GS at init, final
## Can I build the Dispersion relation
## ================================== ##
## Define operator we want to observe over time
no_sym = {"check_symm" : False}
n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis, dtype=np.float64, **no_sym) for i in range(L)]

# n_sites = [basis.Op('n', [i], 1.0, dtype=np.float64) for i in range(basis.N)]
def avg_var_occup(V):
    n_var_sites = [variance(op, V) for op in n_sites]
    avg_var_occup = np.average(n_var_sites)
    return avg_var_occup


E_SF, V_SF = H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10) # only GS
E_MI, V_MI = H.eigsh(time = T, k=1, which='SA',maxiter=1E10) # only GS
print(E_SF)
print(E_MI)

print(avg_var_occup(V_SF))
print(avg_var_occup(V_MI))


## ================================== ##
## Now evolve Initial state over time and observe stuff
## ================================== ##
psi_i = V_SF
psi_f = H.evolve(psi_i, 0, T)

I1 = avg_var_occup(psi_f)
I2 = fid(V_MI, psi_f)

print(I1)
print(I2)


## ================================== ##
##  TESTING ZONE
## ================================== ##
# Basis with sym
b_test = boson_basis_1d(L,Nb=L,sps=3, kblock=0,pblock=1)

# get_proj transfo matrix 
P = b_test.get_proj(np.float64,pcon=False)
P_full = P.toarray()


# get vector from reduced basis to full basis
print(b_test)
i0 = b_test.index("22100") # pick state from basis set
V0 = np.zeros(b_test.Ns,dtype=np.float64)
V0[i0] =1.0
V0_full = b_test.get_vec(V0, False)




