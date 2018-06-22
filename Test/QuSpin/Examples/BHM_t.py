#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:31:54 2018

@author: fred
"""

import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')

from quspin.tools import measurements
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
import matplotlib.pylab as plt
#

## ================================== ##
## Some ad-hoc functions
## ================================== ##
def variance_op(O, V):
    OV = O.dot(V)
    VOOV = np.asscalar(O.matrix_ele(V, OV))
    VOV2 = O.expt_value(V) ** 2
    var = VOOV -VOV2
    return var

def avg_variange_op(list_O, V):
    list_var = [variance_op(O, V) for O in list_O]
    return np.average(list_var)

def gen_linear_ramp(v, T = 1, ymin=0, ymax=1):
    ramp = (lambda t: ymin + v * t)
    T = (ymax - ymin)/v
    return ramp, T

def ip(V1, V2):
    return np.dot(np.conj(np.squeeze(V1)), np.squeeze(V2))

def fid(V1, V2):
    return np.square(np.abs(ip(V1, V2)))


def gen_ramped_h(basis, v = 1, L = 5, mu = 0.0):
    U, T = gen_linear_ramp(v)
    J = (lambda t: 1 - U(t))
    args_U, args_J = [], []
    hop = [[-1,i,(i+1)%L] for i in range(L)] #PBC
    dynamic_hop = [['+-', hop, J, args_J],['-+',hop, J, args_J]]
    inter_nn = [[0.5, i, i] for i in range(L)]
    inter_n = [[-0.5, i] for i in range(L)]
    dynamic_inter = [['nn', inter_nn, U, args_U], ['n', inter_n, U, args_U]]
    dynamic = dynamic_inter + dynamic_hop
    pot_n =  [[mu, i] for i in range(L)]
    static = [['n', pot_n]]

    H=hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    return H, T


## ================================== ##
## Hamiltonian
## ================================== ##
L, mu, v =5, 0.0, 0.1 # system size
symH = {'kblock':0, 'pblock':1}
no_check_sym = {"check_symm" : False}

basis_sym = boson_basis_1d(L,Nb=L, sps=3,**symH)
basis = boson_basis_1d(L,Nb=L, sps=3)

H, T = gen_ramped_h(basis, v, L, mu)
H_sym, T = gen_ramped_h(basis_sym, v, L, mu)





## ================================== ##
## Comparaison between sym/nosym
## Look at the GS at init, final
## Can I build the Dispersion relation??
## 
## ================================== ##
n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis, dtype=np.float64, **no_check_sym) for i in range(L)]
n_sites_sym = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis_sym, dtype=np.float64, **no_check_sym) for i in range(L)]

### Look at GS in SF and MI modes
print('with sym:')
Htmp = H_sym
E_SF_sym, V_SF_sym = Htmp.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10) # only GS
E_MI_sym, V_MI_sym = Htmp.eigsh(time = T, k=1, which='SA',maxiter=1E10) # only GS
print('E_SF with sym: ' + str(E_SF_sym))
print('E_MI with sym: ' + str(E_MI_sym))

var_n_MI_sym = avg_variange_op(n_sites_sym, V_MI_sym)
var_n_SF_sym = avg_variange_op(n_sites_sym, V_SF_sym)
print('varn SF with sym: ' + str(var_n_SF_sym))
print('varn MI with sym: ' + str(var_n_MI_sym))



print('withOUT sym:')
Htmp = H
E_SF, V_SF = Htmp.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10) # only GS
E_MI, V_MI = Htmp.eigsh(time = T, k=1, which='SA',maxiter=1E10) # only GS
print('E_SF without sym: ' + str(E_SF))
print('E_MI without sym: ' + str(E_MI))

var_n_MI = avg_variange_op(n_sites, V_MI)
var_n_SF = avg_variange_op(n_sites, V_SF)
print('varn SF without sym: ' + str(var_n_SF))
print('varn MI without sym: ' + str(var_n_MI))




## Change of basis
#vector
P = basis_sym.get_proj(np.float64, pcon=True)
V_test_SF = np.sum(np.abs(P.dot(V_SF_sym) + V_SF))
V_test_MI = np.sum(np.abs(P.dot(V_MI_sym) - V_MI))

print('change of basis discrpancies vectors: ' + str(V_test_SF) + ' ' + str(V_test_MI))



## ================================== ##
## Now evolve Initial state over time and observe stuff
## ================================== ##
psi_i = V_SF
psi_f = H.evolve(psi_i, 0, T)

I1 = avg_variange_op(n_sites, psi_f)
I2 = fid(V_MI, psi_f)

print(np.sqrt(I1))
print(I2)


## ================================== ##
##  TESTING ZONE
## ================================== ##
# Basis with sym
b_test = boson_basis_1d(L,Nb=L,sps=3, kblock=0, pblock=1)

# get_proj transfo matrix 
P = b_test.get_proj(np.float64,pcon=False)
P_full = P.toarray()


# get vector from reduced basis to full basis
print(b_test)
i0 = b_test.index("22100") # pick state from basis set
V0 = np.zeros(b_test.Ns,dtype=np.float64)
V0[i0] =1.0
V0_full = b_test.get_vec(V0, False)



symH = {'kblock':0, 'pblock':1}
basis_full = boson_basis_1d(L,Nb=L, sps=3)
basis_reduced = boson_basis_1d(L,Nb=L, sps=3,**symH)
P = basis_reduced.get_proj(np.float64, pcon=True)

no_check_sym = {"check_symm" : False}
n_full = hamiltonian([['n', [[1.0, 0]]]], [], basis = basis_full, **no_check_sym)
n_reduced = hamiltonian([['n', [[1.0, 0]]]], [], basis = basis_reduced, **no_check_sym)


H_reduced, T = gen_ramped_h(basis_reduced, v = 1, L = 5, mu = 0.0)
_, V_reduced = H_reduced.eigsh(time=0.0, k = 1, which = 'SA', maxiter = 1E10)
V_recover = P.dot(V_reduced)

H_full, T = gen_ramped_h(basis_full, v = 1, L = 5, mu = 0.0)
_, V_full = H_full.eigsh(time=0.0, k = 1, which = 'SA', maxiter = 1E10)

ip(V_full, V_recover)

n_reduced.expt_value(V_reduced)
n_full.expt_value(V_full)




n_recover = n_reduced.project_to(P)

n_recover.todense()[0:2, 0:2]
n_full.todense()[0:2, 0:2]
