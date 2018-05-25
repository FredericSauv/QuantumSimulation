#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:28:04 2018

@author: fred
"""
import sys
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions


# generate an Hamiltontian without interaction
def gen_H_no_inter(basis, L=5, J=1.0):
    hop=[[-J,i,(i+1)%L] for i in range(L)] #PBC
    static=[['+-',hop],['-+',hop]]
    H=hamiltonian(static,[],basis=basis,dtype=np.float64)
    return H

# generate operator for one site (by defaut 'n0')
def gen_operator_site(basis, op_string = 'n', site = 0):
    no_check_sym = {"check_symm" : False}    
    return hamiltonian([[op_string, [[1.0, site]]]], [], basis = basis, **no_check_sym)



# Basis without symmetry, get the ground state and compute expected value of n0 
# for this state
basis = boson_basis_1d(5, Nb=5, sps=3)
H = gen_H_no_inter(basis)
EGS, VGS = H.eigsh(time=0.0, k = 1, which = 'SA', maxiter = 1E10)
n0 = gen_operator_site(basis) 
print(n0.expt_value(VGS))


# Basis with symmetry
symH = {'kblock':0, 'pblock':1}
basis_sym = boson_basis_1d(5, Nb=5, sps=3, **symH)
H_sym = gen_H_no_inter(basis_sym)
EGS_sym, VGS_sym = H_sym.eigsh(time=0.0, k = 1, which = 'SA', maxiter = 1E10)
n0_sym = gen_operator_site(basis_sym) 
print(n0_sym.expt_value(VGS_sym))


# Recovering VGS from VGS_sym
P = basis_sym.get_proj(np.complex128, pcon=True)
VGS_recover = P.dot(VGS_sym)  # sometimes with a pi phase
print(np.allclose(VGS, VGS_recover))


# Recovering n0 from n0_sym
n0_recover = n0_sym.project_to(P.transpose())
print(np.allclose(n0.todense(), n0_recover.todense()))