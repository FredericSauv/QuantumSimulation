#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:06:47 2019

@author: fred
"""
import numpy as np
from qutip import sigmax, sigmay, sigmaz, identity, tensor, basis, ket2dm
import itertools as it


def ip(A, B):
    return (A.dag() * B).tr()

def proj(listop, basis):
    return [[ip(b, op) for b in basis] for op in listop]

X, Y, Z, I = sigmax(), sigmay(), sigmaz(), identity(2)
b_pauli = [1/np.sqrt(2)*X, 1/np.sqrt(2)*Y, 1/np.sqrt(2)*Z, 1/np.sqrt(2)*I]
b1 = [tensor(A,B) for A, B in it.product(b_pauli, b_pauli)] #bais of unitary operator


zero1, one1 = basis(2, 0), basis(2,1)
plus1, minus1 = 1/np.sqrt(2) * (zero1 + one1), 1/np.sqrt(2) * (zero1 - one1)
st = [zero1, one1, plus1, minus1]
b2 = [ket2dm(tensor(A, B)) for A, B in it.product(st, st)] #basis of input states

cnot = tensor(ket2dm(zero1), I) + tensor(ket2dm(one1), X)
U = cnot

A = np.array(proj(b1, b2))
meas = [np.sum([A[k,l] * U * b1[k] * U.dag() for k in range(A.shape[0])])  for l in range(A.shape[1])]
isherm = np.all([m.isherm for m in meas])

def is_local(meas):
    
    
    
def is_entangled(state):
    rho = ket2dm(state).ptrace(0)
    return np.allclose(1.0, (rho * rho))