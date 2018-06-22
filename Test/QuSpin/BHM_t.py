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


sys.path.insert(0, '../../../')
from QuantumSimulation.Utility import Helper as ut
from QuantumSimulation.Utility.Optim import pFunc_base as pf
from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator as rg

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
    T = 1/v
    U = pf.LinearFunc(w= v, bias = 0)#gen_linear_ramp(v)
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
L, mu, v = 5, 0.0, 0.01 # system size
symH = {'kblock':0, 'pblock':1}
no_check_sym = {"check_symm" : False}

basis = boson_basis_1d(L,Nb=L, sps=3,**symH)


H, T = gen_ramped_h(basis, v, L, mu)

E_SF, V_SF = H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10) # only GS

psi_i = V_SF
psi_f = H.evolve(psi_i, 0, T)


