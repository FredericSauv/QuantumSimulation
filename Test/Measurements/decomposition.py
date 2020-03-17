#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:34:55 2019

@author: fred
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import utilities as ut
import itertools as it
import scipy.optimize as opt

#qubits
zero = qt.qubits.qubit_states(1,[0])
one = qt.qubits.qubit_states(1,[1])

I, X, Y, Z = qt.identity([2]), qt.sigmax(), qt.sigmay(), qt.sigmaz()
Rx, Ry, Rz = qt.rx, qt.ry, qt.rz

plus, minus = X.eigenstates()[1][1], X.eigenstates()[1][0]
ghz1 = ut.get_ghz(2)


###########
# Decomp Hermitian operators Schmidt
###########
H0 = qt.tensor(X, X) + qt.tensor(Y, Y)

A = qt.tensor(zero*zero.dag(), Z) + qt.tensor(one*one.dag(), X)
B = qt.tensor(zero*zero.dag(), X) + qt.tensor(one*one.dag(), Z)
C = qt.tensor(plus*plus.dag(), X) + qt.tensor(minus*minus.dag(), Z)
D = qt.tensor(plus*plus.dag(), Z) + qt.tensor(minus*minus.dag(), X)






###########
# Optimize observables basis
###########
##### State to look at 
#1q stuff
q1_pauli = [I, X, Y, Z]
q1_haar = qt.rand_unitary_haar(2, [[2],[2]]) * zero
q1_haar2 = qt.rand_unitary_haar(2, [[2],[2]]) * zero
q1_haar3 = qt.rand_unitary_haar(2, [[2],[2]]) * zero

#2q stuff
q2_sep = qt.tensor(zero, one)
q2_sep_haar = qt.tensor(*[q1_haar, q1_haar2])
q2_haar = qt.rand_unitary_haar(4, [[2,2],[2,2]]) * q2_sep
q2_ghz = 1/np.sqrt(2) * (qt.tensor(zero, zero) + qt.tensor(one, one))
q2_W = 1/np.sqrt(2) * (qt.tensor(zero, one) + qt.tensor(one, zero))

#3q stuff
q3_sep = qt.tensor(zero, one, one)
q3_sep_haar = qt.tensor(*[q1_haar,q1_haar2,q1_haar3])
q3_haar = qt.rand_unitary_haar(8, [[2,2,2],[2,2,2]]) * q3_sep
q3_ghz = 1/np.sqrt(2) * (qt.tensor(zero, zero, zero) + qt.tensor(one, one, one))
q3_W = 1/np.sqrt(3) * (qt.tensor(zero, zero, one)+ qt.tensor(one, zero, zero) + qt.tensor(zero, one, zero))

##### Estimation with different basis
##
def unitary2(params):
    # A parametrized U(2)
    theta, phi = params[0], params[1]    
    U = qt.operators.Qobj([[np.cos(theta/2), np.exp(1.j * phi) * np.sin(theta/2)],
                       [-np.sin(theta/2), np.exp(1.j* phi) * np.cos(theta/2)]])
    return U
def transfo_pauli(M):
    ## action of the superoperator M on each element of the set of Pauli observables
    P_vect = np.array([ut.vect(p) for p in q1_pauli])
    return [qt.operators.Qobj(ut.devect(p),[[2,2],[2,2]]) for p in M.dot(P_vect)]

def ortho4_4p(params):
    d, a, b, g = params[0], params[1],params[2], params[3]
    v = np.sqrt(d*d +a*a + b*b + g*g)
    M = 1/v * np.array([[d, a, b, g], [-a, d, g, -b],[-b, -g, d, a], [-g, b, -a, d]])
    return M

#strategies
def sumabs0(state, N=2):
    # Use initial pauli basis
    # sum(abs(expectation values of pauli tensors elements))
    basis_obs = it.product(*[q1_pauli]*N)
    expectations =  [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))

def sumabs1(params, state, N=2):
    # rotate the local hilbert spaces (same for each qubit)
    # if params = [0,0] nothing is done
    U = unitary2(params)
    basis_obs_1q = [U * p * U.dag() for p in q1_pauli]
    basis_obs = it.product(*[basis_obs_1q]*N)
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))
    
def sumabs2(params, state, N=2):
    # rotate the local hilbert spaces (different for each qubit)
    # if params = [0,0,..., 0,0] nothing is done
    list_U = [unitary2(params[2*i: 2*i+2]) for i in range(N)]
    basis_obs = it.product(*[[U * p * U.dag() for p in q1_pauli] for U in list_U])
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))


def sumabs3(params, state, N=2):
    # Find orthogonal transfo to be applied on the set of 1q pauli operators
    U = ortho4_4p(params)
    basis_obs_1q = transfo_pauli(U)
    basis_obs = it.product(*[basis_obs_1q]*N)
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))


def sumabs4(params, state, N=2):
    # Find orthogonal transfo to be applied on the set of 1q pauli operators
    list_U = [ortho4_4p(params[4*i:4*i+4]) for i in range(N)]
    basis_obs = it.product(*[transfo_pauli(U) for U in list_U])
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))



##### Look at sum abs for different strategies
state_of_interest = q3_ghz
N_of_interest = 3

p_zeros = np.zeros(20)
p_rdm_pi = np.random.uniform(low=0., high=2*np.pi, size=20)
p_rdm_one = np.random.uniform(low=-1., high=1., size=20)
p_one = np.zeros(20)
p_one[::4] = np.ones(5)

## no optim
f0 = lambda x: sumabs0(state_of_interest, N_of_interest)
res0 = f0(state_of_interest)
  
## optim global H-space
f1 = lambda x:sumabs1(x, state_of_interest, N_of_interest)
#print(f1(p_zeros),f1(p_rdm_pi))
res1 = opt.minimize(f1, p_rdm_pi[:2])

## optim local H-space
f2 = lambda x:sumabs2(x, state_of_interest, N_of_interest)
#print(f2(p_zeros),f2(p_rdm_pi))
res2 = opt.minimize(f2, p_rdm_pi[:2* N_of_interest])

## 4 dof    
f3 = lambda x:sumabs3(x, state_of_interest, N_of_interest)
#print(f3(p_one), f3(p_rdm_one))
res3 = opt.minimize(f3, p_rdm_one[:4])


## 4N dof    
f4 = lambda x:sumabs4(x, state_of_interest, N_of_interest)
# print(f4(p_one), f4(p_rdm_one))
res4 = opt.minimize(f4, p_rdm_one[:4* N_of_interest])

