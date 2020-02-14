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

A = qt.tensor(qt.projection(2,0,0), Z) + qt.tensor(qt.projection(2,1,1), X)
ut.schm_decompo(A)
B = qt.tensor(qt.projection(2,0,0), X) + qt.tensor(qt.projection(2,1,1), Z)
C = qt.tensor(plus*plus.dag(), X) + qt.tensor(minus*minus.dag(), Z)
D = qt.tensor(plus*plus.dag(), Z) + qt.tensor(minus*minus.dag(), X)







##### State to look at 
#1q stuff
q1_pauli = [I, X, Y, Z]
q1_state_non_aligned = 1/2 * zero + np.sqrt(3)/2 * one

#2q stuff
q2_sep = qt.tensor(zero, one)
q2_sep_nonaligned = qt.tensor(*[q1_state_non_aligned]*2)
q2_haar = qt.rand_unitary_haar(4, [[2,2],[2,2]]) * q2_sep
q2_ghz = 1/np.sqrt(2) * (qt.tensor(zero, zero) + qt.tensor(one, one))
q2_W = 1/np.sqrt(2) * (qt.tensor(zero, one) + qt.tensor(one, zero))

#3q stuff
q3_sep = qt.tensor(zero, one)
q3_sep_nonaligned = qt.tensor(*[q1_state_non_aligned]*3)
q3_haar = qt.rand_unitary_haar(8, [[2,2],[2,2],[2,2]]) * q3_sep
q3_ghz = 1/np.sqrt(2) * (qt.tensor(zero, zero, zero) + qt.tensor(one, one, one))
q3_W = 1/np.sqrt(3) * (qt.tensor(zero, zero, one)+ qt.tensor(one, zero, zero) + qt.tensor(zero, one, zero))

##### Estimation with different basis
def subabs0(state, N):
    # Use initial pauli basis
    # sum(abs(expectation values of pauli tensors elements))
    basis_obs = it.product(*[q1_pauli]*N)
    expectations =  [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))

def sumabs1(params, state, N):
    # rotate the local hilbert spaces (same for each qubit)
    # if params = [0,0] nothing is done
    U = unitary2(params)
    basis_obs_1q = [U * p * U.dag() for p in q1_pauli]
    basis_obs = it.product(*[basis_obs_1q]*N)
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))
    
def sumabs2(params, state, N):
    # rotate the local hilbert spaces (different for each qubit)
    # if params = [0,0,..., 0,0] nothing is done
    list_U = [unitary2(params[2*i, 2*i+2]) for i in range(N)]
    basis_obs = it.product(*[[U * p * U.dag() for p in q1_pauli] for U in list_U])
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in basis_obs]
    return np.sum(np.abs(expectations))


def sumabs3(params, state, N):
    # Find orthogonal transfo to be applied on the set of 1q pauli operators
    U = param_44(params)
    new_1q_basis = transfo_pauli(U)
    param_basis = it.product(*[new_1q_basis]*N)
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in param_basis]
    return np.sum(np.abs(expectations))


def sumabs4(params, state, N):
    # Find orthogonal transfo to be applied on the set of 1q pauli operators
    U = [param_44([params[4*i], params[4*i+1],params[4*i+2],params[4*i+3]]) for i in range(N)]
    new_1q_basis = [transfo_pauli(u) for u in U]
    param_basis = it.product(*new_1q_basis)
    expectations = [qt.tensor(*b).matrix_element(state, state) for b in param_basis]
    return np.sum(np.abs(expectations))

def unitary2(params):
    # A parametrized U(2)
    theta, phi = params[0], params[1]    
    qt.operators.Qobj([[np.cos(theta/2), np.exp(1.j * phi) * np.sin(theta/2)],
                       [-np.sin(theta/2), np.exp(1.j* phi) * np.cos(theta/2)]])

def transfo_pauli(M):
    ## action of the superoperator M on each element of the set of Pauli observables
    P_vect = np.array([ut.vect(p) for p in q1_pauli])
    return [qt.operators.Qobj(ut.devect(p),[[2,2],[2,2]]) for p in M.dot(P_vect)]

def ortho4_4p(params):
    d, a, b, g = params[0], params[1],params[2], params[3]
    v = np.sqrt(d*d +a*a + b*b + g*g)
    M = 1/v * np.array([[d, a, b, g], [-a, d, g, -b],[-b, -g, d, a], [-g, b, -a, d]])
    return M



## 2dof
state_of_interest = q2_state2

f = lambda x:sumabs(x, state_of_interest ,2)
x_0 = np.zeros(2)
x_init = np.random.uniform(low=0., high=2*np.pi, size=2)
f(x_init)
f(x_0)
res = opt.minimize(f, x_init)
    
## 2N dof    
f2 = lambda x:sumabs2(x, q2_state2, 2)
x_02 = np.zeros(4)
x_init2 = np.random.uniform(low=0., high=2*np.pi, size=4)
f2(x_02)
f2(x_init2)

res2 = opt.minimize(f2, x_init2)

## 4 dof    
f3 = lambda x:sumabs3(x, q2_state2, 2)
x_03 = np.ones(4)
x_init3 = np.random.uniform(low=-1., high=1., size=4)
f3(x_03)
f3(x_init3)

res3 = opt.minimize(f3, x_init3)

## 4N dof    
f4 = lambda x:sumabs4(x, q2_state2, 2)
x_04 = np.ones(8)
x_init4 = np.random.uniform(low=-1., high=1., size=8)
f4(x_04)
f4(x_init3)

res4 = opt.minimize(f4, x_init4)


