#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:31:17 2019

@author: fred
"""

import qutip as qt
import numpy as np
import itertools as it

zero, one = qt.qubits.basis(2,0), qt.qubits.basis(2,1)
plus, minus = 1/np.sqrt(2) * (zero + one), 1/np.sqrt(2) * (zero - one)


### ====================================================================== ###
### STUDY 1: How many non null elements when characterising entangled states
###          Bell: 4 GHZ: 27
###          Is there a better basis
### ====================================================================== ###

def recursive_tensor(N, list_one, list_current):
    if(N==1):
        return list_current
    else:
        list_new = list(it.chain(*[[qt.tensor(c, o) for o in list_one] for c in list_current]))
        return recursive_tensor(N-1, list_one, list_new)

def recursive_name(N, list_one, list_current):
    if(N==1):
        return list_current
    else:
        list_new = list(it.chain(*[[c+ o for o in list_one] for c in list_current]))
        return recursive_name(N-1, list_one, list_new)


def tensor_paulis(N=1):
    list_paulis = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    list_names = ['I', 'X', 'Y', 'Z']
    tens = recursive_tensor(N, list_paulis, list_paulis)
    names = recursive_name(N, list_names, list_names)
    return names, tens


## 2 qubits
bell = 1/np.sqrt(2) * (qt.tensor(zero, zero) + qt.tensor(one, one))
tp_name2, tp_op2 = tensor_paulis(2)
tgt2 = {n: op.matrix_element(bell, bell) for n, op in zip(tp_name2, tp_op2)}

## 3 qubits
ghz = 1/np.sqrt(2) * (qt.tensor(zero, zero, zero) - qt.tensor(one, one, one))
tp_name3, tp_op3 = tensor_paulis(3)
tgt3 = {n: op.matrix_element(ghz, ghz) for n, op in zip(tp_name3, tp_op3)}

for k, v in tgt3.items():
    if np.abs(v)>1e-6:
        print(k, v)
        
## 3 qubits
W = 1/np.sqrt(3) * (qt.tensor(zero, zero, one) + qt.tensor(zero, one, zero)+ qt.tensor(one, zero, zero))
tgt3_W = {n: op.matrix_element(W, W) for n, op in zip(tp_name3, tp_op3)}

for k, v in tgt3_W.items():
    if np.abs(v)>1e-6:
        print(k, v)

## 4 qubits
ghz4 = 1/np.sqrt(2) * (qt.tensor(zero, zero, zero,zero) - qt.tensor(one, one, one,one))
tp_name4, tp_op4 = tensor_paulis(4)
tgt4 = {n: op.matrix_element(ghz4, ghz4) for n, op in zip(tp_name4, tp_op4)}

for k, v in tgt4.items():
    if np.abs(v)>1e-6:
        print(k, v)
        


### ====================================================================== ###
### STUDY 2: We try to find proxies for FoM
###          
###          
### ====================================================================== ###
def meas_1_obs(obs, state, nb_meas, closed = True, nb_r = 1):
    """ Estimated the expected value of an observable for a given state"""
    ev, EV = obs.eigenstates()
    proba = [(E.dag() * state * E).tr()  for E in EV]
    assert np.all(np.imag(proba)<1e-6), "Imaginary proba"
    if closed:
        nb_m = nb_meas
        freq_0 = np.random.binomial(nb_m, np.real(proba[0]), (nb_r, 1)) / nb_m
        res = ev[0] * freq_0 + ev[1] * (1- freq_0)
    else:
        nb_m = int(nb_meas/2)
        res = np.dot(np.random.binomial(nb_m, np.real(proba), (nb_r, len(proba))), ev) / nb_m
    #if nb_meas == 0: return np.zeros((nb_r))
    return np.nan_to_num(res)


def fom_1qstate(tgt, state, nb_m = 100, adapt = False, nb_r=1):
    """ Generate figure of merit relating a realized state to a tgt one"""
    if tgt.isket: tgt =  tgt * tgt.dag()
    if state.isket: state =  state * state.dag()
    if nb_m == np.inf:
        fid = (tgt * state).tr()
    else:
        op_meas = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        obs_tgt = [(op*tgt).tr() for op in op_meas]
        if adapt: coeffs = np.abs(obs_tgt)            
        else: coeffs = np.ones(len(op_meas)) 
        nb_m_each = (coeffs / np.sum(coeffs) * nb_m).astype(int)        
        obs_state = [meas_1_obs(op, state, n, nb_r = nb_r) for op, n in zip(op_meas, nb_m_each)]
        fid = 0.5 * (np.sum([t*s for t,s in zip(obs_tgt, obs_state)], 0) + 1.)

        
    assert np.all(np.imag(fid)<1e-5)
    return np.squeeze(np.real(fid))



# 1st test:
tgt_state = qt.rand_ket_haar()
tgt_dm = tgt_state * tgt_state.dag()
tgt_exp = [(op*tgt_dm).tr() for op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]]
test_state = qt.rand_dm(2)

n_repeat = 1000
n_m = 100
f_perf = fid_1qstate(tgt_state, test_state, nb_m = np.inf, adapt = False)
f_meas = fid_1qstate(tgt_state, test_state, nb_m = n_m, nb_r = 10)
f_meas_adapt = fid_1qstate(tgt_state, test_state, nb_m = n_m,adapt=True, nb_r = 10)


        
