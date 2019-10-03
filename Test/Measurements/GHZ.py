#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:42:34 2019

@author: fred
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

############ Init, utility
zero = qt.qubits.qubit_states(1,[0])
one = qt.qubits.qubit_states(1,[1])
I, X, Y, Z = qt.identity([2]), qt.sigmax(), qt.sigmay(), qt.sigmaz()
Rx, Ry, Rz = qt.rx, qt.ry, qt.rz






############# Measurement of a ghz state
ghz1 = 1/np.sqrt(2)*(qt.qubits.tensor(zero,zero,zero) + qt.qubits.tensor(one,one,one))
ghz2 = 1/np.sqrt(2)*(qt.qubits.tensor(zero,zero,zero) - qt.qubits.tensor(one,one,one))

op1 = [qt.tensor(I,Z,Z), qt.tensor(X,X,X), qt.tensor(X,Y,Y), qt.tensor(Y,X,Y), qt.tensor(Y,Y,X), qt.tensor(Z,I,Z),qt.tensor(Z,Z,I)]

exp_ghz1 = [o.matrix_element(ghz1, ghz1) for o in op1]
exp_ghz2 = [o.matrix_element(ghz2, ghz2) for o in op1]


def get(x):
    """ parametrized (6 parameters) circuitry which can create a c. GHZ 
    state i.e. |000> - |111>"""
    g1 = qt.tensor(qt.rx(x[0]), qt.rx(x[1]), qt.ry(x[2]))
    g2 = qt.cnot(3, 1,2)
    g3 = qt.cnot(3, 0,2)
    g4 = qt.tensor(qt.rx(x[3]), qt.rx(x[4]), qt.rx(x[5]))
    return g4 * g3 * g2 * g1 * qt.tensor(zero, zero, zero)

x_tgt = np.array([[1.,1.,2.,1.,1.,1.],[1., 3., 2., 3., 1., 3.], [3., 1., 2., 1., 3., 3.], [3., 3., 4., 1., 1., 3.],
                              [3., 1., 4., 3., 1., 1.], [1., 1., 4., 3., 3., 3.],[1., 3., 4., 1., 3., 1.],
                              [1., 1., 0., 3., 3., 3.],[1., 1., 2., 1., 1., 1.],[3., 3., 2., 3., 3., 1.],
                              [1., 3., 0., 1., 3., 1.],[3., 1., 0., 3., 1., 1.],[3., 3., 0., 1., 1., 3.]])

get(x_tgt[0]* np.pi/2)

def get2(x):
    """ parametrized (6 parameters) circuitry which can create a c. GHZ 
    state i.e. |000> - |111>"""
    g1 = qt.tensor(qt.rx(x[0]), qt.rx(x[1]), qt.rx(x[2]))
    g2 = qt.cnot(3, 1,2)
    g3 = qt.cnot(3, 0,2)
    g4 = qt.tensor(qt.rx(x[3]), qt.rx(x[4]), qt.ry(x[5]))
    return g4 * g3 * g2 * g1 * qt.tensor(zero, zero, zero)

def f(x):
    st_res = get2(x* np.pi/2)
    return 1 - np.square(np.abs((st_res.dag() * ghz1).tr()))
    
f(x_tgt[0])

from scipy import optimize as opt

res = opt.fmin_l_bfgs_b(f,approx_grad=True, x0=np.array([np.random.uniform(0,4) for _ in range(6)]), bounds=[(0,4) for _ in range(6)])

f(res[0])



#### Measurement of a ghz state
def Ramsey_exp(state, angle_Z, angle_X):
    nb_qubits = len(state.dims[1])
    transfo = get_circuit_meas(nb_qubits, angle_Z, angle_X)
    if state.isket:
        state_after =  qt.ket2dm(transfo * state)
    else:
        state_after = transfo * state * transfo.dag()
    parity = gen_parity_op2(nb_qubits)
    return (parity * state_after).tr()

def get_circuit_meas(nb_qubits, angle_Z, angle_X):
    layer_Z = qt.tensor([Rz(angle_Z) for _ in range(nb_qubits)])
    layer_X = qt.tensor([Rx(angle_X) for _ in range(nb_qubits)])
    return layer_X * layer_Z

def gen_parity_op(nb_qubits, proj_onto=zero):
    proj = proj_onto * proj_onto.dag()
    list_proj = [(-1)**n * gen_proj_onequbit(nb_qubits, n, proj) for n in range(nb_qubits)] 
    return reduce((lambda x, y: x + y), list_proj)


def gen_parity_op2(nb_qubits):
     return qt.tensor([Z for _ in range(nb_qubits)])


def gen_proj_onequbit(nb_qubits, which_qubit, proj):
    list_op = [I for _ in range(nb_qubits)]
    list_op[which_qubit] = proj.copy()
    return qt.tensor(list_op)

ghz_2q = get_ghz(2)
ghz_2q_mixed = 0.90 * qt.ket2dm(get_ghz(2)) + 0.1 * qt.ket2dm(qt.rand_ket_haar(4, [[2, 2], [1, 1]])) 
ghz_2q_phase = get_ghz(2, np.pi/2)
ghz_3q = get_ghz(3)
ghz_4q = get_ghz(4)
steps = np.linspace(0, 4, 5000)


### Plotting parity expectations
pop_2q = [Ramsey_exp(ghz_2q, p*np.pi, np.pi/2) for p in steps]
pop_2q_phase = [Ramsey_exp(ghz_2q_phase, p*np.pi, np.pi/2) for p in steps]
pop_2q_mixed = [Ramsey_exp(ghz_2q_mixed, p*np.pi, np.pi/2) for p in steps]
plt.plot(steps, pop_2q)
plt.plot(steps, pop_2q_phase)
plt.plot(steps, pop_2q_mixed)

### Plotting parity expectations
pop_3q = [Ramsey_exp(ghz_3q, p*np.pi, np.pi/2) for p in steps]
plt.plot(steps, pop_3q)


pop_4q = [Ramsey_exp(ghz_4q, p*np.pi, np.pi/2) for p in steps]
plt.plot(steps, pop_4q)



#### decomposition into local observables 
t2_0 = get_circuit_meas(2, 0, np.pi/2)
t2_05 = get_circuit_meas(2, np.pi/2, np.pi/2)
p2 = gen_parity_op2(2)
meas_2q = (t2_05.dag() * p2 * t2_05 - t2_0.dag() * p2 * t2_0).data.toarray()




t3_0 = get_circuit_meas(3, np.pi/2 - np.pi/3, np.pi/2)
t3_05 = get_circuit_meas(3, np.pi/2 + np.pi/3, np.pi/2)
p3 = gen_parity_op2(3)
meas_3q_1 = (t3_05.dag() * p3 * t3_05).data.toarray()
meas_3q_2 = (t3_0.dag() * p3 * t3_0).data.toarray()
meas_3q_total = np.imag(meas_3q_1 + meas_3q_2)
meas_3_inspect =np.array([np.diag(meas_3q_1), np.diag(meas_3q_2)])


t4_0 = get_circuit_meas(4, 0, np.pi/2)
t4_025 = get_circuit_meas(4, np.pi/4, np.pi/2)
p4 = gen_parity_op2(4)
meas_4q_1 = (t4_025.dag() * p4 * t4_025).data.toarray()
meas_4q_2 = (t4_0.dag() * p4 * t4_0).data.toarray()
meas_4q_total = meas_4q_1 + meas_4q_2
meas_4_inspect =np.array([np.diag(meas_4q_1), np.diag(meas_4q_2)])


def get_H(nb_qubits, phi):
    H_one = 1/2 * (np.exp(-1.0j*phi)* (zero * one.dag()) + np.exp(1.0j*phi)* (one * zero.dag()))
    list_H = [gen_proj_onequbit(nb_qubits, n, H_one) for n in range(nb_qubits)] 
    return reduce((lambda x, y: x + y), list_H) 

def get_U(nb_qubits, phi):
    H =  get_H(nb_qubits, phi)
    return H.expm()