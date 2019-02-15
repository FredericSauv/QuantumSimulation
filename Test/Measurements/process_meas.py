#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:06:47 2019

@author: fred
"""
import numpy as np
from qutip import sigmax, sigmay, sigmaz, identity, tensor, basis, ket2dm, Qobj, cnot
import itertools as it
import matplotlib.pylab as plt
from scipy.stats import unitary_group
U = unitary_group.rvs(4, 100)
u = [Qobj(U_el, dims = [[2,2],[2,2]]) for U_el in U]
X, Y, Z, I = sigmax(), sigmay(), sigmaz(), identity(2)

### Define stuff
def ip(A, B):
    return (A.dag() * B).tr()

def proj(listop, basis):
    return [[ip(b, op) for b in basis] for op in listop]

def norm_dm(A):
    if(type(A) == Qobj):
        B = A /A.tr()
    else:
        B = A / np.trace(A)
    return B

def flatten(arr):
    if(type(arr) == Qobj):
        arr = arr.full()
    elif(type(arr) == list):
        arr = np.array(arr)
    return arr.reshape((np.size(arr)))

def lqo2arr(lqo):
    arr = [flatten(qo) for qo in lqo]
    return np.transpose(arr)

def arr2lqo(arr):
    d = int(np.sqrt(np.shape(arr)[0]))
    nbq = int(np.log2(d))
    if(nbq > 1):
        dims = [[2, 2] for _ in range(nbq)]
    else:
        dims = [2,2]
    return [Qobj(np.reshape(a, (d, d)), dims = dims) for a in np.transpose(arr)]

def sumqobj(lqobj, coeffs = None):
    if coeffs is None: coeffs = np.ones(len(lqobj))
    dims = lqobj[0].dims
    data = np.sum([c * qo.full() for qo, c in zip(lqobj, coeffs)], 0)
    return Qobj(data, dims)

def is_separable(obs):
    if(type(obs) == Qobj):
        assert obs.isherm, "Observables should be Hermitian"
        _, ES = obs.eigenstates()
    else:
        assert np.allclose(obs, np.transpose(np.conj(obs))), "Observables should be Hermitian"
        _, ES = np.linalg.eigh(obs)
        ES = [Qobj(ES[:, i]) for i in range(S.shape[1])]
    test_entangl = np.all([not(is_entangled(state)) for state in ES])
    return test_entangl
    
def is_entangled(state):
    state = state * state.dag()
    pt = pt_A(state)
    en = pt.eigenenergies()
    test = not(np.any(np.abs(1 - en) < 1e-6))
    return  test


def pt_A(A,  dims = None):
    """ Partial trace on the first subsystem of a bipartite syste,m """
    if type(A) == Qobj:
        dims = A.dims
        A = A.full()
    else:  
         assert dims is not None
    A = A.reshape(flatten(dims))
    pt = np.einsum('ijik->jk', A)
    new_dim = [dims[0][0], dims[1][0]]
    return Qobj(pt, dims = new_dim)

    

### 1Qubit operators, basis
b_1q_pauli = [X, Y, Z, I]
b_1q_comput = arr2lqo(np.sqrt(2) * np.eye(4))
zero1, one1 = basis(2, 0), basis(2,1)
plus1, minus1 = 1/np.sqrt(2) * (zero1 + one1), 1/np.sqrt(2) * (zero1 - one1)
p_1q_naive_unnormed = [I+Z, I-Z, I+X, I-X]
p_1q_nielsen_unnormed = [I, I+X, I+Y, I+Z]
p_1q_naive = [norm_dm(p) for p in p_1q_naive_unnormed]
p_1q_nielsen = [norm_dm(p) for p in p_1q_nielsen_unnormed]  


### 2 Qubits
#orthonorm basis
V = cnot() 
#B = np.eye(16) * 4
b_2q_pauli = [tensor(p[0], p[1]) for p in it.product(b_1q_pauli, b_1q_pauli)]
b_2q_comput = arr2lqo(np.eye(16)*2)

b = b_2q_pauli 
B = lqo2arr(b)
W = lqo2arr([V * b_el * V.dag() for b_el in b])

#input states
p_2q_nielsen = [tensor(p[0], p[1]) for p in it.product(p_1q_nielsen, p_1q_nielsen)]
p_2q_naive = [tensor(p[0], p[1]) for p in it.product(p_1q_naive, p_1q_naive)]
P2q_nielsen = lqo2arr(p_2q_nielsen)
P2q_naive = lqo2arr(p_2q_naive)
assert (arr2lqo(lqo2arr(p_2q_naive)) == p_2q_naive)

# find observables
P = P2q_nielsen
A = np.linalg.inv(P).dot(B)
S = W.dot(np.transpose(A))
s = arr2lqo(S)
C = np.linalg.inv(S).dot(W)
M = A.dot(np.transpose(C))

# Testing
p = arr2lqo(P)
b = arr2lqo(B)
test = [sumqobj(p, A[:, c]) for c in range(A.shape[1])]
assert test == b, "coeffs (A) found don't do what they are supposed to do"


####### Testing fidelities
def fid_perfect(A, T= V):
    if(type(A) == Qobj):
        f = np.square(np.abs((A.dag() * T).tr())) / A.shape[0]
    else:        
        d = A.shape[0]
        f = np.square(np.abs(1/d * np.trace(np.transpose(np.conj(A)).dot(T))))
    return f

def fid_measurable(A, sigma=s, rho=p, coeffs= np.zeros(16)):
    F = 1/(4**3) * np.sum([(s * A.dag() * r * A).tr() for s, r in zip(sigma, rho)])
    assert np.abs(np.imag(F)) <1e-5
    return np.real(F)


fidelities = np.array([(fid_perfect(u_el), fid_measurable(u_el)) for u_el in u])
plt.plot(fidelities[:, 0], fidelities[:, 1])

test = s[0]
test_ES = test.eigenstates()[1][0]



b_pauli = [1/np.sqrt(2)*X, 1/np.sqrt(2)*Y, 1/np.sqrt(2)*Z, 1/np.sqrt(2)*I]
b1 = [tensor(A,B) for A, B in it.product(b_pauli, b_pauli)] #bais of unitary operator



b2 = [ket2dm(tensor(A, B)) for A, B in it.product(st, st)] #basis of input states

cnot = tensor(ket2dm(zero1), I) + tensor(ket2dm(one1), X)
U = cnot

A = np.array(proj(b1, b2))
meas = [np.sum([A[k,l] * U * b1[k] * U.dag() for k in range(A.shape[0])])  for l in range(A.shape[1])]
isherm = np.all([m.isherm for m in meas])




dm = qutip.rand_dm_hs(4, dims=[[2, 2]] * 2).full()
# reshape to do the partial trace easily using np.einsum
reshaped_dm = dm.reshape([2, 2, 2, 2])
# compute the partial trace
reduced_dm = np.einsum('ijik->jk', reshaped_dm)








def vect(mat):
    return np.reshape(mat, np.size(mat), order = 'F')

def devect(vect):
    dsquare = len(vect)
    d = int(np.sqrt(dsquare))
    return np.reshape(vect, (d,d), 'F')
    

def schm_decompo(to_decomp):
    """ Schmidt decompo of states/operators - split it in two even partites"""
    if(type(to_decomp) == Qobj):
        if(to_decomp.type == 'ket'):
            init_type = 'ket'
            data = to_decomp.full()
            dsquare = np.size(data)
            d = int(np.sqrt(dsquare))
            data = np.reshape(data, [d, d])
        
        elif(to_decomp.type == 'oper'):
            init_type = 'oper'
            data = to_decomp.full()
            data = custom_reorder(data)
    else:
        data=to_decomp
    U, D, Vdag = np.linalg.svd(data)
    V = np.transpose(Vdag)
    res = []
    for i, d in enumerate(D):
        if(d>1e-6):
            if(init_type == 'ket'):
                u = Qobj(np.array(U[:, i])[:, np.newaxis])
                v = Qobj(np.array(V[:, i])[:, np.newaxis])
                res.append((d, u, v))
            elif(init_type == 'oper'):
                u = Qobj(devect(U[:, i]))
                v = Qobj(devect(V[:, i]))
                res.append((d, u, v))
    return res

def schm_recompo(decompo):
    """ take the result of schm decompo and reconstruct the initial object"""
    res = None
    for d in decompo:
        if res is None:
            res = d[0] * tensor(d[1], d[2])
        else:
            res += d[0] * tensor(d[1], d[2])
    return res


def custom_reorder(mat):
    mask = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    res = np.reshape(mat, np.size(mat))
    res = np.reshape(res[mask], np.shape(mat))
    return res
    
def close_Qobj(o1, o2):
    return np.allclose(o1.full(), o2.full())
    
    
