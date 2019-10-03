#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:20:28 2019

@author: fred
"""
import numpy as np
import qutip as qt
from qutip import sigmax, sigmay, sigmaz, identity, tensor, basis, ket2dm, Qobj, cnot, hadamard_transform
import qutip
import itertools as it

## Shortcuts of some main qutip objects
zero = qt.qubits.qubit_states(1,[0])
one = qt.qubits.qubit_states(1,[1])
I, X, Y, Z = qt.identity([2]), qt.sigmax(), qt.sigmay(), qt.sigmaz()
Rx, Ry, Rz = qt.rx, qt.ry, qt.rz
op_pauli_1 = [X, Y, Z, I]
s_zero_1, s_one_1 = basis(2, 0), basis(2,1)
s_plus_1, s_minus_1 = 1/np.sqrt(2) * (s_zero_1 + s_one_1), 1/np.sqrt(2) * (s_zero_1 - s_one_1)


#two qubits
CN = cnot()
HDM = hadamard_transform()

# =========================================================================== #
# Define the functions needed
# =========================================================================== #
# GHZ related functions
def get_ghz(nb_qubits, angle=0):
    """ Generate a GHZ state of the form |00..00> + exp(i * angle) |11..11> """
    a = tensor([zero for _ in range(nb_qubits)])
    b = tensor([one for _ in range(nb_qubits)])
    return 1/np.sqrt(2) * (a+ np.exp(1.0j*angle) * b)

def get_ghz_offdiag(nb_qubits):
    """ generate the Hermitian operator |0...0><1...1| + |1...1><0...0|"""
    one_ghz, zero_ghz = tensor([one for _ in range(nb_qubits)]), tensor([zero for _ in 
                              range(nb_qubits)]) 
    return one_ghz * zero_ghz.dag() + zero_ghz * one_ghz.dag()


def is_qobj(obj):
    """ check if qutip object"""
    return type(obj) == Qobj

def is_ket(obj):
    """ check if it is a ket qutip object  """
    return is_qobj(obj) and obj.type == 'ket'

def is_oper(obj):
    """ check if it is a oper qutip object  """
    return is_qobj(obj) and obj.type == 'oper'

def norm_dm(A):
    """ Normalize a density matrix"""
    if is_qobj(A): return A /A.tr()
    else: return A / np.trace(A)

def vect(mat):
    """ Vectorize a matrix (convention F: stack columns)"""
    if is_qobj(mat): mat = mat.full() 
    return np.reshape(mat, np.size(mat), order = 'F')

def devect(vect):
    """ de-vectorized a vectorized object - assume the initial matrix was square"""
    d = int(np.sqrt(len(vect)))
    return np.reshape(vect, (d,d), 'F')
    
def schm_decompo(to_decomp):
    """ Schmidt decompo of states/operators - split it in two even partites"""
    if is_ket(to_decomp):
        init_type = 'ket'
        data = to_decomp.full()
        d = int(np.sqrt(np.size(data)))
        data = np.reshape(data, [d, d])
        
    elif is_oper(to_decomp):
        init_type = 'oper'
        if(to_decomp.full().shape[0] == 4):
            data = custom_reorder_2q(to_decomp.full())
        else:
            data = to_decomp.full()
    else:
        data=to_decomp
        if(np.ndim(data) == 1): init_type = 'ket'
        elif (np.ndim(data) == 1) and (data.shape[0] == data.shape[1]): init_type = 'oper'
        else: raise SystemError('data type/shape not understood: {} / {}'.format)
        
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
    
    return Qobj(np.sum([(d[0] * tensor(d[1], d[2])).full() for d in decompo], 0), dims = tensor(decompo[0][1], decompo[0][2]).dims)

def custom_reorder_2q(mat):
    """ reorganize a 4x4 array such that it can be used for schm decompo"""
    mask = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    return np.reshape(np.reshape(mat, np.size(mat))[mask], np.shape(mat))
    
def is_close(o1, o2):
    """ compare only data"""
    if is_qobj(o1): o1 = o1.full()
    if is_qobj(o2): o2 = o2.full()
    assert np.all(np.shape(o1) == np.shape(o2)), "o1 and o2 don't have the same shape"
    return np.allclose(o1, o2)

def flatten(arr):
    if(type(arr) == Qobj):
        arr = arr.full()
    elif(type(arr) == list):
        arr = np.array(arr)
    return arr.reshape((np.size(arr)))

def lqo2arr(lqo):
    """ transform a list of qobj into a matrix """
    return np.transpose([vect(qo) for qo in lqo])

def arr2lqo(arr):
    """ transform a matrix in a list of qobj """
    d = int(np.sqrt(np.shape(arr)[0]))
    nbq = int(np.log2(d))
    dims = [[2 for _ in range(nbq)], [2 for _ in range(nbq)]]
    return [Qobj(devect(a), dims = dims) for a in np.transpose(arr)]

def sumqobj(lqobj, coeffs = None):
    """ weighted sum of a list of qobj"""
    if coeffs is None: coeffs = np.ones(len(lqobj))
    dims = lqobj[0].dims
    data = np.sum([c * qo.full() for qo, c in zip(lqobj, coeffs)], 0)
    return Qobj(data, dims)

def rank(obj):
    """ rank (how many non zero) of the schm decompo"""
    return len(schm_decompo(obj))

def is_local(obj):
    """ can an operator be decomposed in local projections"""
    _, EV = obj.eigenstates()
    return np.all([is_product(E) for E in EV])
    

def is_product(obj):
    """ can it nbe written as bipartite product works with state/ operator"""
    return rank(obj) == 1

def fid_perfect(A, V):
    """ Real fidelity: 1/d |Tr[V^{\dagger} A]|^2 """
    if(type(A) == Qobj): return np.square(np.abs((A.dag() * V).tr())) / np.square(A.shape[0])
    else: return np.square(np.abs(1/A.shape[0] * np.trace(np.transpose(np.conj(A)).dot(V))))


def get_exp_obs(A, rho, obs, nb_m = np.inf):
    """ get exp observables for an initial state (rho) evolved under a unitary 
    (A) - it may evolve to incorporate more general processes, and where the 
    observable obs is made"""
    rho_final = A.dag() * rho * A
    if(nb_m == np.inf):
        #perfect measurement
        res = (obs * rho_final).tr()
        assert np.imag(res)<1e-5, "imaginary measurement"
        res = np.real(res)
        
    else:
        #linear estimate
        ev, EV = obs.eigenstates()
        proba = [(E.dag() * rho_final * E).tr()  for E in EV]
        assert np.all(np.imag(proba)<1e-6), "Imaginary proba"
        res = np.dot(ev, np.random.binomial(nb_m, np.real(proba))) / nb_m
    return res

def get_h_part(op):
    return (op + op.dag())/2

def get_antih_part(op):
    return (op - op.dag())/(2.j)

def qo_notnull(qo):
    return not(np.allclose(np.abs(qo.full()), 0.))

def after_decomp(decomp):
    """ custom to d after doing a schm decomp of an hermitian operator"""
    res = []
    for c, opA, opB in decomp:
        opA_h, opA_ah = get_h_part(opA), get_antih_part(opA)
        opB_h, opB_ah = get_h_part(opB), get_antih_part(opB)
        if(qo_notnull(opA_h) and qo_notnull(opB_h)):
            res.append([c, opA_h, opB_h])
        if(qo_notnull(opA_ah) and qo_notnull(opB_ah)):
            res.append([-c, opA_ah, opB_ah])
    return res


op_comput_1 = arr2lqo(np.sqrt(2) * np.eye(4))
dm_nielsen_1 = [norm_dm(dm) for dm in [I, I+X, I+Y, I+Z]]
dm_naive_1 = [norm_dm(dm) for dm in [I+Z, I-Z, I+X, I-X]]



def produce_schemes_2q(s_type="nielsen", b = None, p = None, s = None):
    """ Produce schemes i.e. function taking target as input and returning tuple 
    of (coeff, init_state, observable) capital letter (e.g. B are matrices while
    lower case letters refer to list of Qobj ) 
    b/B initial orthogonal basis of unitaries
    p/P input state basis
    s/S operator basis
    """
    if(s_type == "nielsen"):
        #Scheme in Nielsen 2015 
        b = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        B = lqo2arr(b) # Ortho basis matrix form
        p = [tensor(p[0], p[1]) for p in it.product(dm_nielsen_1, dm_nielsen_1)]
        P = lqo2arr(p)
        A = np.linalg.inv(P).dot(B)
 
        def scheme(V):
            W = lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            S = W.dot(np.transpose(A))
            s = arr2lqo(S)
            #C = np.linalg.inv(S).dot(W)
            #M = A.dot(np.transpose(C))
            # aasert(M, np.eye(len(s)))
            return (np.ones(len(s)), p, s)
        
    elif(s_type == "nielsen_prod"):
        # Scheme in Nielsen 2015 + decompose any obs in sum of hermitian product obs
        b = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        B = lqo2arr(b) # Ortho basis matrix form
        p = [tensor(p[0], p[1]) for p in it.product(dm_nielsen_1, dm_nielsen_1)]
        P = lqo2arr(p)
        A = np.linalg.inv(P).dot(B)
 
        def scheme(V):
            W = lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            S = W.dot(np.transpose(A))
            s = arr2lqo(S)
            coeffs, phos, sigmas = [], [], [] 
            for op, st in zip(s, p):
                if rank(op) == 1:
                    coeffs.append(1.)
                    phos.append(st)
                    sigmas.append(op)
                else:
                    dec = after_decomp(schm_decompo(op))
                    assert is_close(schm_recompo(dec), op), "Pb in the decomp"
                    for dec_coeffs, dec_partA, dec_partB in dec:
                        coeffs.append(dec_coeffs)
                        phos.append(st)
                        sigmas.append(tensor(dec_partA, dec_partB))
            #C = np.linalg.inv(S).dot(W)
            #M = A.dot(np.transpose(C))
            # aasert(M, np.eye(len(s)))
            return (coeffs, phos, sigmas)    
    
    elif(s_type == "all_smart"):
        # For a given s, p, b get the coeffs associated 
        if b is None:
            print("b was not supplied ... takes tensor product of Paulis")
            b = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]  

        if p is None:
            print("p was not supplied ... takes tensor product of Paulis(+I)")
            p = [tensor(p[0], p[1]) for p in it.product(dm_nielsen_1, dm_nielsen_1)]  
        if s is None:
            print("s was not supplied ... takes tensor product of Paulis")
            s = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]  
        B = lqo2arr(b)
        P = lqo2arr(p)       
        A = np.linalg.inv(P).dot(B)
        S = lqo2arr(s)
        Sinv = np.linalg.inv(S)
        def scheme(V):
            W = lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            C = Sinv.dot(W)
            M = A.dot(np.transpose(C))
            assert np.allclose(np.imag(M), 0), "Complex entries in M "
            coeffs = np.real(vect(M)) 
            phos = p * len(p)
            sigmas = list(np.repeat(s, len(s))) 
            return (coeffs, phos, sigmas)    
    
    elif(s_type == "nielsen_inv"):
        #Scheme in Nielsen 2015 
        b = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        B = lqo2arr(b) # Ortho basis matrix form
        s = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        S = lqo2arr(s)
        Sm = np.linalg.inv(S)
 
        def scheme(V):
            W = lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            C = Sm.dot(W)
            P = B.dot(np.transpose(C))
            p = arr2lqo(P)
            #C = np.linalg.inv(S).dot(W)
            #M = A.dot(np.transpose(C))
            # aasert(M, np.eye(len(s)))
            return (np.ones(len(s)), p, s)
    
    return scheme
#def recast_scheme(coeffs, rho, obs):
#    en = [o.eigenenergies() for o in obs]
#    en_min

# =========================================================================== #
# Some testing of the functions
# =========================================================================== #
# test methods: vectorization / experimental observables/ schmidt decompo
assert np.allclose(lqo2arr(arr2lqo(np.eye(16))), np.eye(16)), "Pb with the (de)casting of list of qobj to matrices"
test_dim = 4
test_obs, test_rho, test_A, test_ket = qutip.rand_herm(test_dim), qutip.rand_dm(test_dim), qutip.rand_unitary_haar(test_dim), qutip.rand_ket(test_dim)
test_mat = np.random.normal(size=[test_dim, test_dim])
assert np.abs(get_exp_obs(test_A, test_rho, test_obs) - get_exp_obs(test_A, test_rho, test_obs, 1000000)) <1e-3, "for that many obs estimated val and th val should be the same"
assert np.abs(get_exp_obs(test_A, test_rho, test_obs) - get_exp_obs(test_A, test_rho, test_obs, 100)) > 1e-3
assert np.allclose(devect(vect(test_mat)), test_mat), "Pb with (de)vectorization"
assert is_close(schm_recompo(schm_decompo(test_ket)), test_ket), "Schm decompo/recompo don't work"
