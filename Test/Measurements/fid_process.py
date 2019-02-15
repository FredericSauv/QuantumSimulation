
"""
Created on Mon Feb  4 11:06:47 2019

@author: fred
"""

import numpy as np
from qutip import sigmax, sigmay, sigmaz, identity, tensor, basis, ket2dm, Qobj, cnot
import qutip
import itertools as it
import matplotlib.pylab as plt
from scipy.stats import unitary_group


# =========================================================================== #
# Define the functions needed
# =========================================================================== #
def is_qobj(obj):
    return type(obj) == Qobj

def is_ket(obj):
    return is_qobj(obj) and obj.type == 'ket'

def is_oper(obj):
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
        data = custom_reorder_2q(to_decomp.full())
    
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
    
    return Qobj(np.sum([(d[0] * tensor(d[1], d[2])).full() for d in decompo]), dims = tensor(decompo[0][1], decompo[0][2]).dims)

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
    if(nbq > 1): dims = [[2, 2] for _ in range(nbq)]
    else: dims = [2,2]
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

def fid_measurable(A, coeffs, rho, sigma, nb_meas = np.inf):
    """ Fidelity based on experimental observables"""
    return 1/(A.shape[0]**3) * np.sum([c * get_exp_obs(A, r, s, nb_meas) for s, r, c in zip(sigma, rho, coeffs)])    
    
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


# one qubit
X, Y, Z, I = sigmax(), sigmay(), sigmaz(), identity(2)
op_pauli_1 = [X, Y, Z, I]
op_comput_1 = arr2lqo(np.sqrt(2) * np.eye(4))
s_zero_1, s_one_1 = basis(2, 0), basis(2,1)
s_plus_1, s_minus_1 = 1/np.sqrt(2) * (s_zero_1 + s_one_1), 1/np.sqrt(2) * (s_zero_1 - s_one_1)
dm_nielsen_1 = [norm_dm(dm) for dm in [I, I+X, I+Y, I+Z]]
dm_naive_1 = [norm_dm(dm) for dm in [I+Z, I-Z, I+X, I-X]]

#two qubits
CN = cnot()

def produce_schemes_2q(s_type="nielsen"):
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
            W = lqo2arr([V * b_el * V.dag() for b_el in b])
            S = W.dot(np.transpose(A))
            s = arr2lqo(S)
            #C = np.linalg.inv(S).dot(W)
            #M = A.dot(np.transpose(C))
            # aasert(M, np.eye(len(s)))
            return (np.ones(len(s)), p, s)
    elif(s_type == "nielsen_inv"):
        #Scheme in Nielsen 2015 
        b = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        B = lqo2arr(b) # Ortho basis matrix form
        s = [tensor(p[0], p[1]) for p in it.product(op_pauli_1, op_pauli_1)]
        S = lqo2arr(s)
        Sm = np.linalg.inv(S)
 
        def scheme(V):
            W = lqo2arr([V * b_el * V.dag() for b_el in b])
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

# =========================================================================== #
# Testing the different measurement schemes
# =========================================================================== #
U = unitary_group.rvs(4, 100)
u = [Qobj(U_el, dims = [[2,2],[2,2]]) for U_el in U]
real_fid = [fid_perfect(unit, CN) for unit in u]

# Nielsen Scheme
scheme_nielsen = produce_schemes_2q()

res_nielsen = scheme_nielsen(CN)
nielsen_fid = [fid_measurable(unit, *res_nielsen) for unit in u]
rank_obs = [rank(r) for r in res_nielsen[2]]
is_local_proj = [is_local(r) for r in res_nielsen[2]]

plt.plot(real_fid, nielsen_fid)



#Nielsen inverted 
scheme_ninv = produce_schemes_2q("nielsen_inv")
res_ninv = scheme_ninv(CN)
ninv_fid = [fid_measurable(unit, *res_ninv) for unit in u]


#twek to make rho states
lmin = np.array([r.eigenenergies().min()for r in res_ninv[1]])
ltrace = np.array([r.tr() for r in res_ninv[1]])
to_add = (lmin <0) * -lmin 
scale = 1/(ltrace + 4 * to_add)
T = np.eye(16)
T[-1] += to_add
T = T.dot(np.diag(scale))
Tinv =np.linalg.inv(T)

p_new = arr2lqo(lqo2arr(res_ninv[1]).dot(T))
lmin_new = np.array([r.eigenenergies().min()for r in p_new])
ltrace_new = np.array([r.tr() for r in p_new])
isherm_new = np.array([r.isherm for r in p_new])
coeffs_new = Tinv.dot(np.diag(res_ninv[0]))

c_new = np.array(list(np.diag(coeffs_new)) + list(coeffs_new[-1,:-1]))
p_new = p_new + [p_new[-1]] * (len(p_new)-1)
s_new = res_ninv[2] * 2
_ = s_new.pop()
scheme_ninv_custom = (c_new, p_new, s_new)


ninv_custom_fid = [fid_measurable(unit, *scheme_ninv_custom) for unit in u]
is_local_p = np.array([is_local(p) for p in p_new]) + np.array([is_product(p) for p in p_new]) 
is_separable_p = [is_product()]



plt.plot(real_fid, ninv_custom_fid)


# Testing
p = arr2lqo(P)
b = arr2lqo(B)
test = [sumqobj(p, A[:, c]) for c in range(A.shape[1])]
assert test == b, "coeffs (A) found don't do what they are supposed to do"


####### Testing fidelities



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









    
    
