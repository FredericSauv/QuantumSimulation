
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
    
def get_projector(ket):
    return  ket * ket.dag()

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

# =========================================================================== #
# Testing the different measurement schemes
# =========================================================================== #
U = unitary_group.rvs(4, 100)
u = [Qobj(U_el, dims = [[2,2],[2,2]]) for U_el in U]
real_fid = [fid_perfect(unit, CN) for unit in u]

### NIELSEN INITIAL
res_nielsen = produce_schemes_2q()(CN)
nielsen_fid = [fid_measurable(unit, *res_nielsen) for unit in u]
rank_obs = [rank(r) for r in res_nielsen[2]]
is_local_proj = [is_local(r) for r in res_nielsen[2]]
plt.plot(real_fid, nielsen_fid)

### NIELSEN + POLISHING
res_nielsen_prod = produce_schemes_2q("nielsen_prod")(CN)
nielsen_prod_fid = [fid_measurable(unit, *res_nielsen_prod) for unit in u]
rank_obs = [rank(r) for r in res_nielsen_prod[2]]
is_local_proj = [is_local(r) for r in res_nielsen_prod[2]]
plt.plot(real_fid, nielsen_prod_fid)


### FULL
res_all_smart = produce_schemes_2q("all_smart")(CN)
all_smart_fid = [fid_measurable(unit, *res_all_smart) for unit in u]
[rank(r) for r in res_all_smart[2]]
plt.plot(real_fid, all_smart_fid)



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




### Estimation error

s_Y_plus_1 = Y.eigenstates()[1][0]
state_basis = [st * st.dag() for st in [s_zero_1, s_one_1, s_plus_1, s_Y_plus_1]]
def decomp(state, basis):
    binv = np.linalg.inv(lqo2arr(basis))
    coeffs = binv.dot(vect(state))
    assert np.all(np.abs(np.imag(coeffs))<1e-6)
    return np.real(coeffs)
    
def recomp(coeffs, basis):
    return np.sum([c * b.full() for c, b in zip(coeffs, basis)], 0)
    

state = qutip.qutip.rand_dm(2)
O = qutip.qutip.rand_herm(2)
O_proj = [ket * ket.dag() for ket in O.eigenstates()[1]]
O_en = O.eigenenergies()

# can det create the state
proba_1 = [(state * p).tr() for p in O_proj]

# need probabilistic creation
proba_crea = decomp(state, state_basis)
#proba_nested = 
assert np.allclose(recomp(proba_crea, state_basis), state.full())




# =========================================================================== #
# Small study
# =========================================================================== #
# --------------------------------- STUDY 1 ----------------------------------#
# X = 8 X1 +2 X2, N measurement, X1 X2 follow laws with same variance
# How to distribute these N measurements
N = 999
Ex= 10
nb_repeat = 10000

ratio_1 = 1/2
ratio_2 = 2/3
ratio_3 = 3/4
ratio_4 = 4/5
ratio_5 = 5/6
ratio_6 = 6/7


def ens(n_s, n_r=nb_repeat):
    return np.average(np.random.normal(0,1, (int(n_s), n_r)), 0)

#case 1
Ex_1 = 8 * ens(int(N*ratio_1)) + 2 * ens(int(N*(1-ratio_1)))
Ex_2 = 8 * ens(int(N*ratio_2)) + 2 * ens(int(N*(1-ratio_2)))
Ex_3 = 8 * ens(int(N*ratio_3)) + 2 * ens(int(N*(1-ratio_3)))
Ex_4 = 8 * ens(int(N*ratio_4)) + 2 * ens(int(N*(1-ratio_4)))
Ex_5 = 8 * ens(int(N*ratio_5)) + 2 * ens(int(N*(1-ratio_5)))
Ex_6 = 8 * ens(int(N*ratio_6)) + 2 * ens(int(N*(1-ratio_6)))

print(np.sum(np.square(Ex_1)) / len(Ex_1))
print(np.sum(np.square(Ex_2)) / len(Ex_2))
print(np.sum(np.square(Ex_3)) / len(Ex_3))
print(np.sum(np.square(Ex_4)) / len(Ex_4))
print(np.sum(np.square(Ex_5)) / len(Ex_5))
print(np.sum(np.square(Ex_6)) / len(Ex_6))
      
#case 2
Ex_1 = 8 * ens(int(N*ratio_1)) - 2 * ens(int(N*(1-ratio_1)))
Ex_2 = 8 * ens(int(N*ratio_2)) - 2 * ens(int(N*(1-ratio_2)))
Ex_3 = 8 * ens(int(N*ratio_3)) - 2 * ens(int(N*(1-ratio_3)))
Ex_4 = 8 * ens(int(N*ratio_4)) - 2 * ens(int(N*(1-ratio_4)))
Ex_5 = 8 * ens(int(N*ratio_5)) - 2 * ens(int(N*(1-ratio_5)))
Ex_6 = 8 * ens(int(N*ratio_6)) - 2 * ens(int(N*(1-ratio_6)))

print(np.sum(np.square(Ex_1)) / len(Ex_1))
print(np.sum(np.square(Ex_2)) / len(Ex_2))
print(np.sum(np.square(Ex_3)) / len(Ex_3))
print(np.sum(np.square(Ex_4)) / len(Ex_4))
print(np.sum(np.square(Ex_5)) / len(Ex_5))
print(np.sum(np.square(Ex_6)) / len(Ex_6))

#case 3
Ex_1 = 8 * ens(int(N*1/4)) + 2 * ens(int(N*1/4)) - 3 * ens(int(N*1/4)) - 2 * ens(int(N*1/4))
Ex_2 = 8 * ens(int(N*8/15)) + 2 * ens(int(N*2/15)) - 3 * ens(int(N*1/5)) - 2 * ens(int(N*2/15))
print(np.sum(np.square(Ex_1)) / len(Ex_1))
print(np.sum(np.square(Ex_2)) / len(Ex_2))


# --------------------------------- STUDY 2 ----------------------------------#
# Is there a difference between deterministically preparing a state or not
# from a measurement perspective



# --------------------------------- STUDY 3 ----------------------------------#
# expected value with pure vs mixed states
nb_repeat = 1000
rho_pure = [qutip.ket2dm(qutip.rand_ket_haar(2)) for _ in range(nb_repeat)]
rho_mixed_2 = [0.5* qutip.ket2dm(qutip.rand_ket_haar(2)) + 0.5 * qutip.ket2dm(qutip.rand_ket_haar(2))  for _ in range(nb_repeat)]
rho_mixed_3 = [0.3333 * qutip.ket2dm(qutip.rand_ket_haar(2)) + 0.3333 * qutip.ket2dm(qutip.rand_ket_haar(2)) + 0.3333 * qutip.ket2dm(qutip.rand_ket_haar(2)) for _ in range(nb_repeat)]
rho_id = 1/2 * qutip.identity(2)
herm_list = [qutip.rand_herm(2) for _ in range(nb_repeat)]
res_exp = np.array([((rp * h).tr(), (rm2 * h).tr(), (rm3 * h).tr(), (rho_id * h).tr()) for h, rp, rm2, rm3 in zip(herm_list, rho_pure, rho_mixed_2, rho_mixed_3)])
np.average(res_exp, 0)
np.average(np.square(res_exp), 0)



def get_var_est(rho, obs, nb_m, nb_r):
    """ get exp observables for an initial state (rho) evolved under a unitary 
    (A) - it may evolve to incorporate more general processes, and where the 
    observable obs is made"""
    ref = (obs * rho).tr()
    assert np.imag(ref)<1e-5, "imaginary measurement"
    ref = np.real(ref)
        
    ev, EV = obs.eigenstates()
    proba = [(E.dag() * rho * E).tr()  for E in EV]
    assert np.all(np.imag(proba)<1e-6), "Imaginary proba"
    est = np.dot(np.random.binomial(nb_m, np.real(proba), (nb_r, len(proba))) / nb_m, ev)
    
    return np.average(np.square(est-ref))
    
res_pure = np.array([get_var_est(r, h, 100, 1000) for h, r in zip(herm_list, rho_pure)])
res_mixed_2 = np.array([get_var_est(r, h, 100, 1000) for h, r in zip(herm_list, rho_mixed_2)])
res_mixed_3 = np.array([get_var_est(r, h, 100, 1000) for h, r in zip(herm_list, rho_mixed_3)])
res_id = np.array([get_var_est(rho_id, h, 166, 1000) for h in herm_list])


print(np.average(res_pure), np.average(res_mixed_2), np.average(res_mixed_3), np.average(res_id))

### Small study: different ways to compute the same estimate
### M = (a A + b B)/(A) 
state = qutip.rand_dm(2)
proba = np.random.uniform(size = 4)
proba /= np.sum(proba)
op_to_estim = proba[0] * op_pauli_1[0] + proba[1] * op_pauli_1[1] + proba[2] * op_pauli_1[2] + proba[3] * op_pauli_1[3]

def estim_study(obs, state, coeffs = None, nb_m = 100):
    if coeffs is None:
        if(nb_m == np.inf):
            #perfect measurement
            res = (obs * state).tr()
            assert np.imag(res)<1e-5, "imaginary measurement"
            res = np.real(res)
            
        else:
            #linear estimate
            ev, EV = obs.eigenstates()
            proba = [(E.dag() * state * E).tr()  for E in EV]
            assert np.all(np.imag(proba)<1e-6), "Imaginary proba"
            res = np.dot(ev, np.random.binomial(nb_m, np.real(proba))) / nb_m
    else:
        assert np.all(np.array(coeffs)>=0)
        proba = coeffs / np.sum(coeffs)
        if(nb_m == np.inf):
            nb_meas_each = [np.inf] * len(coeffs)
            res = np.sum(np.array(proba) * np.array([estim_study(o, state, None, n) for o, n in zip(obs, nb_meas_each)]))
            
        else:
            
            draws = np.random.choice(len(proba), nb_m, p=proba)
            nb_meas_each = np.array([np.sum(draws == i) for i in range(len(proba))])
            res =np.sum( nb_meas_each * np.array([estim_study(o, state, None, n) for o, n in zip(obs, nb_meas_each)]) / nb_m)
        
    return res

estim_study(op_to_estim, state, None, np.inf)
estim_study(op_to_estim, state, None, 100)
estim_study(op_pauli_1, state, proba, np.inf)
estim_study(op_pauli_1, state, proba, 100)

state = qutip.rand_dm(2)
proba = np.random.uniform(size = 2)
proba /= np.sum(proba)
op_to_estim = proba[0] * op_pauli_1[0] + proba[1] * op_pauli_1[2]
nb_meas = 100
nb_repeat = 1000
target = estim_study(op_to_estim, state, None, np.inf)
res_samples = [(estim_study(op_to_estim, state, None, nb_meas), estim_study([op_pauli_1[0], op_pauli_1[2]], state, proba, 100)) for _ in range(nb_repeat)]
np.nanstd(np.abs(target - np.array(res_samples)), 0)






### 
def estim_esp(obs, state, nb)

s1 = qutip.rand_herm(2)
s2 = qutip.rand_herm(2)
ss = 0.5 * (s1 + s2)

