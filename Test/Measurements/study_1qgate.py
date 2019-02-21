"""
Created on Mon Feb  4 11:06:47 2019

@author: fred
"""
import numpy as np
import qutip as qt
import utilities as ut
import matplotlib.pylab as plt
from scipy.stats import unitary_group


# =========================================================================== #
# Define the functions needed
# =========================================================================== #
def build_tilde(V):
    return np.kron(np.conjugate(V), V)


def produce_schemes_1q(s_type="strat1", b = None, p = None, s = None):
    """ Produce schemes i.e. function taking target as input and returning tuple 
    of (coeff, ratios, init_state, observable) capital letter (e.g. B are matrices while
    lower case letters refer to list of Qobj ) 
    + b/B initial orthogonal basis of unitaries
    + p/P input state basis
    + s/S operator basis
    """
    if(s_type == "strat1"):
        #Simpler easy pho, Hermitians
        if b is None:
            print("b was not supplied ... takes tensor product of Paulis")
            b = ut.op_pauli_1
        if p is None:
            print("p was not supplied ... takes tensor product of Paulis(+I)")
            p = ut.dm_nielsen_1 
        if s is None:
            print("s was not supplied ... takes tensor product of Paulis")
            s = ut.op_pauli_1  
        
        B = ut.lqo2arr(b)
        P = ut.lqo2arr(p)       
        A = np.linalg.inv(P).dot(B)
        S = ut.lqo2arr(s)
        Sinv = np.linalg.inv(S)
        
        def scheme(V, full=False):
            W = ut.lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            C = Sinv.dot(W)
            M = A.dot(np.transpose(C))
            assert np.allclose(np.imag(M), 0), "Complex entries in M "
            coeffs = np.real(ut.vect(M)) 
            rhos = p * len(p)
            sigmas = list(np.repeat(s, len(s)))
            ratios = (np.ones(len(coeffs)) * (coeffs != 0))/ np.sum(coeffs != 0) 
            if not(full):
                filt = coeffs !=0
                coeffs = coeffs[filt]
                ratios = ratios[filt]
                sigmas = [sigmas[i] for i, f in enumerate(filt) if f]
                rhos = [rhos[i] for i, f in enumerate(filt) if f]
            return (coeffs, rhos, sigmas, ratios)
        
    elif(s_type == "strat1_smart"):
        #Simpler easy pho, Hermitians
        if b is None:
            print("b was not supplied ... takes tensor product of Paulis")
            b = ut.op_pauli_1
        if p is None:
            print("p was not supplied ... takes tensor product of Paulis(+I)")
            p = ut.dm_nielsen_1 
        if s is None:
            print("s was not supplied ... takes tensor product of Paulis")
            s = ut.op_pauli_1  
        
        B = ut.lqo2arr(b)
        P = ut.lqo2arr(p)       
        A = np.linalg.inv(P).dot(B)
        S = ut.lqo2arr(s)
        Sinv = np.linalg.inv(S)
        
        def scheme(V, full=False):
            W = ut.lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            C = Sinv.dot(W)
            M = A.dot(np.transpose(C))
            assert np.allclose(np.imag(M), 0), "Complex entries in M "
            coeffs, rhos, sigmas, ratios = smart_allocation(np.real(M), p, s)
            return (coeffs, rhos, sigmas, ratios)
        
    elif(s_type == "diag"):
        b, p = ut.op_pauli_1, ut.dm_nielsen_1
        B, P = ut.lqo2arr(b), ut.lqo2arr(p) 
        A = np.linalg.inv(P).dot(B)
 
        def scheme(V):
            W = ut.lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            S = W.dot(np.transpose(A))
            s = ut.arr2lqo(S)
            coeffs, ratios, rhos, sigmas = np.ones(len(p)), np.ones(len(p))/len(p), p, s 
            return (coeffs, rhos, sigmas, ratios)    
    
    elif(s_type == "diag_smart"):
        b, p = ut.op_pauli_1, ut.dm_nielsen_1
        B, P = ut.lqo2arr(b), ut.lqo2arr(p) 
        A = np.linalg.inv(P).dot(B)
 
        def scheme(V):
            W = ut.lqo2arr([V * b_el.dag() * V.dag() for b_el in b])
            S = W.dot(np.transpose(A))
            s = ut.arr2lqo(S)
            scales = np.array([diff_eig(op) for op in s])
            coeffs, rhos, sigmas = np.ones(len(p)),  p, s 
            ratios = coeffs * scales * np.array([1., 1., 1., 1.])
            ratios /= np.sum(ratios)
            return (coeffs, rhos, sigmas, ratios)      
    
    elif(s_type == "svd"):
        if p is None:
            print("p was not supplied ... takes tensor product of Paulis(+I)")
            p = ut.dm_nielsen_1 
        if s is None:
            print("s was not supplied ... takes tensor product of Paulis")
            s = ut.op_pauli_1  
        def scheme(V):
            a,b,c = np.linalg.svd(np.kron(np.conjugate(V.full()), V.full()))
            p = ut.arr2lqo(a)
            s = ut.arr2lqo(c)
            coeffs, rhos, sigmas = np.ones(len(p)),  p, s 
            ratios = coeffs
            ratios /= np.sum(ratios)
            return (coeffs, rhos, sigmas, ratios)  
    
    
    elif(s_type == "strat1_smart"):
        raise NotImplementedError()
    
    elif(s_type == "diag_inv"):
        raise NotImplementedError()
    return scheme

def diff_eig(op):
    en = op.eigenenergies()
    return np.max(en) - np.min(en)

def smart_allocation(coeffs, rhos, sigmas):
    res_coeff, res_rho, res_sig = [], [], []
    for i, c in enumerate(np.transpose(coeffs)):
        filter_p = c>0
        filter_n = c<0
        if(np.sum(filter_p)>0):
            sum_p = np.sum(c[filter_p])
            res_coeff += [sum_p]
            res_sig += [sigmas[i]]
            res_rho += [ [c[filter_p] / sum_p, [rhos[i] for i, f in enumerate(filter_p) if f]] ]
        
        if(np.sum(filter_n)>0):
            sum_n = np.sum(c[filter_n])
            res_coeff += [sum_n]
            res_sig += [sigmas[i]]
            res_rho += [ [c[filter_n] / sum_n, [rhos[i] for i, f in enumerate(filter_n) if f]] ]

    res_coeff = np.array(res_coeff)
    res_ratio = np.abs(res_coeff) / np.sum(np.abs(res_coeff))
    
    return res_coeff, res_rho, res_sig, res_ratio

def measure_fid(A, coeffs, rho, sigma, ratios = None, nb_meas = np.inf, full = False):
    """ Fidelity based on experimental observables"""
    if(nb_meas == np.inf):
        obs = [get_exp_obs(A, r, s, nb_meas) for s, r in zip(sigma, rho)]
        
    else:
        ratios = (np.ones(len(coeffs)) * (coeffs != 0))/ np.sum(coeffs != 0) if ratios is None else ratios
        meas = dispatch(nb_meas, ratios)
        obs = [get_exp_obs(A, r, s, m) for s, r, m in zip(sigma, rho, meas)]
    res = 1/(A.shape[0]**3) * np.sum([c * o for c, o in zip(coeffs, obs)])
    if full:
        return res, obs
    else:
        return res

def get_exp_obs(A, rho, obs, nb_m = np.inf):
    """ get exp observables for an initial state (rho) evolved under a unitary 
    (A) 
    
    New: rhos can also be a list [list<probas>, list<states_dm>]
    """
    if(type(rho) == list):
        proba_rho = np.array(rho[0])
        assert np.allclose(np.sum(proba_rho), 1.), "probas don't sum to 1"
        rhos = rho[1]
    else:
        rhos = [rho]
        proba_rho = np.array([1.])
    rhos_final = [A.dag() * r * A for r in rhos]
    
    if(nb_m == np.inf):
        #perfect measurement
        res_each = [(obs * r_final).tr() for r_final in rhos_final]
        assert np.all(np.imag(res_each)<1e-5), "imaginary measurement"
        res = np.real(res_each).dot(proba_rho)
        
    else:
        #linear estimate
        meas_dispatch = dispatch(nb_m, proba_rho) 
        ev, EV = obs.eigenstates()
        proba_proj = [[(E.dag() * r_f * E).tr()  for E in EV] for r_f in rhos_final]
        assert np.all(np.imag(proba_proj)<1e-6), "Imaginary proba"
        res_each = [np.dot(ev, np.random.binomial(m, np.real(p)) / m) for m, p in zip(meas_dispatch, proba_proj)]
        res = np.dot(res_each, proba_rho)
    return res

def dispatch(nb_meas, probas, rule='rdm'):
    assert np.allclose(np.sum(probas), 1.)
    if rule == 'rdm':
        samples = np.random.choice(len(probas), nb_meas, p = probas)
        meas = [np.sum(samples == i) for i in range(len(probas))]
    else:
        meas = [int(p* nb_meas) for p in probas]
    return np.array(meas)
# =========================================================================== #
# Testing the different measurement schemes
# =========================================================================== #
Target = ut.HDM
nb_meas = 1000
U = unitary_group.rvs(2, 1000)
u = [qt.Qobj(U_el, dims = [[2],[2]]) for U_el in U]
real_fid = np.array([ut.fid_perfect(unit, Target) for unit in u])


# Standard scheme
sch1 = produce_schemes_1q('strat1')(Target)
sch1_fid_perfect = np.array([measure_fid(unit, *sch1) for unit in u])
sch1_fid_limited = np.array([measure_fid(unit, *sch1, nb_meas = nb_meas) for unit in u])

sch1_smart = produce_schemes_1q('strat1_smart')(Target)
sch1_smart_perfect = np.array([measure_fid(unit, *sch1_smart) for unit in u])
sch1_smart_limited = np.array([measure_fid(unit, *sch1_smart, nb_meas = nb_meas) for unit in u])

sch1_diag = produce_schemes_1q('diag')(Target)
sch1_diag_perfect = np.array([measure_fid(unit, *sch1_diag) for unit in u])
sch1_diag_limited = np.array([measure_fid(unit, *sch1_diag, nb_meas = nb_meas) for unit in u])

sch1_diag_smart = produce_schemes_1q('diag_smart')(Target)
sch1_diag_smart_perfect = np.array([measure_fid(unit, *sch1_diag_smart) for unit in u])
sch1_diag_smart_limited = np.array([measure_fid(unit, *sch1_diag_smart, nb_meas = nb_meas) for unit in u])

sch1_svd = produce_schemes_1q('svd')(Target)
sch1_svd_perfect = np.array([measure_fid(unit, *sch1_svd) for unit in u])

#plt.scatter(real_fid, sch1_fid_limited-real_fid)
plt.scatter(real_fid, sch1_diag_limited-real_fid)
plt.scatter(real_fid, sch1_diag_smart_limited-real_fid)

np.sqrt(np.average(np.square(sch1_diag_limited-real_fid)))
np.sqrt(np.average(np.square(sch1_diag_smart_limited-real_fid)))

np.std(sch1_fid_limited-real_fid)
np.std(sch1_smart_limited-real_fid)


sch = sch1_diag_smart
unperfect = [measure_fid(unit, *sch, nb_meas = nb_meas, full = True) for unit in u]
perfect = [measure_fid(unit, *sch, nb_meas = np.inf, full = True) for unit in u]
diff_element = [np.abs(np.array(un[1]) - np.array(perf[1])) for un, perf in zip(unperfect, perfect)]
np.average(np.square(diff_element), 0)



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
    return np.sum([c * b.full() for c, b in zip(coeffs, basis)], real_fid0)
    

state = qutip.qutip.rand_dm(2)
O = qutip.qutip.rand_herm(2)
O_proj = [ket * ket.dag() for ket in O.eigenstates()[1]]
O_en = O.eigenenergies()

# can det create the state
proba_1 = [(state * p).tr() for p in O_proj]

# need probabilistic creation
proba_crea = decomp(state, state_basis)
proba_nested = 
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
herm_list = [qutip.rand_herm(2) for _ in range(nb_repeat)]
res_exp = np.array([((rp * h).tr(), (rm2 * h).tr(), (rm3 * h).tr()) for h, rp, rm2, rm3 in zip(herm_list, rho_pure, rho_mixed_2, rho_mixed_3)])
np.average(res_exp, 0)
np.average(np.square(res_exp), 0)




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

