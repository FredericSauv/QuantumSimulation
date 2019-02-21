#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:48:58 2019

@author: fred
"""

import qutip as qt
import numpy as np
import matplotlib.pylab as plt
import utilities as ut

tgt_state = qt.rand_ket_haar()
tgt_dm = tgt_state * tgt_state.dag()
tgt_exp = [(op*tgt_dm).tr() for op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]]

def fid_1qstate(tgt, state, nb_m = 100, adapt = False, nb_r=1):
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

def meas_1_obs(obs, state, nb_meas, closed = True, nb_r = 1):
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



# --------------------------------- STUDY 1 ----------------------------------#
# Is there a difference between deterministically preparing a state or not
# from a measurement perspective
test_states = [qt.rand_dm(2) for i in range(500)]
tgt_state = qt.rand_ket_haar()
tgt_state = qt.basis(2,0)

n_repeat = 1000
n_m = 1000
f_perf = [fid_1qstate(tgt_state, st, nb_m = np.inf, adapt = False) for st in test_states]
f_meas = [fid_1qstate(tgt_state, st, nb_m = n_m, nb_r = n_repeat) for st in test_states]
f_meas_adapt = [fid_1qstate(tgt_state, st, nb_m = n_m,adapt=True, nb_r = n_repeat) for st in test_states]

stats = np.array([(perf, np.sqrt(np.average(np.square(m1 -perf))), np.sqrt(np.average(np.square(m2 -perf)))) for  perf, m1, m2 in zip(f_perf, f_meas, f_meas_adapt)])

plt.scatter(stats[:,0], stats[:,1])
plt.scatter(stats[:,0], stats[:,2])


# --------------------------------- STUDY 2 ----------------------------------#
# can we decomp any state as cI/2 
# Is there an advantage in doing so
def decomp_1q(state_dm):
    I = 0.5 * qt.identity(2)
    c = (state_dm* state_dm).tr()
    r = np.sqrt(2 * c -1)
    if np.allclose(r, 0):
        return [(1., I)]
    st_rest = state_dm - (1-r) * qt.identity(2)/2
    assert np.sum(np.abs(np.linalg.eigvals(st_rest.full()))>1e-5) == 1
    return [(1-r, I), (r, st_rest/r)]


def meas(obs, unit, s_init, nb_m = 100, decomp = False, nb_r=1):
    if nb_m == np.inf: 
        s_final = unit * s_init * unit.dag()
        exp = (obs * s_final).tr()
    else:
        obs_ev, obsEV = obs.eigenstates()
        if decomp: 
            dec = decomp_1q(s_init)
            if len(dec) == 1:
                assert (dec[0] == 0.5) and (2 * dec[1] == qt.identity(2)) 
                exp = 0.5 + np.sum(obs_ev)
            else:
                exp = dec[0][0] * 0.5 * np.sum(obs_ev)
                s_final = unit * dec[1][1] * unit.dag()
                exp += dec[1][0] * meas_1_obs(obs, s_final, nb_meas = nb_m, nb_r=nb_r)
        else: 
            s_final = unit * s_init * unit.dag()
            exp = meas_1_obs(obs, s_final, nb_meas = nb_m, nb_r=nb_r)
    assert np.all(np.imag(exp)<1e-5)
    return np.squeeze(np.real(exp))


Unit = ut.HDM
test_st = qt.rand_dm(2)
decomp_1q(test_st)
obs = qt.rand_herm(2)
perf = meas(obs, Unit, test_st, np.inf)
m = meas(obs, Unit, test_st, 1000, nb_r=1000)
m_dec = meas(obs, Unit, test_st, 1000, decomp = True, nb_r=1000)
np.sqrt(np.average(np.square(m-perf)))
np.sqrt(np.average(np.square(m_dec-perf)))

dm_list = [qt.rand_dm_hs(2) for r in range(50)]
op_list = [qt.rand_herm(2)for r in range(50)]

res_perfect = np.array([[meas(obs, Unit, test_st, np.inf) for op in op_list] for dm in dm_list])
res_exp = np.array([[meas(obs, Unit, test_st, 1000, nb_r=1000) for op in op_list] for dm in dm_list])
res_exp_decompo = np.array([[meas(obs, Unit, test_st, 1000, decomp = True, nb_r=1000) for op in op_list] for dm in dm_list])


std_exp = np.sqrt(np.average(np.square(res_exp - res_perfect[:, :, np.newaxis]), -1))
std_exp_decompo = np.sqrt(np.average(np.square(res_exp_decompo  - res_perfect[:, :, np.newaxis]), -1))

index_dm = range(std_exp.shape[0])
for i in range(std_exp.shape[1]):
    plt.scatter(index_dm, std_exp[:, i], c = 'blue')
    plt.scatter(index_dm, std_exp_decompo[:, i], c='orange')



