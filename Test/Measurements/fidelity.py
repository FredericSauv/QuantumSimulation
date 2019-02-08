#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:25:18 2019

@author: fred
"""

import matplotlib.pylab as plt
from scipy.stats import unitary_group
U = unitary_group.rvs(4, 100000)

import numpy as np

comput_basis = [np.array([1.,0.,0.,0.]), np.array([0.,1.,0.,0.]), np.array([0.,0.,1.,0.]), np.array([0.,0.,0.,1.])]
state_basis = [np.array([1.,0.,0.,0.]), np.array([0.,1.,0.,0.]), np.array([0., 0., 1/np.sqrt(2), 1/np.sqrt(2)]),np.array([0.,0.,1.,0.]), np.array([0.,0.,0.,1.])]
state_decompo = np.array([1,1,2,-1,-1])
cnot = np.array([[1.-0.j, 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]])

def fid1(A, T= cnot):
    d = A.shape[0]
    f = np.square(np.abs(1/d * np.trace(np.transpose(np.conj(A)).dot(T))))
    return f

def fid_cheap(A, T=cnot, basis=comput_basis):
    ev_A = [A.dot(b) for b in basis]
    ev_T = [T.dot(b) for b in basis]
    f = np.average([np.square(np.abs(np.dot(np.conj(t), a))) for a, t in zip(ev_A, ev_T)])
    return f


fs = np.array([(fid1(u), fid_cheap(u)) for u in U]) 
plt.scatter(fs[:,0], fs[:,1])