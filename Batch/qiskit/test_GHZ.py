#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:24:40 2020

@author: form Kiran
"""

from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer, transpile
import numpy as np

simulator = Aer.get_backend('qasm_simulator')
MEAS_DEFAULT = ['xxx', '1zz', 'z1z', 'zz1', 'yyx', 'xyy', 'yxy']
MEAS_WEIGHTS = np.array([1., 1., 1., 1., -1., -1., -1.])/4.0
NB_SHOTS_DEFAULT = 128
pi = np.pi

def create_circ(params = np.zeros(6)):
    c = QuantumCircuit(3)
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    c.cnot(0,2) 
    c.cnot(1,2) 
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    return c
   

def append_measurements(circ, measurements):
    """ Assumes creator returns an instance of the relevant circuit"""
    num_classical = len(measurements.replace('1',''))
    if num_classical > 0:
        cr = ClassicalRegister(num_classical, 'c')
        circ.add_register(cr)
    ct_m = 0
    ct_q = 0
    for basis in measurements:
        if basis == 'z':
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'x':
            circ.u3(pi/2, 0, 0, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'y':
            circ.u3(pi/2, -pi/2, -pi/2, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == '1':
            pass
        ct_q+=1
    return circ


def gen_meas_circuits(creator, meas_settings, params):
    """ Return a list of measurable circuit with same parameters but with 
    different measurement settings"""
    if type(meas_settings) is not list:
        meas_settings = [meas_settings]
    c_list = [append_measurements(creator(params), m)for m in meas_settings]
    return c_list

def Mean(results):
    'computes the mean of a list of reqults [0,1]'
    keys = list(results.keys())
    running_mean = 0
    for k in keys:
        val = results[k]
        neg_valus = k.count('0')
        if neg_valus%2 == 0:
            running_mean+=val
       
    return running_mean / sum(list(results.values()))

def F(experimental_params, flags = False, shots = NB_SHOTS_DEFAULT, 
      meas_settings = MEAS_DEFAULT, meas_weights = MEAS_WEIGHTS, sample_meas=False):
    """ Main function: take parameters and return an estimation of the fidelity 
    """
    if sample_meas:
        proba_sampled = np.abs(meas_weights)/np.sum(np.abs(meas_weights))
        index_sampled = np.random.choice(range(len(meas_settings)), p=proba_sampled)
        meas_sampled = meas_settings[index_sampled]
        circs = gen_meas_circuits(create_circ, [meas_sampled], experimental_params)
        qjobs = [transpile(c, optimization_level=2) for c in circs]
    else:
        circs = gen_meas_circuits(create_circ, meas_settings, experimental_params)
        qjobs = [transpile(c, optimization_level=2) for c in circs]

    submitted = []
    for jj in qjobs:
        submitted.append(execute(jj, simulator, shots = shots))
        if flags:
            print('submitted')

    current_status = str([str(jj.status()) for jj in submitted])
    if flags: print(current_status)
    ###### Uncomment following section to run on IBMQ
    #while 'RUN' in current_status or 'QUE' in current_status: # can optimise to only look at one at a time
    #    print(current_status)
    #    time.sleep(10)
    #    current_status = str([str(jj.status()) for jj in submitted])

    measurement_results = []
    for j in submitted:
        if flags: print(j.result().get_counts())
        measurement_results.append(Mean(j.result().get_counts()))

    #print(measurement_results)
    if sample_meas:
        measurement_results = np.array(measurement_results)
        weight_sampled = meas_weights[index_sampled]
        Fidelity = measurement_results if weight_sampled >0 else 1 - measurement_results
    else:
        Fidelity = np.dot(measurement_results, meas_weights)
    #print('Fidelity = {}'.format(str(Fidelity)))
    return np.squeeze(Fidelity)


x_opt = np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
x_loc = np.array([1., 0., 4., 0., 3., 0.]) * np.pi/2
F(x_loc)
F(x_opt)
F(x_loc, sample_meas=True)
F(x_opt, sample_meas=True)

