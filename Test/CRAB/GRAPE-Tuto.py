#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:14:40 2019

@author: fred
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
from qutip import cnot, sigmax, sigmay, identity, sigmaz, tensor
#from qutip.control import *
import qutip.logging_utils as logging
logger = logging.get_logger()
log_level = logging.INFO
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

example_name = 'CNOT'


## MODEL
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5*identity(2)
H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz)) # Drift Hamiltonian
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)] # The (four) control Hamiltonians

#H_d = (tensor(Sz, Si) + tensor(Si, Sz)) # Drift Hamiltonian
#H_c = [tensor(Sz, Sz), tensor(Sx, Si) + tensor(Si, Sx)] # The (four) control Hamiltonians

n_ctrls = len(H_c)
U_0 = identity(4) #start
U_targ = cnot() # tget

## Defining pulse
# Duration of each timeslot
dt = 0.05
# List of evolution times to try
evo_times = [1.9, 1.95, 2.]
n_evo_times = len(evo_times)
evo_time = evo_times[0]
n_ts = int(float(evo_time) / dt)
results = list()


### OPTIMIZATION
fid_err_targ = 1e-5 # Fidelity error target
max_iter = 500 # Maximum iterations for the optisation algorithm
max_wall_time = 120 # Maximum (elapsed) time allowed in seconds
min_grad = 1e-20 # Minimum gradient (sum of gradients squared)
p_type = 'LIN' # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|

optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                amp_lbound=-5.0, amp_ubound=5.0, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                optim_method='fmin_l_bfgs_b',
                method_params={'max_metric_corr':20, 'accuracy_factor':1e8},
                dyn_type='UNIT', 
                fid_params={'phase_option':'PSU'},
                init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)



# **** get handles to the other objects ****
optim.test_out_files = 0
dyn = optim.dynamics
dyn.test_out_files = 0
p_gen = optim.pulse_generator
                

for i in range(n_evo_times):
    # Generate the tau (duration) and time (cumulative) arrays
    # so that it can be used to create the pulse generator
    # with matching timeslots
    dyn.init_timeslots()
    if i > 0:
        # Create a new pulse generator for the new dynamics
        p_gen = pulsegen.create_pulse_gen(p_type, dyn)
        
    #Generate different initial pulses for each of the controls
    init_amps = np.zeros([n_ts, n_ctrls])
    if (p_gen.periodic):
        phase_diff = np.pi / n_ctrls
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse(start_phase=phase_diff*j)
    elif (isinstance(p_gen, pulsegen.PulseGenLinear)):
        for j in range(n_ctrls):
            p_gen.scaling = float(j) - float(n_ctrls - 1)/2
            init_amps[:, j] = p_gen.gen_pulse()
    elif (isinstance(p_gen, pulsegen.PulseGenZero)):
        for j in range(n_ctrls):
            p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
            init_amps[:, j] = p_gen.gen_pulse()
    else:
        # Should be random pulse
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse()
    
    dyn.initialize_controls(init_amps)
    


    print("***********************************")
    print("\n+++++++++++++++++++++++++++++++++++")
    print("Starting pulse optimisation for T={}".format(evo_time))
    print("+++++++++++++++++++++++++++++++++++\n")
    result = optim.run_optimization()
    results.append(result)


    # Report the results
    result.stats.report()
    print("Final evolution\n{}\n".format(result.evo_full_final))
    print("********* Summary *****************")
    print("Final fidelity error {}".format(result.fid_err))
    print("Final gradient normal {}".format(result.grad_norm_final))
    print("Terminated due to {}".format(result.termination_reason))
    print("Number of iterations {}".format(result.num_iter))
    print("Completed in {} HH:MM:SS.US".format(
            datetime.timedelta(seconds=result.wall_time)))
    
    if i+1 < len(evo_times):
        # reconfigure the dynamics for the next evo time
        evo_time = evo_times[i+1]
        n_ts = int(float(evo_time) / dt)
        dyn.tau = None
        dyn.evo_time = evo_time
        dyn.num_tslots = n_ts







fig1 = plt.figure(figsize=(12,8))
for i in range(n_evo_times):
    #Initial amps
    ax1 = fig1.add_subplot(2, n_evo_times, i+1)
    ax1.set_title("Init amps T={}".format(evo_times[i]))
    # ax1.set_xlabel("Time")
    ax1.get_xaxis().set_visible(False)
    if i == 0:
        ax1.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax1.step(results[i].time, 
             np.hstack((results[i].initial_amps[:, j], 
                        results[i].initial_amps[-1, j])), 
                 where='post')
        
    ax2 = fig1.add_subplot(2, n_evo_times, i+n_evo_times+1)
    ax2.set_title("Final amps T={}".format(evo_times[i]))
    ax2.set_xlabel("Time")
    #Optimised amps
    if i == 0:
        ax2.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax2.step(results[i].time, 
             np.hstack((results[i].final_amps[:, j], 
                        results[i].final_amps[-1, j])), 
                 where='post')

plt.tight_layout()
plt.show()

