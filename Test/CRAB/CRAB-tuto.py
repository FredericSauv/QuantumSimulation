#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-pulseoptim-CRAB-QFT.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip.algorithms import qft
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen
example_name = 'QFT'

## Defining the physics
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5*identity(2)
# Drift Hamiltonian
H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = qft.qft(2)

## Defining the time evolution parameters
# Number of time slots
n_ts = 200
# Time allowed for the evolution
evo_time = 10

##Set the conditions which will cause the pulse optimisation to terminate
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 20000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 600


##Set to None to suppress output files
f_ext = None#"{}_n_ts{}.txt".format(example_name, n_ts)

##Create the optimiser objects
optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, 
                max_iter=max_iter, max_wall_time=max_wall_time,
                alg='CRAB', 
                dyn_type='UNIT', 
                prop_type='DIAG', 
                fid_type='UNIT', fid_params={'phase_option':'PSU'}, 
                log_level=log_level, gen_stats=True)
                

##Configure the pulses for each of the controls
dyn = optim.dynamics
# Control 1
crab_pgen = optim.pulse_generator[0]
# Start from a ramped pulse
guess_pgen = pulsegen.create_pulse_gen('LIN', dyn=dyn, 
                                           pulse_params={'scaling':3.0})
crab_pgen.guess_pulse = guess_pgen.gen_pulse()
crab_pgen.scaling = 0.0
# Add some higher frequency components
crab_pgen.num_coeffs = 5

# Control 2
crab_pgen = optim.pulse_generator[1]
# Apply a ramping pulse that will force the start and end to zero
ramp_pgen = pulsegen.create_pulse_gen('GAUSSIAN_EDGE', dyn=dyn, 
                                    pulse_params={'decay_time':evo_time/50.0})
crab_pgen.ramping_pulse = ramp_pgen.gen_pulse()

# Control 3
crab_pgen = optim.pulse_generator[2]
# Add bounds
crab_pgen.scaling = 0.5
crab_pgen.lbound = -2.0
crab_pgen.ubound = 2.0


# Control 4
crab_pgen = optim.pulse_generator[3]
# Start from a triangular pulse with small signal
guess_pgen = pulsegen.PulseGenTriangle(dyn=dyn)
guess_pgen.num_waves = 1
guess_pgen.scaling = 2.0
guess_pgen.offset = 2.0
crab_pgen.guess_pulse = guess_pgen.gen_pulse()
crab_pgen.scaling = 0.1

init_amps = np.zeros([n_ts, n_ctrls])
for j in range(dyn.num_ctrls):
    pgen = optim.pulse_generator[j]
    pgen.init_pulse()
    init_amps[:, j] = pgen.gen_pulse()

dyn.initialize_controls(init_amps)


## Run the pulse optimisation
# Save initial amplitudes to a text file
if f_ext is not None:
    pulsefile = "ctrl_amps_initial_" + f_ext
    dyn.save_amps(pulsefile)
    print("Initial amplitudes output to file: " + pulsefile)

print("***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

# Save final amplitudes to a text file
if f_ext is not None:
    pulsefile = "ctrl_amps_final_" + f_ext
    dyn.save_amps(pulsefile)
    print("Final amplitudes output to file: " + pulsefile)


result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial Control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])), 
             where='post')
ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Amplitudes")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])), 
             where='post', label='u{}'.format(j))
ax2.legend(loc=8, ncol=n_ctrls)
plt.tight_layout()
plt.show()
