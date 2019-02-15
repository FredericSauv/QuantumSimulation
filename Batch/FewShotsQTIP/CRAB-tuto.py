#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-pulseoptim-CRAB-QFT.ipynb
"""
%matplotlib inline
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
max_wall_time = 300


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

Initial amplitudes output to file: ctrl_amps_initial_QFT_n_ts200.txt
***********************************

INFO:qutip.control.optimizer:Optimising pulse(s) using CRAB with 'fmin' (Nelder-Mead) method

Starting pulse optimisation
Final amplitudes output to file: ctrl_amps_final_QFT_n_ts200.txt

Report the results

result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

------------------------------------
---- Control optimisation stats ----
**** Timings (HH:MM:SS.US) ****
Total wall time elapsed during optimisation: 0:05:00.003775
Wall time computing Hamiltonians: 0:00:22.547817 (7.52%)
Wall time computing propagators: 0:04:24.789280 (88.26%)
Wall time computing forward propagation: 0:00:03.857988 (1.29%)
Wall time computing onward propagation: 0:00:03.634444 (1.21%)
Wall time computing gradient: 0:00:00 (0.00%)

**** Iterations and function calls ****
Number of iterations: 8610
Number of fidelity function calls: 10459
Number of times fidelity is computed: 10459
Number of gradient function calls: 0
Number of times gradients are computed: 0
Number of times timeslot evolution is recomputed: 10459

**** Control amplitudes ****
Number of control amplitude updates: 10458
Mean number of updates per iteration: 1.2146341463414634
Number of timeslot values changed: 2091599
Mean number of timeslot changes per update: 199.99990437942245
Number of amplitude values changed: 8350196
Mean number of amplitude changes per update: 798.4505641614076
------------------------------------
Final evolution
Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = False
Qobj data =
[[-0.47890815-0.25124383j -0.47862720-0.14293885j -0.47089104-0.19654053j
  -0.39814103-0.19780087j]
 [-0.48524763-0.18754841j  0.19902638-0.45262542j  0.45666538+0.17490099j
  -0.22150294+0.44348832j]
 [-0.44119050-0.18412439j  0.49542987+0.17619236j -0.42828736-0.17952947j
   0.50360656+0.16023168j]
 [-0.41755379-0.18434167j -0.18861574+0.44037804j  0.50645054+0.16836502j
   0.23840544-0.4695553j ]]

********* Summary *****************
Initial fidelity error 0.9483036893449602
Final fidelity error 0.0032564547889728512
Terminated due to Max wall time exceeded
Number of iterations 8611
Completed in 0:05:00.003775 HH:MM:SS.US

Plot the initial and final amplitudes

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

Versions

from qutip.ipynbtools import version_table

version_table()

Software	Version
QuTiP	4.1.0
Numpy	1.11.3
SciPy	0.18.1
matplotlib	2.0.0
Cython	0.25.2
Number of CPUs	4
BLAS Info	INTEL MKL
IPython	5.1.0
Python	3.6.0 |Anaconda 4.3.1 (64-bit)| (default, Dec 23 2016, 12:22:00) [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
OS	posix [linux]
Fri Jul 14 16:41:51 2017 BST
References: 3. Doria, P., Calarco, T. & Montangero, S. Optimal Control Technique for Many-Body Quantum Dynamics. Phys. Rev. Lett. 106, 1–4 (2011). 4. Caneva, T., Calarco, T. & Montangero, S. Chopped random-basis quantum optimization. Phys. Rev. A - At. Mol. Opt. Phys. 84, (2011).

This website does not host notebooks, it only renders notebooks available on other websites.

Delivered by Fastly, Rendered by Rackspace

nbviewer GitHub repository.

nbviewer version: aa567da

nbconvert version: 5.3.1

Rendered 2 minutes ago
