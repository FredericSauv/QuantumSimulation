#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:43:10 2018

@author: fred
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:58:53 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Example to demonstrate configuration through manual creation of the
the optimiser and its child objects. Note that this is not necessary for
using the CRAB algorithm, it's just one way of configuring.
See the main Hadamard example for how to call the CRAB alg using the
pulseoptim functions.

The system in this example is a single qubit in a constant field in z
with a variable control field in x
The target evolution is the Hadamard gate irrespective of global phase

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging

logger = logging.get_logger()
# QuTiP control modules
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
#import qutip.control.errors as errors
#import qutip.control.pulsegen as pulsegen
#import sys
#sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')  # my GPyOpt's fork
#import GPyOpt
#import GPy
#import ctrlbayesoptim as cbo
example_name = 'Hadamard-CRAB-man_cfg'
log_level = logging.INFO


# Define the physics of the problem

nSpins = 1

# Note that for now the dynamics must be specified as ndarrays
# when using manual config
# This is until GitHub issue #370 is resolved
H_d = sigmaz()
H_c = [sigmax()]
# Number of ctrls
n_ctrls = len(H_c)

U_0 = identity(2 ** nSpins)
# Hadamard gate
# U_targ = Qobj(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2))
U_targ = hadamard_transform(nSpins)

# Evolution parameters
# time-slicing
n_ts = 10
# Total drive time
evo_time = 6.0

print("\n***********************************")
print("Creating and configuring control optimisation objects")

# Create the OptimConfig object
cfg = optimconfig.OptimConfig()
cfg.log_level = log_level

# Create the dynamics object
dyn = dynamics.DynamicsUnitary(cfg)
dyn.num_tslots = n_ts
dyn.evo_time = evo_time

# Physical parameters
dyn.target = U_targ
dyn.initial = U_0
dyn.drift_dyn_gen = H_d
dyn.ctrl_dyn_gen = H_c

# Create the TerminationConditions instance
tc = termcond.TerminationConditions()
tc.fid_err_targ = 1e-4
tc.min_gradient_norm = 1e-10
tc.max_iter_total = 200
tc.max_wall_time_total = 30
tc.break_on_targ = True

sts = stats.Stats()
dyn.stats = sts
init_amps = np.zeros([n_ts, n_ctrls])
dyn.initialize_controls(init_amps)

lbnd = -1
ubnd = 1
optim = optimizer.Optimizer(cfg, dyn)
optim.amp_lbound = lbnd
optim.amp_ubound = ubnd

dyn.stats = sts
optim.stats = sts
optim.config = cfg
optim.dynamics = dyn
optim.termination_conditions = tc
optim.method = 'L-BFGS-B'
optim.approx_grad = True

# 
# Run the optimisation - Qtrl
print("\n***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

print("\n***********************************")
print("Optimising complete. Stats follow:")
result.stats.report()
print("\nFinal evolution\n{}\n".format(result.evo_full_final))

print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
# print("wall time: ", result.wall_time
print("Completed in {} HH:MM:SS.US". \
      format(datetime.timedelta(seconds=result.wall_time)))
print("***********************************")

# 
# Try the Bayesian optimiser
# --- CHOOSE the objective
copt = cbo.QtrlBayesOptim(cfg, dyn)


def func_wrap(x):
    amps = np.reshape(x, (n_ts, n_ctrls))
    dyn.update_ctrl_amps(amps)
    res = dyn.fid_computer.get_fid_err()
    print(res)
    return res


objective = GPyOpt.core.task.SingleObjective(func_wrap)

bnd = (lbnd, ubnd)
bdict = {'name': 'var', 'type': 'continuous', 'domain': bnd}

space = [{'name':'amp{}'.format(i), 'type': 'continuous', 'domain': bnd} for i in range(n_ts)]

feasible_region = GPyOpt.Design_space(space=space)

# --- CHOOSE the model type
model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=10, verbose=False)

# --- CHOOSE the acquisition optimizer
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

# --- CHOOSE the type of acquisition
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 2)

bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
                                                initial_design)

# --- Stop conditions
max_time = None
max_iter = 40
tolerance = 1e-4  # distance between two consecutive observations

# Run the optimization
bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
# bo.plot_acquisition()
bo.plot_convergence()

print("Final fid err: ", bo.fx_opt)
print("BO time: ", bo.cum_time)

final_pulse_bo = np.reshape(bo.x_opt, (n_ts, n_ctrls))


###optim 2 (slower why??)
base_dico = {'acquisition_type': 'EI', 'domain': space,
             'optim_num_anchor': 15,
             'optim_num_samples': 10000,
             'initial_design_numdata':20}
bo2 = GPyOpt.methods.BayesianOptimization(func_wrap, **base_dico)
bo2.run_optimization(max_iter=max_iter, eps=0)
#exploitation
bo2.acquisition_type = 'LCB'
bo2.acquisition_weight = 0.000001
bo2.kwargs['acquisition_weight'] = 0.000001
bo2.acquisition = bo2._acquisition_chooser()
bo2.evaluator = bo2._evaluator_chooser()
bo2.run_optimization(20)
bo2.plot_convergence()



# 

## *************************************************************
## File extension for output files
#
# f_ext = "{}_n_ts{}.txt".format(example_name, n_ts)
#
## Run the optimisation
# print("\n***********************************")
# print("Starting pulse optimisation")
# result = optim.run_optimization()
#
# print("\n***********************************")
# print("Optimising complete. Stats follow:")
# result.stats.report()
# print("\nFinal evolution\n{}\n".format(result.evo_full_final))
#
# print("********* Summary *****************")
# print("Initial fidelity error {}".format(result.initial_fid_err))
# print("Final fidelity error {}".format(result.fid_err))
# print("Final gradient normal {}".format(result.grad_norm_final))
# print("Terminated due to {}".format(result.termination_reason))
# print("Number of iterations {}".format(result.num_iter))
##print("wall time: ", result.wall_time
# print("Completed in {} HH:MM:SS.US".\
#        format(datetime.timedelta(seconds=result.wall_time)))
# print("***********************************")
#
# Plot the initial and final amplitudes
fig1 = plt.figure()
# ax1 = fig1.add_subplot(2, 1, 1)
# ax1.set_title("Initial control amps")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Control amplitude")
# for j in range(n_ctrls):
#    ax1.step(result.time,
#             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
#             where='post')
#
ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time,
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
             label='GRAPE', where='post')
    ax2.step(result.time,
             np.hstack((final_pulse_bo[:, j], final_pulse_bo[-1, j])),
             linestyle='--', label="BO",
             where='post')
    ax2.step(result.time,
             np.hstack((final_pulse_bo[:, j], final_pulse_bo[-1, j])),
             linestyle='--', label="BO",
             where='post')
plt.show()
