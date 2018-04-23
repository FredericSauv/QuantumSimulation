#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
sys.path.append('../../../')
import QuantumSimulation.ToyModels.ControlledSpin as cspin
import QuantumSimulation.Simulation.Spin.ControlledSpinOptim as cspinopt
import QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory as pfun
import QuantumSimulation.Utility.Helper as ut

import importlib as ilib
import numpy as np
import copy
import pdb
import matplotlib.pylab as plt
# import pdb
#import matplotlib.pylab as plt

#==============================================================================
# SETUP 1 T
#==============================================================================
T = 1
dt = 0.01
fT = 1
setup = '1Q1'
state_init = 'zero'
state_target = 'minus'
nb_parameters = 50
nb_trials = 1
x = np.arange(0, T, dt)

#------------------------------------------------------------------------------
#                   NM - 
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': [0,1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_0_1', 'nm_maxiter':3000,
               'nm_maxfev':10000, 'nm_ftol':1e-6}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optimNM = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optimNM.Run(writeLogs = True)

    fomNew = optimNM.best['fom']
    if fomNew < fomBest:
        bestRes = optimNM.best
        fomBest = fomNew
        cfBestNM = copy.copy(optimNM.simulator.controlFun)
        print(bestRes)


cfBestNM.PlotFunction(x)

modelTestNM = cspin.ControlledSpin(cfBestNM, setup, state_init, state_target)
res_optimNM = modelTestNM.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm', 'smooth'], store = True)


#------------------------------------------------------------------------------
#                   StepFunc - BangBang - DE
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005', 'print': False}

paramsOptim = {'algo': 'DE', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 'de_maxiter':50}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optimDE = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optimDE.Run(writeLogs = True)

    fomNew = optimDE.best['fom']
    if fomNew < fomBest:
        bestRes = optimDE.best 
        fomBest = fomNew
        cfBestDE = copy.copy(optimDE.simulator.controlFun)
        print(bestRes)

cfBestDE.PlotFunction(x)


modelTestDE = cspin.ControlledSpin(cfBestDE, setup, state_init, state_target)
res_optimDE = modelTestDE.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True)



#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = {'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.000_smooth:0.0005', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 2*20, 'gp_acq':'ei', 'gp_kernel': None,
               'gp_maxiter':300}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optimBO = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optimBO.Run(writeLogs=True)

    fomNew = optimBO.best['fom']
    if fomNew < fomBest:
        bestRes = optimBO.best
        fomBest = fomNew
        cfBestBO = copy.copy(optimBO.simulator.controlFun)
        print(bestRes)

cfBestBO.PlotFunction(x)


modelTestBO = cspin.ControlledSpin(cfBestBO, setup, state_init, state_target)
res_optimBO = modelTestBO.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True)



#------------------------------------------------------------------------------
#                   Comp
#------------------------------------------------------------------------------
name = ['NM', 'DE', 'BO']
list_res_full = [optimNM.resOptim, optimDE.resOptim, optimBO.resOptim]
list_res_tfom = [np.array(res['time_fom']) for res in list_res_full]
list_res_nfom = [np.array(res['fev_fom']) for res in list_res_full]

fig, ax = plt.subplots()
for i in range(len(name)):
    res = list_res_nfom[i]
    ax.plot(res[:,0], np.log(res[:,1]), label=name[i])
ax.legend()
plt.xlim([0,50000])


