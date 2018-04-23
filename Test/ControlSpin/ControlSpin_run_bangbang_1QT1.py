#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys

sys.path.append('../../')
#sys.path.append('../Utility/BayesianOptimization-master/')
import QuantumSimulation.ToyModels.ControlledSpin as cspin
import QuantumSimulation.Simulation.ControlledSpinOptim as cspinopt
import importlib as ilib
import numpy as np
import copy
import QuantumSimulation.Utility.ParametrizedFunctionFactory as pfun
import QuantumSimulation.Utility.Utility as ut
import matplotlib.pylab as plt
# import pdb

break
#==============================================================================
# SETUP 1QT1 i.e qtrirt with gen of Z and X
#==============================================================================
setup = 1
T = 3
dt = 0.01
fT = 1
setup = '1QT1'
state_init = 'z0' # 1st evector of Z operator
state_target = 'x0' # 1st evector of X operator
nb_parameters = 20
nb_trials = 1
x = np.arange(0, T, dt)



#------------------------------------------------------------------------------
#                   Linear driving
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
fLinear = pfun.LinearFunc([fT/(T-dt), 0])
model = cspin.ControlledSpin(controlFun = fLinear, setup = setup, state_init = state_init, state_target = state_target, T=T, dt=dt)
res_constant = model.Simulate(T, time_step = dt, method = 'testing', fom = 'last:f2t2', store = True)

## Some plots
st_tgt = model.state_tgt
st = model.state_PWCH
st_ad = model.state_ADIAB
en_ad = model.energies_ADIAB
adiabaticity = model.fidelity2(st, st_ad)

model.PrintPopulation()
model.PrintPopulation(st_ad)

#plt.plot(adiabaticity)
# Fidelity over time (to ensure that the last point is the max)
fidelity_t = [model.fidelity(st_tgt, s) for s in st]
plt.plot(fidelity_t)

#------------------------------------------------------------------------------
#                   ForceMax driving
#------------------------------------------------------------------------------
fForce = pfun.ConstantFunc(1)
model = cspin.ControlledSpin(controlFun = fForce, setup = setup, state_init = state_init, state_target = state_target, T=T, dt=dt)
res_constant = model.Simulate(T, time_step = dt, method = 'testing', fom = 'last:f2t2', store = True)

## Some plots
st_tgt = model.state_tgt
st = model.state_PWCH
st_ad = model.state_ADIAB
en_ad = model.energies_ADIAB
adiabaticity = model.fidelity2(st, st_ad)

model.PrintPopulation()
model.PrintPopulation(st_ad)

#plt.plot(adiabaticity)
# Fidelity over time (to ensure that the last point is the max)
fidelity_t = [model.fidelity2(st_tgt, s) for s in st]
plt.plot(fidelity_t)



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
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_0_1', 'nm_maxiter':1000,
               'nm_maxfev':3000, 'nm_ftol':1e-9}


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
res_optimNM = modelTestNM.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True)

modelTestNM.PrintPopulation()
resFullNM = optimNM.resOptim

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
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001', 'print': False}

paramsOptim = {'algo': 'DE', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 'de_maxiter':20,
               'nm_maxfev':5000}


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

modelTestDE.PrintPopulation()
resFullDE = optimDE.resOptim

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
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 2*20, 'gp_acq':'ucb', 'gp_kernel': None,
               'gp_maxiter':60, 'gp_kappa':'linear_5', 'gp_wnoise':0.1}


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
resFullBO = optimBO.resOptim

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
             'T': T, 'fom_name': 'last:f2t2:neg_fluenceNorm:0.0001', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 2*20, 'gp_acq':'ei', 'gp_kernel': '',
               'gp_maxiter':60, 'gp_kappa':'linear_5'}


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
resFullBO = optimBO.resOptim


#------------------------------------------------------------------------------
#                   Comp
#------------------------------------------------------------------------------
name = ['NM', 'DE', 'BO']
list_res_full = [resFullNM, resFullDE, resFullBO]
list_res_tfom = [np.array(res['time_fom']) for res in list_res_full]
list_res_nfom = [np.array(res['fev_fom']) for res in list_res_full]

fig, ax = plt.subplots()
for i in range(len(name)):
    res = list_res_nfom[i]
    ax.plot(res[:,0], np.log(res[:,1]), label=name[i])
ax.legend()
plt.xlim([0,1000])





