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

#==============================================================================
# SETUP 1QT1 i.e qtrirt with gen of Z and X
#==============================================================================
T = 1
dt = 0.01
fT = 1
setup = '1QT1'
state_init = 'z0' # 1st evector of Z operator
state_target = 'x0' # 1st evector of X operator
nb_parameters = 30
nb_trials = 1
x = np.arange(0, T, dt)
gp_iter = 100



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
#                   StepFunc - BangBang - BO - ucb/5
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 30, 'gp_acq':'ucb', 'gp_kernel': None,
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim1 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim1.Run(writeLogs=True)

    fomNew = optim1.best['fom']
    if fomNew < fomBest:
        bestRes = optim1.best
        fomBest1 = fomNew
        cfBest1 = copy.copy(optim1.simulator.controlFun)
        print(bestRes)


cfBest1.PlotFunction(x)

modelTest1 = cspin.ControlledSpin(cfBest1, setup, state_init, state_target)
res_optim1 = modelTest1.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull1 = optim1.resOptim

modelTest1.PrintPopulation()

#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO - EI
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 30, 'gp_acq':'ei', 'gp_kernel': None,
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim2 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim2.Run(writeLogs=True)

    fomNew = optim2.best['fom']
    if fomNew < fomBest:
        bestRes = optim2.best
        fomBest2 = fomNew
        cfBest2 = copy.copy(optim2.simulator.controlFun)
        print(bestRes)


cfBest2.PlotFunction(x)

modelTest2 = cspin.ControlledSpin(cfBest2, setup, state_init, state_target)
res_optim2 = modelTest2.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull2 = optim2.resOptim




#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO - ucb5 / lhs
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 'lhs_30', 'gp_acq':'ucb', 'gp_kernel': None,
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim3 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim3.Run(writeLogs=True)

    fomNew = optim3.best['fom']
    if fomNew < fomBest:
        bestRes = optim3.best
        fomBest3 = fomNew
        cfBest3 = copy.copy(optim3.simulator.controlFun)
        print(bestRes)


cfBest3.PlotFunction(x)

modelTest3 = cspin.ControlledSpin(cfBest3, setup, state_init, state_target)
res_optim3 = modelTest3.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull3 = optim3.resOptim


#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO - ucb5 / ARD / No scale
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 'lhs_30', 'gp_acq':'ucb', 'gp_kernel': 'ard',
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim4 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim4.Run(writeLogs=True)

    fomNew = optim4.best['fom']
    if fomNew < fomBest:
        bestRes = optim4.best
        fomBest4 = fomNew
        cfBest4 = copy.copy(optim4.simulator.controlFun)
        print(bestRes)


cfBest4.PlotFunction(x)

modelTest4 = cspin.ControlledSpin(cfBest4, setup, state_init, state_target)
res_optim4 = modelTest4.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull4 = optim4.resOptim



#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO - ucb5 / ARD / No scale
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 'lhs_30', 'gp_acq':'ucb', 'gp_kernel': 'ard_aniso',
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim5 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim5.Run(writeLogs=True)

    fomNew = optim5.best['fom']
    if fomNew < fomBest:
        bestRes = optim5.best
        fomBest5 = fomNew
        cfBest5 = copy.copy(optim5.simulator.controlFun)
        print(bestRes)


cfBest5.PlotFunction(x)

modelTest5 = cspin.ControlledSpin(cfBest5, setup, state_init, state_target)
res_optim5 = modelTest5.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull5 = optim5.resOptim






#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO - ucb5 / ARD / No scale
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = None #{'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 'lhs_30', 'gp_acq':'ucb', 'gp_kernel': 'ard_aniso',
               'gp_maxiter':gp_iter, 'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':1.0}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim6 = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim6.Run(writeLogs=True)

    fomNew = optim6.best['fom']
    if fomNew < fomBest:
        bestRes = optim6.best
        fomBest6 = fomNew
        cfBest6 = copy.copy(optim6.simulator.controlFun)
        print(bestRes)


cfBest3.PlotFunction(x)

modelTest6 = cspin.ControlledSpin(cfBest6, setup, state_init, state_target)
res_optim6 = modelTest6.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2'], store = True)
resFull6 = optim6.resOptim




