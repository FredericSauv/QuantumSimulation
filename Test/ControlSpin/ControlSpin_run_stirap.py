#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
break
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

#==============================================================================
# SETUP 1 T
#==============================================================================
#==============================================================================
# SETUP 1QT1 i.e qtrirt with gen of Z and X
#==============================================================================
T = 3
dt = 0.01
fT = 1
setup = '1QT2'
state_init = 'z0' # |0>
state_target = 'z1' # |1>
nb_trials = 1
x = np.arange(0, T, dt)
gp_iter = 150


#------------------------------------------------------------------------------
#                   Linear driving
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
fDummy = pfun.SquareExponential({'sigma':0.9, 'mu':2.4*T/5, 'l':T/6})
fDummy2 = pfun.SquareExponential({'sigma':0.9, 'mu':2.6*T/5 , 'l':T/6})
fDummy.PlotFunction(x)
fDummy2.PlotFunction(x)
model = cspin.ControlledSpin(controlFun = [fDummy, fDummy2], setup = setup, state_init = state_init, state_target = state_target, T=T, dt=dt)
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
#                   NM - 
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


# 2 control functions
overall = None
guess = None
control = [{'func':'squareexp'}, {'func':'squareexp'}]
nb_parameters = 6

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 'nm_maxiter':1000,
               'nm_maxfev':3000, 'nm_ftol':1e-6}


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


#------------------------------------------------------------------------------
#                   - DE
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


# 2 control functions
overall = [{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'},{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'}]
#overall = None
guess = None
control = [{'func':'squareexp'}, {'func':'squareexp'}]
nb_parameters = 6

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'DE', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1_0_5_0_2.5_0_1_0_5_0_2.5', 
               'params_bound':'range_0_1_0_5_0_2.5_0_1_0_5_0_2.5', 
               'de_maxiter':5, 'nm_maxfev':5000}


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

cfBestDE[0].PlotFunction(x)
cfBestDE[1].PlotFunction(x)


modelTestDE = cspin.ControlledSpin(cfBestDE, setup, state_init, state_target)
res_optimDE = modelTestDE.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True, debug = True)
modelTestDE.PrintPopulation()


#------------------------------------------------------------------------------
#                   BO - EI
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

# 2 control functions
overall = [{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'},{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'}]
#overall = None
guess = None
control = [{'func':'squareexp'}, {'func':'squareexp'}]
nb_parameters = 6


paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1_0_5_0.01_2.5_0_1_0_5_0.01_2.5', 
               'params_bound':'range_0_1_0_5_0.01_2.5_0_1_0_5_0.01_2.5', 
               'gp_verbose': False, 'gp_init': 2*6, 'gp_acq':'ei', 'gp_kernel': None,
               'gp_maxiter':100, 'gp_kappa':'linear_5'}


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

cfBestBO[0].PlotFunction(x)
cfBestBO[1].PlotFunction(x)


modelTestBO = cspin.ControlledSpin(cfBestBO, setup, state_init, state_target)
res_optimBO = modelTestBO.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True)




#------------------------------------------------------------------------------
#                   BO - EI
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

# 2 control functions
overall = [{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'},{'constraints':[[0,0], [T,0]], 'constraints_type':'sin'}]
#overall = None
guess = None
control = [{'func':'squareexp'}, {'func':'squareexp'}]
nb_parameters = 6


paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1_0_5_0.01_2.5_0_1_0_5_0.01_2.5', 
               'params_bound':'range_0_1_0_5_0.01_2.5_0_1_0_5_0.01_2.5', 
               'gp_verbose': False, 'gp_init': 2*6, 
               'gp_maxiter':50, 'gp_acq':'ucb', 'gp_kernel': None,
               'gp_kappa':'linear_5', 'gp_wnoise':0.1, 'gp_scaling':None}


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

cfBestBO[0].PlotFunction(x)
cfBestBO[1].PlotFunction(x)


modelTestBO = cspin.ControlledSpin(cfBestBO, setup, state_init, state_target)
res_optimBO = modelTestBO.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:neg', 'fluence', 'fluenceNorm'], store = True)