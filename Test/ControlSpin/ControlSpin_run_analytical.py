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
# import pdb
# import matplotlib.pylab as plt

#==============================================================================
# SETUP 1 T
#==============================================================================
setup = 1
T = 1
dt = 0.01
fT = 1
setup = '1Q1'
state_init = 'zero'
state_target = 'minus'
nb_parameters = 20
nb_trials = 1
x = np.arange(0, T, dt)
break

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
               'gp_maxiter':20, 'gp_kappa':'linear_5'}


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
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
ilib.reload(cspin)
ilib.reload(pfun)
funToTest = pfun.LinearFunc([fT/(T-dt), 0])
model = cspin.ControlledSpin(controlFun = funToTest, setup = setup, state_init = state_init, state_target = state_target, T=T, dt=dt)
res_constant = model.Simulate(T, time_step = dt, method = 'testing', fom = 'lstf2t', store = True)

## Some plots
st_tgt = model.state_tgt
st = model.state_PWCH
st_ad = model.state_ADIAB
en_ad = model.energies_ADIAB
adiabaticity = model.fidelity_distance_t(st, st_ad)

# Instantaneous eigen values
#plt.plot(np.real(en_ad[:,0]))
#plt.plot(np.real(en_ad[:,1]))

# Adiabaticity defined as fidelity to instantaneous eigen states
plt.plot(adiabaticity)

# pop in o and 1 (should add +/-)
pb = model.probaFromState(st)
plt.plot(pb[:,0])
plt.plot(pb[:,1])

# Fidelity over time (to ensure that the last point is the max)
fidelity_t = [model.fidelity_distance(st_tgt, s) for s in st]
plt.plot(fidelity_t)
ham

#------------------------------------------------------------------------------
#                   Optim - Fourrier Series - CRAB 
#------------------------------------------------------------------------------
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': None, 'constraints':None, 'constraints_type':'scale'}
guess = {'func': 'linear', 'a':fT/(T-dt), 'b':0}
control = {'func':'fourFixedCt', 'c0':1, 'constraints':[[0,1], [T,1]], 'constraints_type':'sin', 'name_rdm':'CRAB'}

paramsSim = {'setup': setup, 'target_state_name':state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': guess, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':650,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)

#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)


modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)



#------------------------------------------------------------------------------
#                   Optim - Sin + trend
#------------------------------------------------------------------------------
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

overall = {'bounds': None, 'constraints':None, 'constraints_type':'scale'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'sineTrendFixedCt', 'b':0, 'name_rdm':None}

paramsSim = {'setup':setup, 'target_state_name':state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': guess, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)

modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)


#------------------------------------------------------------------------------
#                   Optim - Fourrier Series - CRAB 
#------------------------------------------------------------------------------
ilib.reload(cspin)
ilib.reload(cspinopt)
ilib.reload(pfun)

overall = {'bounds': None, 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'fourFixedCt', 'c0':0, 'name_rdm':'CRAB'}

paramsSim = {'setup':setup, 'target_state_name':state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': guess, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)

#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)


modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)




#------------------------------------------------------------------------------
#                   Optim - Chebyshev
#------------------------------------------------------------------------------
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': None, 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'chebyFixedCt', 'c0':0, 'name_rdm':None}

paramsSim = {'setup':setup, 'target_state_name':state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': guess, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)

modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)




#------------------------------------------------------------------------------
#                   Optim - Cheby + trend
#------------------------------------------------------------------------------
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': None, 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'chebyFixedTrend', 'a':fT/(T-dt), 'b':0, 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)

modelTest = cspin.ControlledSpin(controlFun = cfBest, setup = 1, state_init = 'zero', state_target = 'minus')
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)


#------------------------------------------------------------------------------
#                   StepFunc
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)


overall = {'bounds': None, 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': 0.1, 
             'T': T, 'fom_name': 'lstf2tNeg_fluence0.2', 'print': False,}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40}


bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.Run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
c = np.array([cfBest(t) for t in x])
g = np.array([cfBest.listFun[1](t) for t in x])
f = np.array([cfBest.listFun[0](t) for t in x])

plt.plot(x, c)
plt.plot(x, g)
plt.plot(x, f)
plt.plot(x, f*g)

modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'testing', fom = ['lstf2t', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)


#------------------------------------------------------------------------------
#                   StepFunc - BangBang - DE
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
T=1

overall = {'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluenceNorm0.005', 'print': False}

paramsOptim = {'algo': 'DE', 'name_res': None, 'nb_params' :  20, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 'nm_maxiter':1000,
               'nm_maxfev':3000}

paramsTesting = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg', 'print': False}



bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.Run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
cfBest.PlotFunction(x)


modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'PWCH', fom = ['lstf2tNeg', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)



#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
T=1

overall = {'bounds': [0, 1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg_fluenceNorm0.005', 'print': False}

paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  20, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 2*20, 'gp_acq':'ei', 'gp_kernel': '',
               'gp_maxiter':50, 'gp_kappa':'linear_5'}

paramsTesting = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'lstf2tNeg', 'print': False}



bestRes = None
fomBest = 41000

for i in np.arange(nb_trials):
    print(i)
    optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
    optim.Run()

    fomNew = optim.best['fom']
    if fomNew < fomBest:
        bestRes = optim.best
        fomBest = fomNew
        cfBest = copy.copy(optim.simulator.controlFun)
        print(bestRes)


#resOptim = optim.resOptim
x = np.arange(0, T, dt)
cfBest.PlotFunction(x)


modelTest = cspin.ControlledSpin(cfBest, setup, state_init, state_target)
res_optim = modelTest.Simulate(T, time_step = dt, method = 'PWCH', fom = ['lstf2tNeg', 'lstf2tNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)







#------------------------------------------------------------------------------
#                   D-CRAB
#------------------------------------------------------------------------------
ilib.reload(cspin)
ilib.reload(cspinopt)

guess = {'function': 'linear', 'f0': 0,'fT':1}
control = {'function':'fourierAmplitude', 'c0':1, 'name_rdm':'D-CRAB', 'norm':1, 'norm_t':T}

paramsSim = {'setup':1, 'target_state_name':'minus', 'init_state_name':'zero', 
             'guess': guess, 'control': control, 'dt': 0.01, 'T': T, 
             'fom_name': 'f2targetNeg_fluenceNorm0.05', 'print': False,}

paramsOptim = {'algo': 'D-CRAB', 'name_res': None, 'nb_params' :  6, 
               'params_init':'uniform_-2_2', 'params_bound':'range_-1_1', 'nm_maxiter':400,
               'nm_maxfev':3000, 'gp_verbose': False, 'gp_init': 40, 'nb_superiter': 10}


optim = cspinopt.ToyModelOptim(paramsSim, paramsOptim, None)
optim.run()

cfBest = copy.copy(optim.simulator.controlFun)

modelTest = cspin.ControlledSpin(controlFun = cfBest, setup = 1, state_init = 'zero', state_target = 'minus')
res_optim = modelTest.Simulate(2, time_step = 0.01, method = 'testing', fom = ['f2target', 'f2targetNeg_fluenceNorm0.05', 'fluence', 'fluenceNorm'], store = True)

x = np.arange(0, 2, 0.001)
c = np.array([cfBest(t) for t in x])
plt.plot(x, c)

st = modelTest.state_PWCH
pb = modelTest.ss.probaFromState(st)
plt.plot(pb[:,0])
plt.plot(pb[:,1])
