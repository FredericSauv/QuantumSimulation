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
import matplotlib.pylab as plt
# import pdb
#import matplotlib.pylab as plt
break
#==============================================================================
# SETUP 1 T
#==============================================================================
T = 1.5
dt = 0.01
fT = 1
setup = '1Q1'
state_init = 'zero'
state_target = 'minus'
nb_parameters = 15
nb_trials = 1
x = np.arange(0, T, dt)

#------------------------------------------------------------------------------
#                   NM - 
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)

noise = {'fom':'normal_0_0.00', 'Ex':'normal_0_0.00', 'Ez':'normal_0_0.00'}
overall = {'bounds': [0,1], 'constraints':[[0,0]], 'constraints_type':'shift'}
guess = {'func': 'constant', 'c0':1}
control = {'func':'step', 'name_rdm':None}

paramsSim = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': 'last:f2t2:neg:rdmtime_fluenceNorm:0.0001_smooth:0.0005',
             'noise': noise}

paramsOptim = {'algo': 'NM', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_-2_2', 'params_bound':'range_0_1', 'nm_maxiter':10,
               'nm_maxfev':10000, 'nm_ftol':1e-6}

paramsTesting = {'setup' : setup, 'target_state_name' : state_target, 'init_state_name':state_init, 
             'overall': overall, 'guess': None, 'control': control, 'dt': dt, 
             'T': T, 'fom_name': ['last:f2t2', 'fluenceNorm', 'smooth', 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005'],
             'noise': None}

bestRes = None
fomBest = 41000


optimNM = cspinopt.ToyModelOptim(paramsSim, paramsOptim, paramsTesting)
resNM = optimNM.Run(writeLogs = True)

cfBestNM = copy.copy(optimNM.simulator.controlFun)
cfBestNM2 = copy.copy(optimNM.simulator_testing.controlFun)
cfBestNM.PlotFunction(x)
cfBestNM2.PlotFunction(x)

noise_testing = {'fom':'normal_0_0.00', 'Ex':'normal_0_0.00', 'Ez':'normal_0_0.00'}
modelTestNM = cspin.ControlledSpin(cfBestNM, setup, state_init, state_target, noise = noise_testing)
res_optimNM = modelTestNM.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2', 'fluenceNorm', 'smooth', 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005'], store = True)

modelTestNM2 = cspin.ControlledSpin(cfBestNM2, setup, state_init, state_target, noise = noise_testing)
res_optimNM2 = modelTestNM2.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2', 'fluenceNorm', 'smooth', 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005'], store = True)


modelTestNM.PrintPopulation()

#------------------------------------------------------------------------------
#                   StepFunc - BangBang - DE
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
paramsOptim = {'algo': 'DE', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 'de_maxiter':2}


optimNM = cspinopt.ToyModelOptim(paramsSim, paramsOptim, paramsTesting)
resNM = optimNM.Run(writeLogs = True)

cfBestNM = copy.copy(optimNM.simulator.controlFun)
cfBestNM2 = copy.copy(optimNM.simulator_testing.controlFun)
cfBestNM.PlotFunction(x)
cfBestNM2.PlotFunction(x)

noise_testing = {'fom':'normal_0_0.00', 'Ex':'normal_0_0.00', 'Ez':'normal_0_0.00'}
modelTestNM = cspin.ControlledSpin(cfBestNM, setup, state_init, state_target, noise = noise_testing)
res_optimNM = modelTestNM.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2', 'fluenceNorm', 'smooth', 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005'], store = True)

modelTestNM2 = cspin.ControlledSpin(cfBestNM2, setup, state_init, state_target, noise = noise_testing)
res_optimNM2 = modelTestNM2.Simulate(T, time_step = dt, method = 'testing', fom = ['last:f2t2', 'fluenceNorm', 'smooth', 'last:f2t2:neg_fluenceNorm:0.0001_smooth:0.0005'], store = True)



#------------------------------------------------------------------------------
#                   StepFunc - BangBang - BO
#------------------------------------------------------------------------------
ilib.reload(ut)
ilib.reload(pfun)
ilib.reload(cspin)
ilib.reload(cspinopt)
paramsOptim = {'algo': 'GP', 'name_res': None, 'nb_params' :  nb_parameters, 
               'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
               'gp_verbose': False, 'gp_init': 2*20, 'gp_acq':'ei', 'gp_kernel': None,
               'gp_wnoise': 0.01, 'gp_maxiter':500}


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


modelTestBO = cspin.ControlledSpin(cfBestBO, setup, state_init, state_target, noise = noise_testing)
res_optimBO = modelTestBO.Simulate(T, time_step = dt, method = 'PWCH', fom = ['last:f2t2:rdmtime', 'fluence', 'fluenceNorm'], store = True)



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
plt.xlim([0,20000])

cfBestNM.PlotFunction(x)
cfBestDE.PlotFunction(x)
cfBestBO.PlotFunction(x)

#------------------------------------------------------------------------------
#                   MoreAnalysis
#------------------------------------------------------------------------------
# L2 on last
if(False):
    ilib.reload(pfun)
    from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pfunfun
    from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import CollectionParametrizedFunctionFactory as pfuncollec
    
    pfunfun.l2_norm_list([cfBestBO, cfBestDE, cfBestNM], x)
    
    # Pick the best then compare to all the stuff over time
    list_funcNM = [pfuncollec.Repr2Fun(r.replace(" ", "").replace(",fixedParams=", "")) for r, _ in optimNM.resOptim['func_fom']]
    list_funcDE = [pfuncollec.Repr2Fun(r.replace(" ", "").replace(",fixedParams=", "")) for r, _ in optimDE.resOptim['func_fom']]
    list_funcBO = [pfuncollec.Repr2Fun(r.replace(" ", "").replace(",fixedParams=", "")) for r, _ in optimBO.resOptim['func_fom']]
    
    
    
    list_fomNM = [fom for _, fom in optimNM.resOptim['func_fom']]
    list_fomDE = [fom for _, fom in optimDE.resOptim['func_fom']]
    list_fomBO = [fom for _, fom in  optimBO.resOptim['func_fom']]
    
    #define optimal
    optim_func = cfBestBO
    
    close_optimalNM = pfuncollec.l2_norm_list(list_funcNM, x, optim_func)
    close_optimalDE = pfuncollec.l2_norm_list(list_funcDE, x, optim_func)
    close_optimalBO = pfuncollec.l2_norm_list(list_funcBO, x, optim_func)
    
    list_iterNM = [n for n, _ in optimNM.resOptim['fev_fom']]
    list_iterDE = [n for n, _ in optimDE.resOptim['fev_fom']]
    list_iterBO = [n for n, _ in  optimBO.resOptim['fev_fom']]
    list_iter = [list_iterNM, list_iterDE, list_iterBO]
    
    close_optimal = [close_optimalNM, close_optimalDE, close_optimalBO]
    list_fom = [list_fomNM, list_fomDE, list_fomBO]
    
    fig, ax = plt.subplots()
    #ax.plot(list_fom[0], close_optimal[0])
    for i in range(len(name)):
        ax.plot(list_iter[i], close_optimal[i], label=name[i])
    ax.legend()
    plt.xlim([200,500])
    plt.ylim([0, 0.1])
