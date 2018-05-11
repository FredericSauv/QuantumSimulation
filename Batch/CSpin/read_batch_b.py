#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
break
import sys
sys.path.append('../../')

from  QuantumSimulation.Simulation.Spin.ControlledSpinOptimBatch import ControlledSpinOptimBatch as OptBatch
from  QuantumSimulation.Utility import Helper as ut
import numpy as np
from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pf
import QuantumSimulation.ToyModels.ControlledSpin as cs
from functools import partial
import importlib as ilib
ilib.reload(ut)


minlogFunc = (lambda x: np.log10(1-x))
#==============================================================================
# SETUP 1 T Normal
#==============================================================================
all_name = ['Batch0_NM', 'Batch0_DE', 'Batch0_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

list_resObjects = [OptBatch.process_list_res(all_res[n], printing = True) for n, _ in enumerate(all_name)]

list_res_tmp = list_resObjects
lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + '_zoom', 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom', 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)

lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)


lookat = 'opt_control'
dico_plot = {'legend': all_name, 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot)




#==============================================================================
# SETUP 1 T Noise
# TODO: Add real res (i.e. compute against the real model)
# TODO: testing on several random instance
#==============================================================================
all_name = ['Batch0noise_NM', 'Batch0noise_DE', 'Batch0noise_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

list_resObjects_noise = [OptBatch.process_list_res(all_res[n], printing = True) for n, _ in enumerate(all_name)]

list_res_tmp = list_resObjects_noise
lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + '_zoom', 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom', 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)

lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)


lookat = 'opt_control'
dico_plot = {'legend': all_name, 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot)




#==============================================================================
# SETUP 1 T Noise Proj 10
# TODO: 
# TODO: 
#==============================================================================
all_name = ['Batch0proj10_NM', 'Batch0proj10_DE', 'Batch0proj10_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

list_resObjects_proj10 = [OptBatch.process_list_res(all_res[n], printing = True) for n, _ in enumerate(all_name)]

list_res_tmp = list_resObjects_proj10
lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom', 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fom'
dico_plot = {'legend': all_name, 'xlim':[0, 15000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(FOM)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 1000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)

lookat = 'evol_ideal_fidelity'
dico_plot = {'legend': all_name, 'xlim':[0, 10000], 'suptitle': lookat, 'xlabel':'nb evals', 'ylabel':'log10(1-F)'}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)


lookat = 'opt_control'
dico_plot = {'legend': all_name, 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_res_tmp], component = 'avg', dico_plot = dico_plot)


#==============================================================================
# DISGRESION 
#==============================================================================
all_name = ['Batch0_GP', 'MP_Batch0_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

list_resObjects_comp_MP = [OptBatch.process_list_res(all_res[n], printing = True) for n, _ in enumerate(all_name)]




#==============================================================================
# SETUP 1 T Noise Proj 100 TODO
# TODO: 
# TODO: 
#==============================================================================
all_name = ['Batch0proj100_NM', 'Batch0proj100_DE', 'Batch0proj100_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

all_fom_nev = [extract_fom_nev(res) for res in all_res]
all_fom_test = [extract_res_test(res) for res in all_res]

# avg, mini, maxi, std, avg_pstd, avg_mstd
all_stats_learning = [ut.merge_and_stats_TS(extract_fom_nev(res)) for res in all_res]
all_tested_res = np.array([ut.get_stats(extract_res_test(res)) for res in all_res])
print('Proj100')
print('TESTED RES:avg')
print(all_tested_res[:,0])
print('TESTED RES:min')
print(all_tested_res[:,1])
print('TESTED RES:max')
print(all_tested_res[:,2])

x_lim = 350
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats, 'minmax', dico_plot = {'xlim':[0, x_lim]})

x_lim = 12000
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats, 'minmax', dico_plot = {'xlim':[0, x_lim]})






#==============================================================================
# SETUP 1 T NoConstraint
#==============================================================================
all_name = ['Batch0nc_NM', 'Batch0nc_DE', 'Batch0nc_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

all_fom_nev = [extract_fom_nev(res) for res in all_res]
all_fom_test = [extract_res_test_index(res,1) for res in all_res]

# avg, mini, maxi, std, avg_pstd, avg_mstd
all_stats_learning = [ut.merge_and_stats_TS(extract_fom_nev(res)) for res in all_res]
all_tested_res = np.array([ut.get_stats(extract_res_test(res)) for res in all_res])
print('NC')
print('TESTED RES:avg')
print(all_tested_res[:,0])
print('TESTED RES:min')
print(all_tested_res[:,1])
print('TESTED RES:max')
print(all_tested_res[:,2])



# Res -> Run -> Config
#            -> Optim -> fev_fom
#            -> Testing
#

x_lim = 350
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim]})

x_lim = 12000
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim], 'ylim':[0,0.1]})



#==============================================================================
# Look at res MP
#==============================================================================
all_name = ['MP_Batch0_GP', 'Batch0_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

all_fom_nev = [extract_fom_nev(res) for res in all_res]
all_fom_test = [extract_res_test_index(res,1) for res in all_res]

# avg, mini, maxi, std, avg_pstd, avg_mstd
all_stats_learning = [ut.merge_and_stats_TS(extract_fom_nev(res)) for res in all_res]
all_tested_res = np.array([ut.get_stats(extract_res_test(res)) for res in all_res])
print('NC')
print('TESTED RES:avg')
print(all_tested_res[:,0])
print('TESTED RES:min')
print(all_tested_res[:,1])
print('TESTED RES:max')
print(all_tested_res[:,2])



# Res -> Run -> Config
#            -> Optim -> fev_fom
#            -> Testing
#

x_lim = 350
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim]})

x_lim = 12000
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim], 'ylim':[0,0.1]})


#==============================================================================
# SETUP T 30
#==============================================================================
all_name = ['Batch0_T30b']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]

all_fom_nev = [extract_fom_nev(res) for res in all_res]
all_fom_test = [extract_res_test(res) for res in all_res]

# avg, mini, maxi, std, avg_pstd, avg_mstd
all_stats_learning = [ut.merge_and_stats_TS(extract_fom_nev(res)) for res in all_res]
all_tested_res = np.array([ut.get_stats(extract_res_test(res)) for res in all_res])
print('NORMAL')
print('TESTED RES:avg')
print(all_tested_res[:,0])
print('TESTED RES:min')
print(all_tested_res[:,1])
print('TESTED RES:max')
print(all_tested_res[:,2])



# Res -> Run -> Config
#            -> Optim -> fev_fom
#            -> Testing
#

x_lim = 350
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim]})

x_lim = 12000
#plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0, x_lim]})
plot_from_list_stats(all_stats_learning, 'minmax', dico_plot = {'xlim':[0, x_lim]})


