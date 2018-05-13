#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
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


#==============================================================================
# SETUP T 30
#==============================================================================


### Get a lot of res
all_name = 'Batch0_T30b'
key1 = ['config', 'paramsSim', '_FLAG_NAME']
key2 = ['config', 'paramsOptim', '_FLAG_NAME']
all_collect = OptBatch.collect_res(keys = [key1, key2], allPrefix = 'res', folderName = all_name + '/' + all_name)
names = list(all_collect.keys())
print(names)


### Look at one list_res in particular
name_test = '15pT30_GP'
listRes_test = all_collect[name_test]
dicoRes_test = OptBatch.process_list_res(listRes_test, printing = True)
print(list(dicoRes_test.keys()))

minlogFunc = (lambda x: np.log10(1-x))
dico_plot = {'legend': name_test + ' observed fom'}
ut.plot_from_stats(dicoRes_test['evol_fom'], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)
dico_plot = {'legend': name_test + ' ideal fom'}
ut.plot_from_stats(dicoRes_test['evol_ideal_fom]'], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)
dico_plot = {'legend': name_test + ' ideal fidelity'}
ut.plot_from_stats(dicoRes_test['evol_ideal_fidelity]'], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)
dico_plot = {'legend': name_test + ' optim control'}
ut.plot_from_stats(dicoRes_test['opt_control'], component = 'avgminmax', dico_plot = dico_plot)

print(dicoRes_test['best_fom'][0:3])
print(dicoRes_test['test_fom'][0:3])
print(dicoRes_test['test_fidelity'][0:3])



### Compare several list_res
names_config = ['15pT30_NM', '15pT30_DE', '15pT30_GP']
list_resObjects = [OptBatch.process_list_res(all_collect[n], printing = True) for n in names_config]


lookat = 'evol_fom'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fom]'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fidelity]'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)


lookat = 'opt_control'
dico_plot = {'legend': names_config, 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects], component = 'avgminmax', dico_plot = dico_plot)






### Compare several list_res
names_config_proj10 = ['15pT30_proj10_NM', '15pT30_proj10_DE', '15pT30_proj10_GP']
list_resObjects_proj10 = [OptBatch.process_list_res(all_collect[n], printing = True) for n in names_config_proj10]


lookat = 'evol_fom'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom'}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_fom'
dico_plot = {'legend': names_config, 'xlim':[0, 10000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fom]'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat + 'zoom'}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)

lookat = 'evol_ideal_fom]'
dico_plot = {'legend': names_config, 'xlim':[0, 10000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot, func_wrap = np.log10)


lookat = 'evol_ideal_fidelity]'
dico_plot = {'legend': names_config, 'xlim':[0, 1000], 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot, func_wrap = minlogFunc)


lookat = 'opt_control'
dico_plot = {'legend': names_config, 'suptitle': lookat}
ut.plot_from_list_stats([r[lookat] for r in list_resObjects_proj10], component = 'avgminmax', dico_plot = dico_plot)















