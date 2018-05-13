#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
break
import sys
sys.path.append('../../../../')
from  QuantumSimulation.Simulation.Spin.ControlledSpinOptimBatch import ControlledSpinOptimBatch as OptBatch
from  QuantumSimulation.Utility import Helper as ut
import numpy as np
from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pf
import QuantumSimulation.ToyModels.ControlledSpin as cs
from functools import partial
import importlib as ilib
ilib.reload(ut)



pathRes = "/Users/frederic/Desktop/Res/"
saveRes = False

minlogFunc = (lambda x: np.log10(1-x))


#==============================================================================
# Get all the results needed
#==============================================================================
all_names = ['Batch0_NM', 'Batch0_DE', 'Batch0_GP', 'Batch0noise_NM', 'Batch0noise_DE',
             'Batch0noise_GP', 'Batch0proj10_NM', 'Batch0proj10_DE', 'Batch0proj10_GP',
             'Batch0proj100_NM', 'Batch0proj100_DE', 'Batch0proj100_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = pathRes + name + '/' + name) 
            for name in all_names]
             
all_resObjects = {name:OptBatch.process_list_res(all_res[n], printing = True) 
                    for n, name in enumerate(all_names)}
             

study = []
study.append(['Batch0_NM','Batch0_DE', 'Batch0_GP'])
study.append(['Batch0noise_NM', 'Batch0noise_DE','Batch0noise_GP'])
study.append(['Batch0proj10_NM', 'Batch0proj10_DE', 'Batch0proj10_GP'])

names_study = ['ideal', 'noise experimental', 'noise measurement']
names_plot = ['NM', 'DE', 'GP']


# inset dicos                    
dico_final_error_inset = {'legend': names_plot, 'xaxis':[0,4]}
d_fom_zoom = {'legend': names_plot, 'xlim':[0, 1000]}
d_fom_nl = {'suptitle': 'Learning - observed FoM', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(FoM)$"}
d_fom_r_nl = {'suptitle': 'Learning - real FoM', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(FoM)$"}
d_fid_nl = {'suptitle': 'Real fidelity', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(1-F)$"}
d_opt_control ={'suptitle': 'Optimal control found', 'xlim':[0, 1.5],'legend': names_plot, 'ylabel':r"$f(t)$", 'xlabel':r"$t$"}


#==============================================================================
# SETUP 1 T Normal
#==============================================================================
ilib.reload(ut)
study_nb = 2
names_tmp = study[study_nb]
res_tmp = [all_resObjects[n] for n in names_tmp]




#ut.plot_from_list_stats([r['evol_fom'] for r in res_tmp], component = 'avgminmax',
#                        dico_plot = dico_plot_fom_zoom, func_wrap = np.log10)

look_at = 'evol_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_fom_nl, func_wrap = np.log10, 
                        component_inset = 'avgminmax', dico_inset = d_fom_zoom)


look_at = 'evol_ideal_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_fom_r_nl, func_wrap = np.log10, 
                        component_inset = 'avgminmax', dico_inset = d_fom_zoom)


look_at = 'evol_ideal_fidelity'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_fid_nl, func_wrap = minlogFunc, 
                        component_inset = 'avgminmax', dico_inset = d_fom_zoom)

look_at = 'opt_control'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_opt_control)

ilib.reload(ut)
look_at = 'evol_ideal_fom'
names_tmp = all_names
res_tmp = [all_resObjects[n] for n in names_tmp]
col = ['b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g']
shp = ['p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v']
tick = np.arange(2, 1 + len(names_tmp),3)
tick_label = ['no noise', 'noise gaussian','proj 10', 'proj 100']
d_fom_nl = {'suptitle': 'Optimal FoM', 'ylabel':r"$log_{10}(FoM)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fom_nl,func_wrap = np.log10)

look_at = 'evol_ideal_fidelity'
names_tmp = all_names
res_tmp = [all_resObjects[n] for n in names_tmp]
col = ['b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g']
shp = ['p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v']
tick = np.arange(2, 1 + len(names_tmp),3)
tick_label = ['no noise', 'noise gaussian','proj 10', 'proj 100']
d_fom_nl = {'suptitle': 'Optimal fidelity', 'ylabel':r"$log_{10}(1-F)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fom_nl,func_wrap = minlogFunc)







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


