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

pathRes = "/home/fred/OneDrive/Quantum/Projects/Python/Dynamic1.3/ResBatch/Res/"
# pathRes = "/Users/frederic/Desktop/Res/"
saveRes = False

minlogFunc = (lambda x: np.log10(1-x))


#==============================================================================
# Get all the results needed
#==============================================================================
all_names = ['Batch0_NM', 'Batch0_DE', 'Batch0_GP', 
             'Batch0noise_NM', 'Batch0noise_DE','Batch0noise_GP', 
             'Batch0proj10_NM', 'Batch0proj10_DE', 'Batch0proj10_GP',
             'Batch0proj100_NM', 'Batch0proj100_DE', 'Batch0proj100_GP']

all_res = [OptBatch.read_res(allPrefix = 'res', folderName = pathRes + name + '/' + name) 
            for name in all_names]             
all_resObjects = {name:OptBatch.process_list_res(all_res[n], printing = True) 
                    for n, name in enumerate(all_names)}
             



#==============================================================================
# Compare all final results
#==============================================================================
ilib.reload(ut)
names_tmp = all_names
names_plot = ['NM', 'DE', 'BO']
res_tmp = [all_resObjects[n] for n in names_tmp]
col = ['b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g']
shp = ['p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v', 'p', 's', 'v']
tick = np.arange(2, 1 + len(names_tmp),3)
tick_label = ['no noise', 'Gaussian noise','10 meas.', '100 meas.']
d_fom_error = {'suptitle': 'Optimal FoM, T=1.5', 'ylabel':r"$log_{10}(FoM)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}
d_fid_error = {'suptitle': 'Optimal fidelity, T = 1.5', 'ylabel':r"$log_{10}(1-F)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}



look_at = 'evol_ideal_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fom_error,func_wrap = np.log10)

look_at = 'evol_ideal_fidelity'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fid_error,func_wrap = minlogFunc)


#==============================================================================
# Look at different studies
#==============================================================================
ilib.reload(ut)
study = []
study.append(['Batch0_NM','Batch0_DE', 'Batch0_GP'])
study.append(['Batch0noise_NM', 'Batch0noise_DE','Batch0noise_GP'])
study.append(['Batch0proj10_NM', 'Batch0proj10_DE', 'Batch0proj10_GP'])
study.append(['Batch0proj100_NM', 'Batch0proj100_DE', 'Batch0proj100_GP'])


names_study = ['No noise', 'Gausian noise', 'Noise 10 measurements', 'Noise 100 measurements']
inset_fom =[[0.4, 0.4, 0.46, 0.46], [0.5, 0.5, 0.36, 0.36], [0.5, 0.5, 0.36, 0.36], [0.5, 0.5, 0.36, 0.36]]
inset_size_fom = [8,8,8,8]

inset_fid =[[0.5, 0.5, 0.35, 0.35], [0.5, 0.5, 0.36, 0.36], [0.5, 0.2, 0.4, 0.4], [0.5, 0.2, 0.36, 0.3]]
inset_size_fid = [8,8,8,8]

study_nb = 3
# inset dicos                    
d_fom_zoom = {'legend': names_plot, 'xlim':[0, 1000], 'inset_size':inset_size_fom[study_nb], 'inset': inset_fom[study_nb]}
d_fom_nl = {'suptitle': names_study[study_nb]+', observed FoM', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(FoM)$"}

d_fom_r_nl = {'suptitle': names_study[study_nb] +', real FoM', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(FoM)$"}

d_fid_zoom = {'legend': names_plot, 'xlim':[0, 1000], 'inset_size':inset_size_fid[study_nb], 'inset': inset_fid[study_nb]}
d_fid_nl = {'suptitle': names_study[study_nb] + ', real fidelity', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(1-F)$"}

d_opt_control ={'suptitle': names_study[study_nb] +', optimal control function', 'xlim':[0, 1],'legend': names_plot, 'ylabel':r"$f(t)$", 'xlabel':r"$t$"}


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
                        component_inset = 'avgminmax', dico_inset = d_fid_zoom)

## Plot Optimal Control
look_at = 'opt_control'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_opt_control)







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


