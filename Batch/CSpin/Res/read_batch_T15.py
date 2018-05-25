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
from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory_old import ParametrizedFunctionFactory as pf
import QuantumSimulation.ToyModels.ControlledSpin as cs
from functools import partial
import importlib as ilib
import copy as cp
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


#Topup with new simulations
name2 = 'Batch0_DE_testpopsize'
key1 = ['config', 'paramsSim', '_FLAG_NAME']
key2 = ['config', 'paramsOptim', '_FLAG_NAME']
collect2 = OptBatch.collect_res(keys = [key1, key2], allPrefix = 'res', folderName = pathRes + name2 + '/' + name2)
names2 = list(collect2.keys())
print(names2)


resObjects2 = {name:OptBatch.process_list_res(collect2[name], printing = True) 
                    for n, name in enumerate(names2)}


all_res = {**all_resObjects, **resObjects2}



#==============================================================================
# Compare all final results
#==============================================================================
ilib.reload(ut)
list_study = [['Batch0_NM', 'NoNoise_DE1', 'NoNoise_DE2','NoNoise_DE5','NoNoise_DE10', 'Batch0_DE', 'Batch0_GP'], 
             ['Batch0noise_NM', 'GaussianNoise_DE1', 'GaussianNoise_DE2', 'GaussianNoise_DE5', 'GaussianNoise_DE10', 'Batch0noise_DE', 'Batch0noise_GP'], 
             ['Batch0proj10_NM','Proj10_DE1', 'Proj10_DE2','Proj10_DE5','Proj10_DE10','Batch0proj10_DE', 'Batch0proj10_GP'],
             ['Batch0proj100_NM','Proj100_DE1', 'Proj100_DE2','Proj100_DE5','Proj100_DE10', 'Batch0proj100_DE', 'Batch0proj100_GP']]

# flattening
names_all_study = [n for l in list_study for n in l]

pivot1 = ['NM', 'DE1', 'DE2', 'DE5', 'DE10', 'DE15', 'BO']
pivot2 = ['No noise', 'Gausian noise', 'Noise 10 measurements', 'Noise 100 measurements']

res_tmp = [all_res[n] for n in names_all_study]
col_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
shp_list = ['p', 's', 'v', 'D', 'P', '1', '2', '4']

nb_per_bucket = 7
col = np.tile(col_list[:nb_per_bucket], 4)
shp = np.tile(shp_list[:nb_per_bucket], 4)
tick = np.arange(2, 1 + len(names_all_study), nb_per_bucket)
tick_label = ['no noise', 'Gaussian noise','10 meas.', '100 meas.']
d_fom_error = {'component':'finalerror', 'func_wrap':np.log10, 'suptitle': 'Optimal FoM, T=1.5'
               , 'ylabel':r"$log_{10}(FoM)$", 'colors':col, 'xlim':[0,1 + len(names_all_study)], 
            'shapes':shp, 'legend': pivot1, 'xticks': tick,'xtick_label': tick_label}

d_fid_error = {'component':'finalerror', 'func_wrap':minlogFunc, 'suptitle': 'Optimal fidelity, T = 1.5', 
               'ylabel':r"$log_{10}(1-F)$", 'colors':col, 'xlim':[0,1 + len(names_all_study)], 
            'shapes':shp, 'legend': pivot1, 'xticks': tick,'xtick_label': tick_label}



look_at = 'evol_ideal_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom_error)

look_at = 'evol_ideal_fidelity'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fid_error)


#==============================================================================
# Look at different studies
#==============================================================================
ilib.reload(ut)

xlim_zoom = 500

inset_fom =[[0.4, 0.4, 0.46, 0.46], [0.5, 0.5, 0.36, 0.36], [0.5, 0.5, 0.36, 0.36], [0.5, 0.5, 0.36, 0.36]]
inset_size_fom = [8,8,8,8]
inset_fid =[[0.5, 0.5, 0.35, 0.35], [0.5, 0.5, 0.36, 0.36], [0.5, 0.2, 0.4, 0.4], [0.5, 0.2, 0.36, 0.3]]
inset_size_fid = [8,8,8,8]

study_nb = 0
name_subset_study = pivot2[study_nb]
res_tmp = [all_res[n] for n in list_study[study_nb]]

# OBS FOM
d_fom = {'component':'avgminmax', 'func_wrap':np.log10, 'legend': pivot1,
         'suptitle': name_subset_study+', observed FoM', 'xlabel':'nb evals', 
         'ylabel':r"$log_{10}(FoM)$", 'colors':col_list, 'shapes': shp_list}

d_fom_zoom = cp.copy(d_fom)
d_fom_zoom['xlim'] = [0, xlim_zoom]

d_fom_zoom_inset = cp.copy(d_fom_zoom)
d_fom_zoom_inset['inset_size'] = inset_size_fom[study_nb]
d_fom_zoom_inset['inset']: inset_fom[study_nb]

# REAL FOM
d_fom_real = cp.copy(d_fom)
d_fom_real['suptitle'] = name_subset_study+', real FoM'
d_fom_zoom_real = cp.copy(d_fom)
d_fom_zoom_real['suptitle'] = name_subset_study+', real FoM'
d_fom_zoom_real = cp.copy(d_fom_zoom)


d_fom_r_nl = {'suptitle': names_study[study_nb] +', real FoM', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(FoM)$"}

d_fid_zoom = {'legend': names_plot, 'xlim':[0, 1000], 'inset_size':inset_size_fid[study_nb], 'inset': inset_fid[study_nb]}
d_fid_nl = {'suptitle': names_study[study_nb] + ', real fidelity', 'xlabel':'nb evals', 'ylabel':r"$log_{10}(1-F)$"}

d_opt_control ={'suptitle': names_study[study_nb] +', optimal control function', 'xlim':[0, 1],'legend': names_plot, 'ylabel':r"$f(t)$", 'xlabel':r"$t$"}





#ut.plot_from_list_stats([r['evol_fom'] for r in res_tmp], component = 'avgminmax',
#                        dico_plot = dico_plot_fom_zoom, func_wrap = np.log10)

look_at = 'evol_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom)

ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom_zoom)

ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom, dico_inset = d_fom_zoom_inset)




look_at = 'evol_ideal_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom)

ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom_zoom)

ut.plot_from_list_stats([r[look_at] for r in res_tmp], dico_main = d_fom, dico_inset = d_fom_zoom_inset)


look_at = 'evol_ideal_fidelity'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_fid_nl, func_wrap = minlogFunc, 
                        component_inset = 'avgminmax', dico_inset = d_fid_zoom)

## Plot Optimal Control
look_at = 'opt_control'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_opt_control)



