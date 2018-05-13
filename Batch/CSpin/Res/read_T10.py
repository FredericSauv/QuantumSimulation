#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
break
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
# SETUP T 30
#==============================================================================


### Get a lot of res
all_name = 'Batch0_T10'
key1 = ['config', 'paramsSim', '_FLAG_NAME']
key2 = ['config', 'paramsOptim', '_FLAG_NAME']
all_collect = OptBatch.collect_res(keys = [key1, key2], allPrefix = 'res', folderName = pathRes + all_name + '/' + all_name)
names = list(all_collect.keys())
print(names)


all_names = ['15pT10_NM', '15pT10_DE', '15pT10_GP', 
             '15pT10_noiseall_NM', '15pT10_noiseall_DE','15pT10_noiseall_GP', 
             '15pT10_noiseZ_NM', '15pT10_noiseZ_DE','15pT10_noiseZ_GP',              
             '15pT10_proj10_NM', '15pT10_proj10_DE', '15pT10_proj10_GP',
             '15pT10_proj100_NM', '15pT10_proj100_DE', '15pT10_proj100_GP']

             
all_resObjects = {name:OptBatch.process_list_res(all_collect[name], printing = True) 
                    for n, name in enumerate(all_names)}
             

study = []
study.append(['15pT10_NM', '15pT10_DE', '15pT10_GP'])
study.append(['15pT10_noiseall_NM', '15pT10_noiseall_DE','15pT10_noiseall_GP'])
study.append(['15pT10_proj10_NM', '15pT10_proj10_DE', '15pT10_proj10_GP'])
study.append(['15pT10_proj100_NM', '15pT10_proj100_DE', '15pT10_proj100_GP'])
study.append(['15pT10_noiseZ_NM', '15pT10_noiseZ_DE','15pT10_noiseZ_GP'])

names_study = ['ideal', 'noise experimental', 'noise measurement', 'noise measurement weak', 'noise Z, ']
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
study_nb = 0
names_tmp = study[study_nb]
res_tmp = [all_resObjects[n] for n in names_tmp]




#ut.plot_from_list_stats([r['evol_fom'] for r in res_tmp], component = 'avgminmax',
#                        dico_plot = dico_plot_fom_zoom, func_wrap = np.log10)

look_at = 'evol_fom'
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'avgminmax', 
                        dico_plot = d_fom_nl, func_wrap = np.log10, 
                        component_inset = 'avgminmax', dico_inset = d_fom_zoom)
plt.savefig('foo.png', bbox_inches='tight')


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
col = np.tile(['b', 'r', 'g'], 10)
shp = np.tile(['p', 's', 'v'], 10)
tick = np.arange(2, 1 + len(names_tmp),3)
tick_label = ['no noise', 'noise gaussian', 'noise Z','proj 10', 'proj 100']
d_fom_nl = {'suptitle': 'Optimal FoM', 'ylabel':r"$log_{10}(FoM)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fom_nl,func_wrap = np.log10)

look_at = 'evol_ideal_fidelity'
names_tmp = all_names
res_tmp = [all_resObjects[n] for n in names_tmp]
d_fom_nl = {'suptitle': 'Optimal fidelity', 'ylabel':r"$log_{10}(1-F)$", 'colors':col, 'xlim':[0,1 + len(names_tmp)], 
            'shapes':shp, 'legend': names_plot, 'xticks': tick,'xtick_label': tick_label}
ut.plot_from_list_stats([r[look_at] for r in res_tmp], component = 'finalerror', 
                        dico_plot = d_fom_nl,func_wrap = minlogFunc)













