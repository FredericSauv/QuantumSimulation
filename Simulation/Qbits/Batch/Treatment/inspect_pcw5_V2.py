#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:39:09 2018

@author: fred
"""

import sys
import pdb
import copy
import numpy as np
from functools import partial

sys.path.append("../../../")
from QuantumSimulation.Utility import Helper as ut
from QuantumSimulation.Utility.Optim import pFunc_zoo, pFunc_base, Learner, Batch 
from QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH as learner
from QuantumSimulation.ToyModels import BH1D as bh1d
import matplotlib.pylab as plt


### CUSTOM METHODS ARE USED BECAUSE OF A BUG IN RES FILE
#==============================================================================
# ***GETDATA*** Linear benchmark
#==============================================================================
name_batch = 'bch_linear'
name_folder = name_batch + '/' + name_batch
key_T = ['config', 'testing_dico', 'T']
bch_res = learner.collect_res([key_T], None, 'benchmark', name_folder)
bch_res_fom = {k: ut.extract_from_nested(r, [0,'test_fom']) for k, r in bch_res.items()}

#==============================================================================
# ***GETDATA***  
#==============================================================================
name_batch = 'pcw5_V2'
name_folder = name_batch + '/' + name_batch
one_res = learner.eval_from_onefile_bug(name_folder +'/res120.txt')
names_keys_res = list(one_res.keys())
ut.print_nested_keys(one_res, ' +')

key1 = ['config', 'optim_dico', '_FLAG_NAME']
key2 = ['config', 'model_dico', '_FLAG_NAME']

rawres = learner.collect_res([key1, key2], None, 'res', name_folder)
summary = learner.collect_and_process_res([key1, key2], None, 'res', name_folder, True, False)
names_configurations = list(summary.keys())
print('names configurations: ' + str(names_configurations) + '\n')
print('data/stats available: ' + str(summary[names_configurations[0]].keys()))



#==============================================================================
# ***PLOTTING ASSETS*** 
#==============================================================================
col_list = ['b', 'g', 'r', 'c', 'm', 'k','C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] #list of colors
shp_list = ['p', 's', 'v', 'D', 'P', '1', '2', '4','3', 'p', 's', 'v', 'D', 'P', '1', '2', '4','3', '5',] #list of points shape

# to update with 'legend': pivot1,'suptitle': 'aaaa', 'xlim' = [0, xlim_zoom]
d_evol = {'component':'avgminmax', 'xlabel':'nb evals', 'colors':col_list, 'shapes': shp_list}
d_evol_log10 = copy.copy(d_evol)
d_evol_log10.update({'func_wrap':np.log10})
d_evol_log10fom = copy.copy(d_evol_log10)
d_evol_log10fom.update({'ylabel':r"$log_{10}(FoM)$"})
minlogFunc = (lambda x: np.log10(1-x))
# to update with 'legend': pivot1,'suptitle': 'aaaa', 'xlim' = [0, xlim_zoom]
# 'xticks': tick,'xtick_label': tick_label
d_final = {'component':'finalerror', 'colors':col_list, 'shapes':shp_list}
d_final_log10 = copy.copy(d_final)
d_final_log10.update({'func_wrap':np.log10})
d_final_log10fom = copy.copy(d_final_log10)
d_final_log10fom.update({'ylabel':r"$log_{10}(FoM)$"})

#insets (x, y, dx, dy)
in_sm_bl, in_sm_bm =  [0.2, 0.2, 0.3, 0.3], [0.4, 0.2, 0.3, 0.3]
in_sm_ml, in_sm_mm = [0.2, 0.4, 0.3, 0.3], [0.4, 0.4, 0.3, 0.3]



#==============================================================================
# *** PLOT  DATA *** COMAPRE CONFIGURATIONS
#==============================================================================
def can_find(string, *args):
    res = True
    for a in args:
        res *= (string.find(a) != -1)
    return res

def custom_filter(list_to_filt, *args):
    def fiter(n):
        res = True
        for t in args:
            res *= t(n)
        return res
    return [l for l in list_to_filt if fiter(l)]

t_BO = lambda n: can_find(n,'BO')
t_NM = lambda n: can_find(n,'NM')
t_DE = lambda n: can_find(n,'DE')

t_500 = lambda n: can_find(n,'500EI')
t_1000 = lambda n: can_find(n,'1000EI')

t_05T = lambda n: can_find(n, '05T')
t_1T = lambda n: can_find(n, '1T')
t_2T = lambda n: can_find(n, '2T')
t_3T = lambda n: can_find(n, '3T')

t_good_250 = lambda n: can_find(n,'BO_250_50x100k_001j_latin_2T') or can_find(n,'oldB_250')
t_good_500 = lambda n: can_find(n,'BO_500_50x100k_001j_latin_2T') or can_find(n,'oldB_500')

t_BO_b = lambda n: can_find(n,'BO') and can_find(n,'1000')
t_DE_b = lambda n: can_find(n,'DE10')
t_NM_b = t_NM



### Comp specific
list_test = [t_3T, t_BO]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0.5,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,200], 'ylim':[-2,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-2,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)


### Comp 05T
t_05_b = lambda n: t_05T(n) and (t_BO_b(n) or t_DE_b(n) or t_NM_b(n))

list_test = [t_05_b]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0.5,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,500], 'ylim':[-0.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-0.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)


### Comp 1T
t_1_b = lambda n: t_1T(n) and (t_BO_b(n) or t_DE_b(n) or t_NM_b(n))

list_test = [t_1_b]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0.5,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,500], 'ylim':[-1.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-1.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)





### Comp 2T
t_2_b = lambda n: t_2T(n) and (t_BO_b(n) or t_DE_b(n) or t_NM_b(n))

list_test = [t_2_b]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0.5,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,500], 'ylim':[-3,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-3,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)



### Comp 3T
t_3_b = lambda n: t_3T(n) and (t_BO_b(n) or t_DE_b(n) or t_NM_b(n))

list_test = [t_3_b]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0.5,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,500], 'ylim':[-2,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-3,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)



