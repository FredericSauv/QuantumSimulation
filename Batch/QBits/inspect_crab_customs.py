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

# REMARKS
# A LOT OF DE FAILED (LACK OF TIME) ==> reduce nbev to max 50000
# 3T: NM equiv to BO // Is it the same for pcw5 NO maybe because params nb increase=>less local minimas
# // Use Exploit strategy sooner
#
#



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
name_batch = 'crab_customs'
name_folder = name_batch + '/' + name_batch
one_res = learner.eval_from_onefile(name_folder +'/res120.txt')
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

t_not_pct = lambda n: not(can_find(n, '_25pctE_'))

t_BO = lambda n: can_find(n,'BO')
t_BO_b = lambda n: can_find(n,'BO') and t_not_pct(n, '_25pctE_')
t_NM = lambda n: can_find(n,'NM')
t_DE = lambda n: can_find(n,'DE')

t_05T = lambda n: can_find(n, '05T')
t_1T = lambda n: can_find(n, '1T')
t_2T = lambda n: can_find(n, '2T')
t_3T = lambda n: can_find(n, '3T')

t_CRAB0 = lambda n: can_find(n, 'CRAB0')
t_CRAB1 = lambda n: can_find(n, 'CRAB1')
t_CRAB2 = lambda n: can_find(n, 'CRAB2')
t_CRAB3 = lambda n: can_find(n, 'CRAB3')

t_2P = lambda n: can_find(n,'_2P_')
t_4P = lambda n: can_find(n,'_4P_')
t_6P = lambda n: can_find(n,'_6P_')
t_8P = lambda n: can_find(n,'_8P_')


#==============================================================================
# *** PLOT  DATA *** COMAPRE CONFIGURATIONS
#==============================================================================
### Comp specific
list_test = [t_05T, t_8P, t_CRAB0, t_not_pct]
names_tmp = custom_filter(names_configurations, *list_test)
names_tmp.sort()
list_tmp = [summary[n] for n in names_tmp]
raw_res_tmp = [summary[n] for n in names_tmp]

look_at = 'test_fom'
d_tmp = copy.copy(d_final_log10fom)
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

look_at = 'test_f2t2'
d_tmp = copy.copy(d_final)
d_tmp.update({'legend':names_tmp, 'ylim':[0,1]})
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)

d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'xlim':[0,200], 'ylim':[-0.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)
d_tmp = copy.copy(d_evol_log10fom)
d_tmp.update({'legend':names_tmp, 'ylim':[-0.5,0]})
look_at = 'ev_fom'
ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp)


## LOOKING at a function
names_interesting = ['BO1000EI_CRAB0_8P_05T', 'NM1m1_CRAB0_8P_3T']
raw_res_filt = rawres['BO1000EI_CRAB0_4P_3T']
best_run_fun = [ut.extract_from_nested(r, ['fun']) for r in raw_res_filt]
arg_tla = np.argmin(best_run_fun)
best_run_tla = raw_res_filt[arg_tla]
fun = pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(best_run_tla, ['func']))
T_tla = ut.extract_from_nested(best_run_tla, ['config', 'testing_dico', 'T'])
tt_tla = np.linspace(-0.1, T_tla + 0.1, 10000)
fun.plot_function(tt_tla)
plt.savefig('test.pdf')

testing_dico = best_run_tla['config']['testing_dico']
params_opt = best_run_tla['params']
names_fom = testing_dico['fom']
model_tmp = bh1d.BH1D(**testing_dico)
model_tmp(params_opt)


def get_main_info(res, field = 'test_fom'):
    data = res[field]
    n = str(int(data['n']))
    av = '%.4f' % data['avg']
    m = '%.4f' % data['min']
    M = '%.4f' % data['max']

    return ",".join([n, m+"/"+av+"/"+M])


def plot_one_best_func(list_res, path = ['fun'], criterion = np.argmin):
    run_fun = [ut.extract_from_nested(r, path) for r in list_res]
    best_arg = criterion(run_fun)
    best_run = list_res[best_arg]
    best_func = pFunc_base.pFunc_base.build_pfunc(ut.extract_from_nested(best_run, ['func']))
    best_T = ut.extract_from_nested(best_run, ['config', 'testing_dico', 'T'])
    best_tt = np.linspace(-0.1, best_T + 0.1, 10000)
    best_func.plot_function(best_tt)

def save_plot_best_func(list_name, save_fig):
    if(ut.is_str(save_fig)):
        plt.close()
        for n in list_name:
            plot_one_best_func(rawres[n])
        plt.legend(list_name)
        plt.savefig(save_fig)
        plt.close()





#==============================================================================
# *** PLOT  SAVING *** 
#==============================================================================
save = True
path_save = 'fig_crab/'
list_test = [t_05T, t_2P, t_CRAB0, t_not_pct]
l_param = ['1H', '2H', '3H', '4H']


map_name_filter = {'05T':t_05T,'1T':t_1T,'2T':t_2T, '3T':t_3T, '1H':t_2P, '2H':t_4P,'3H':t_6P,'4H':t_8P,
                   'CRAB0':t_CRAB0, 'CRAB1':t_CRAB1,'CRAB2':t_CRAB2, 'CRAB3':t_CRAB3}

#0.5T
name_0 = '05T'
name_2 = 'CRAB0'
list_test[0] = map_name_filter[name_0]
list_test[2] = map_name_filter[name_2]

for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_tmp = [summary[n] for n in names_tmp]
    infos = [get_main_info(l) for l in list_tmp]
    legend_infos = [nam+":"+infos[n] for n, nam in enumerate(names_tmp)]

    ylim_tmp_fom = [-0.5, 0]
    x_lim_tmp_fom = [0, 10]

    look_at = 'test_fom'
    d_tmp = copy.copy(d_final_log10fom)
    d_tmp.update({'legend':legend_infos, 'ylim':ylim_tmp_fom, 'xlim':x_lim_tmp_fom, 'xtick_label':[]})
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'final_fom.pdf'])
    else:
        name_save = None
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

    
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'xlim':[0,300], 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom_zoom.pdf'])
    else:
        name_save = None
        
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
    
    
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom.pdf'])
    else:
        name_save = None
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
    
    
#rawrestmp = [rawres[n] for n in names_tmp]
#Nb_map = [[r['config']['_RES_NAME'] for r in l_r] for l_r in rawrestmp]
#SEED_map = [[r['config']['_RDM_SEED'] for r in l_r] for l_r in rawrestmp]
#Om_used = [[eval(r['func'].split('Om')[-2][2:-2]) for r in l_r] for l_r in rawrestmp]
#
#seed_list = np.unique(SEED_map[0])
#
#one_seed = seed_list[14]
#mask = [np.array(ss) == one_seed for ss in SEED_map]
#om_found = [np.array(Om_used[n])[m]  for n,m in enumerate(mask)]


#for plotting reason  has been splitted
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_test[1] = map_name_filter[name_1]
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'func.pdf'])
    else:
        name_save = None
    
    save_plot_best_func(names_tmp, name_save)

save = True
name_0 = '1T'
list_test[0] = map_name_filter[name_0]
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_tmp = [summary[n] for n in names_tmp]
    infos = [get_main_info(l) for l in list_tmp]
    legend_infos = [nam+":"+infos[n] for n, nam in enumerate(names_tmp)]

    ylim_tmp_fom = [-1, 0]
    x_lim_tmp_fom = [0, 10]

    look_at = 'test_fom'
    d_tmp = copy.copy(d_final_log10fom)
    d_tmp.update({'legend':legend_infos, 'ylim':ylim_tmp_fom, 'xlim':x_lim_tmp_fom, 'xtick_label':[]})
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'final_fom.pdf'])
    else:
        name_save = None
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

    
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'xlim':[0,300], 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom_zoom.pdf'])
    else:
        name_save = None
        
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
    
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom.pdf'])
    else:
        name_save = None
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

#for plotting reason  has been splitted
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_test[1] = map_name_filter[name_1]
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'func.pdf'])
    else:
        name_save = None
    
    save_plot_best_func(names_tmp, name_save)

save = True
name_0 = '2T'
list_test[0] = map_name_filter[name_0]
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_tmp = [summary[n] for n in names_tmp]
    infos = [get_main_info(l) for l in list_tmp]
    legend_infos = [nam+":"+infos[n] for n, nam in enumerate(names_tmp)]

    ylim_tmp_fom = [-3.5, 0]
    x_lim_tmp_fom = [0, 10]

    look_at = 'test_fom'
    d_tmp = copy.copy(d_final_log10fom)
    d_tmp.update({'legend':legend_infos, 'ylim':ylim_tmp_fom, 'xlim':x_lim_tmp_fom, 'xtick_label':[]})
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'final_fom.pdf'])
    else:
        name_save = None
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

    
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'xlim':[0,300], 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom_zoom.pdf'])
    else:
        name_save = None
        
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
    
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom.pdf'])
    else:
        name_save = None
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

#for plotting reason  has been splitted
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_test[1] = map_name_filter[name_1]
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'func.pdf'])
    else:
        name_save = None
    
    save_plot_best_func(names_tmp, name_save)

save = True
name_0 = '3T'
list_test[0] = map_name_filter[name_0]
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_tmp = [summary[n] for n in names_tmp]
    infos = [get_main_info(l) for l in list_tmp]
    legend_infos = [nam+":"+infos[n] for n, nam in enumerate(names_tmp)]

    ylim_tmp_fom = [-4, 0]
    x_lim_tmp_fom = [0, 10]
    look_at = 'test_fom'
    d_tmp = copy.copy(d_final_log10fom)
    d_tmp.update({'legend':legend_infos, 'ylim':ylim_tmp_fom, 'xlim':x_lim_tmp_fom, 'xtick_label':[]})
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'final_fom.pdf'])
    else:
        name_save = None
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)

    
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'xlim':[0,300], 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom_zoom.pdf'])
    else:
        name_save = None
        
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
    
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'evol_fom.pdf'])
    else:
        name_save = None
    d_tmp = copy.copy(d_evol_log10fom)
    d_tmp.update({'legend':names_tmp, 'ylim':ylim_tmp_fom})
    look_at = 'ev_fom'
    ut.plot_from_list_stats([r[look_at] for r in list_tmp], dico_main = d_tmp, save_fig = name_save)
#for plotting reason  has been splitted
for name_1 in l_param:
    list_test[1] = map_name_filter[name_1]
    names_tmp = custom_filter(names_configurations, *list_test)
    names_tmp.sort()
    list_test[1] = map_name_filter[name_1]
    if(save):
        name_save = "_".join([path_save, name_0, name_1, name_2,'func.pdf'])
    else:
        name_save = None
    
    save_plot_best_func(names_tmp, name_save)



rawrestmp = [rawres[n] for n in names_tmp]


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



#==============================================================================
# *** PLOT  DATA *** COMAPRE CONFIGURATIONS
#==============================================================================
res_DE3T = rawres['DE10_3T']
opt_params_DE3T = [r['params'] for r in res_DE3T]








res_tla = learner.find_best_run_in_dico_simul(rawres, ['test_fom', 1], test = 'max', filt = test1T)

dico_test_tla = res_tla['config']['testing_dico']
model_tla = bh1d.BH1D(**dico_test_tla)
test_tla = model_tla(res_tla['params']) 
T_tla = dico_test_tla['T']
fun_tla = model_tla.control_fun
x_tla = np.linspace(-0.1, T_tla+0.1, 500)
fun_tla.plot_function(x_tla)
plt.plot(x_tla, x_tla/T_tla)


res_tla = rawres['BO1000EI_2T'][3]
dico_test_tla = res_tla['config']['testing_dico']
model_tla = bh1d.BH1D(**dico_test_tla)
test_tla = model_tla(res_tla['params']) 
T_tla = dico_test_tla['T']
fun_tla = model_tla.control_fun
x_tla = np.linspace(-0.1, T_tla+0.1, 500)
fun_tla.plot_function(x_tla)
plt.plot(x_tla, x_tla/T_tla)


