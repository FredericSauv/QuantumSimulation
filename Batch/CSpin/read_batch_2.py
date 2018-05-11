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
import pdb
import numpy as np
import matplotlib.pylab as plt
import importlib as ilib
from QuantumSimulation.Utility.Optim.ParametrizedFunctionFactory import ParametrizedFunctionFactory as pf
import QuantumSimulation.ToyModels.ControlledSpin as cs
from functools import partial

def extract_fom_nev(res):
    return [np.array(run['optim']['fev_fom']) for run in res]

def extract_res_test(res):
    return [run['best_fom'] for run in res]     

def extract_res_test_index(res, index = 0):
    return [run['testing']['fun'][index] for run in res]

def extract_optim_func(res, index = 0):
    repr_func = [run['optim']['fun'][index] for run in res]
    return repr_func


def plot_from_stats(stats, component = 'avg', ax_plt = None, dico_plot = {}):
    """ from a stats create some plots
    """
    # stats = [[index, avg, mini, maxi, std, avg_pstd, avg_mstd],...]
    
    indices = stats['index']
    if(ax_plt is None):
        fig, ax_plt = plt.subplots()
    
    legend = dico_plot.get('legend')

    if(component  == 'minmax'):
        ymin = stats['min']
        ymax = stats['max']
        if(dico_plot.get('color')):
            color = dico_plot.get('color')
        else:
            color = 'b'
        ax_plt.plot(indices, ymin, color = color, label=legend)
        ax_plt.plot(indices, ymax, color = color)
        ax_plt.fill_between(indices, ymin, ymax,alpha=0.2, color = color)
    
    if(component  == 'avgminmax'):
        ymin = stats['min']
        ymax = stats['max']
        yavg = stats['avg']
        if(dico_plot.get('color')):
            color = dico_plot.get('color')
        else:
            color = 'b'
        ax_plt.plot(indices, ymin, color = color, label=legend)
        ax_plt.plot(indices, ymax, color = color)
        ax_plt.fill_between(indices, ymin, ymax,alpha=0.2, color = color)
        ax_plt.plot(indices, yavg, dashes=[6, 2], color = color)
        
    elif(component == 'pm1sd'):
        m = stats['avg_mstd']
        p = stats['avg_pstd']
        if(dico_plot.get('color')):
            color = dico_plot.get('color')
        else:
            color = 'b'
        ax_plt.plot(indices, m, color = color, label=legend)
        ax_plt.plot(indices, p, color = color)
        ax_plt.fill_between(indices, m, p, color = color, alpha = 0.2)
        res_opt_func
    else:
        comp = ut.splitString(component)
        for c in comp:
            ax_plt.plot(indices, stats[c], label = str(c))
       
        if(dico_plot.get('legend', False)):
            ax_plt.legend()
        if(dico_plot.get('ylim', False)):
            ylim = dico_plot.get('ylim')
            ax_plt.set_ylim(ylim[0], ylim[1])
        if(dico_plot.get('xlim', False)):
            xlim = dico_plot.get('xlim')
            ax_plt.set_xlim(xlim[0], xlim[1])
    
    if(legend is not None):
        ax_plt.legend()
    
def plot_from_list_stats(list_stats, component = 'avg', dico_plot = {}):
    # stats = [[index, avg, mini, maxi, std, avg_pstd, avg_mstd],...]
    listColors = ['orange', 'g', 'r', 'b']
    fig, ax_plt = plt.subplots()
    for stats in list_stats:
        col_tmp = listColors.pop()
        plot_from_stats(stats, component, ax_plt = ax_plt, dico_plot = {'color':col_tmp})
    
    if(dico_plot.get('legend')):
        ax_plt.legend()
    if(dico_plot.get('ylim')):
        ylim = dico_plot.get('ylim')
        ax_plt.set_ylim(ylim[0], ylim[1])
    if(dico_plot.get('xlim')):
        xlim = dico_plot.get('xlim')
        ax_plt.set_xlim(xlim[0], xlim[1])
    


#==============================================================================
# SETUP T 30
#==============================================================================

all_name = 'Batch0_T30b'
key1 = ['config', 'paramsSim', '_FLAG_NAME']
key2 = ['config', 'paramsOptim', '_FLAG_NAME']
all_collect = OptBatch.collect_res(keys = [key1, key2], allPrefix = 'res', folderName = all_name + '/' + all_name)
names = list(all_collect.keys())

# Res 
path_optim_fev_fom = ['optim', 'fev_fom']
path_optim_func_fom = ['optim', 'func_fom']
path_test_fom = ['best_fom']
path_test_fun0 = ['testing', 'fun', 0]
path_test_fun1 = ['testing', 'fun', 1]
path_test_namefom = ['config', 'paramsTesting', 'fom_name']
path_test_t = ['config', 'paramsTesting', 'T']

name_tmp = names[11]

res_tmp = all_collect[name_tmp]
res_optim_fev_fom = [np.array(ut.extract_from_nested(run, path_optim_fev_fom)) for run in res_tmp]
res_optim_func_fom = [ut.extract_from_nested(run, path_optim_func_fom) for run in res_tmp]
res_test_fom = [np.array(ut.extract_from_nested(run, path_test_fom)) for run in res_tmp]
res_test_fun0 = [np.array(ut.extract_from_nested(run, path_test_fun0)) for run in res_tmp]
res_test_fun1 = [np.array(ut.extract_from_nested(run, path_test_fun1)) for run in res_tmp]
res_test_namefom = [ut.extract_from_nested(run, path_test_namefom) for run in res_tmp]
res_T = [ut.extract_from_nested(run, path_test_t) for run in res_tmp]
T = res_T[0]
array_T = np.arange(0, T, T/100)


dico_plot = {'legend':name_tmp}
res_optim_fev_fom_stats = ut.merge_and_stats_TS(res_optim_fev_fom)
plot_from_stats(res_optim_fev_fom_stats, 'avgminmax', None, dico_plot)


## New get real learning
#c_test = pf.Repr2Fun(res_optim_func_fom[0][-1][0])
#p_repr = ut.extract_from_nested(res_tmp[0], ['config', 'paramsTesting'])
#
#sim_test = cs.ControlledSpin(setup = p_repr.get('setup'), state_init = p_repr.get('init_state_name')
#                ,controlFun = c_test,  state_target = p_repr.get('target_state_name')
#                ,T = p_repr.get('T'), dt = p_repr.get('dt'), noise = p_repr.get('noise'))
#
#sim_test.Simulate(method = p_repr.get('method_simul'), fom = p_repr.get('fom_name'))   
#


#### Res -> 10 runs


    
def runSimul(dicoArgs, control):
    setup = dicoArgs['setup']
    init = dicoArgs['init_state_name']
    final = dicoArgs['target_state_name']
    T = dicoArgs['T']
    dt = dicoArgs['dt']
    noise = dicoArgs['noise']
    method = dicoArgs['method_simul']
    fom = dicoArgs['fom_name']
    if(ut.is_str(control)):
        control = pf.Repr2Fun(control)
    
    sim_tmp = cs.ControlledSpin(setup = setup, state_init = init ,controlFun = control,  
                state_target = final ,T = T, dt = dt, noise = noise)
    res = sim_tmp.Simulate(method = method, fom = fom)   
    return res

def ideal_learning_from_run(run):
    """ Compute the fom (under testing conditions) for the different functions 
    found along optimization
    """
    dico_testing = ut.extract_from_nested(res_tmp[0], ['config', 'paramsTesting'])
    func_fom = ut.extract_from_nested(run, ['optim', 'func_fom'])
    fev_fom = ut.extract_from_nested(run, ['optim', 'fev_fom'])
    nb_fom = len(dico_testing['fom_name'])
    
    run = partial(runSimul, dicoArgs = dico_testing)
    tmp = np.array([[fev_fom[nb][0]] + run(control = item[0]) for nb, item in enumerate(func_fom)])
    res = [tmp[:,[0, n + 1]] for n in range(nb_fom)]
    
    return res
    
def ideal_learning_from_res(simul):
    tmp = [ideal_learning_from_run(run) for run in simul]
    nb_run = len(simul)
    nb_res = len(tmp[0])
    res = [[tmp[r][n] for r in range(nb_run)] for n in range(nb_res)]
    return res



real_fom = ideal_learning_from_res(res_tmp)
real_fom_0  = real_fom[0]
real_fom_1 = real_fom[1]

dico_plot = {'legend':'fom0'}
real_fom_0_stats = ut.merge_and_stats_TS(real_fom_0)
plot_from_stats(real_fom_0_stats, 'avgminmax', None, dico_plot)

dico_plot = {'legend':'fom1'}
real_fom_1_stats = ut.merge_and_stats_TS(real_fom_1)
plot_from_stats(real_fom_1_stats, 'avgminmax', None, dico_plot)





#Optimal functions
res_opt_func = [pf.Repr2Fun(run[-1][0]) for run in res_optim_func_fom]
res_opt_func_T = [np.c_[array_T, func(array_T)] for func in res_opt_func]
res_opt_func_stats = ut.merge_and_stats_TS(res_opt_func_T)
dico_plot = {'legend':'optimal pulse'}
plot_from_stats(res_opt_func_stats, 'avgminmax', None, dico_plot)





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


