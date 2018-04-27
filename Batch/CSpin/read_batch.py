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

def extract_fom_nev(res):
    return [np.array(run['optim']['fev_fom']) for run in res]
     
def plot_from_stats(stats, component = 'avg', ax_plt = None, dico_plot = {}):
    # stats = [[index, avg, mini, maxi, std, avg_pstd, avg_mstd],...]
    
    indices = stats['index']
    if(ax_plt is None):
        fig, ax_plt = plt.subplots()


    if(component  == 'minmax'):
        ymin = stats['min']
        ymax = stats['max']
        if(dico_plot.get('color')):
            color = dico_plot.get('color')
        else:
            color = 'b'
        ax_plt.plot(indices, ymin, color = color)
        ax_plt.plot(indices, ymax, color = color)
        ax_plt.fill_between(indices, ymin, ymax,alpha=0.5, color = color)
        
    elif(component == 'pm1sd'):
        m = stats['avg_mstd']
        p = stats['avg_pstd']
        if(dico_plot.get('color')):
            color = dico_plot.get('color')
        else:
            color = 'b'
        ax_plt.plot(indices, m, color = color)
        ax_plt.plot(indices, p, color = color)
        ax_plt.fill_between(indices, m, p, color = color, alpha = '0.5')
        
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
    
    
def plot_from_list_stats(list_stats, component = 'avg', dico_plot = {}):
    # stats = [[index, avg, mini, maxi, std, avg_pstd, avg_mstd],...]
    fig, ax_plt = plt.subplots()
    for stats in list_stats:
        plot_from_stats(stats, component, ax_plt = ax_plt)
    
    if(dico_plot.get('legend')):
        ax_plt.legend()
    if(dico_plot.get('ylim')):
        ylim = dico_plot.get('ylim')
        ax_plt.set_ylim(ylim[0], ylim[1])
    if(dico_plot.get('xlim')):
        xlim = dico_plot.get('xlim')
        ax_plt.set_xlim(xlim[0], xlim[1])
    
#==============================================================================
# SETUP 1 T
#==============================================================================
all_name = ['Batch0_NM', 'Batch0_DE', 'Batch0_GP']
all_res = [OptBatch.read_res(allPrefix = 'res', folderName = name + '/' + name) for name in all_name]


all_fom_nev = [extract_fom_nev(res) for res in all_res]
all_stats = [ut.merge_and_stats_TS(extract_fom_nev(res)) for res in all_res]

ut.merge_and_stats_TS(all_fom_nev[1])

plot_from_list_stats(all_stats, 'avg', dico_plot = {'xlim':[0,100]})

# Res -> Run -> Config
#            -> Optim -> fev_fom
#            -> Testing
#


plot_from_stats(all_stats[2], 'minmax', dico_plot = {'legend':True, 'xlim': [0,100]})