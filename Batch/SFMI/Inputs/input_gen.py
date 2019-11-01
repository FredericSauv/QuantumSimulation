#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:22:43 2019

@author: fred
"""
import itertools as it


# generate string for simulations
list_nbparams = [(5,'5p'), (10,'10p'),(15,'15p'),(20,'20p')]
list_noise =[('fid:neg','fid'), ('varN','varN'), ('varN10','varN10'), ('varN100','varN100'), ('varN1000','varN1000'), ('varN10000','varN10000')]
list_time = [(2.568255867176727,'05T'),(5.136511734353454,'1T'),(6.163814081224145,'12T'),(10.273023468706908,'2T')]
list_time = [(2.568255867176727,'05T')]


placeholder = "'ctl_shortcut':'owbds01_intcub{0}','fom':[{2},'f2t2:neg'],'T':{4},'_FLAG':'{5}{3}_{1}'"

combins = it.product(list_nbparams,list_noise,list_time)
combins_string = ["{"+ placeholder.format(c[0][0],c[0][1],c[1][0],c[1][1],c[2][0],c[2][1]) + "}" for c in combins]
final = " ".join(combins_string)


{'type_optim':'NM','init_obj':None,'maxfev':2000,'max_time':83000,'maxiter':2000,'_FLAG':'NM'} {'type_optim':'DE','maxiter':50,'popsize':10,'init_obj':None,'_FLAG':'DE'} {'type_optim':'BO','save_extra_bo':True,'nb_anchors':5,'type_acq':'LCB','type_lik':None,'nb_init':150,'nb_iter':500,'num_cores':4,'optimize_restarts':1,'acquisition_weight_lindec':True,'acq_weight':5,'exploitation_steps':250,'_FLAG':'BO_LCBlindec','hp_constrains':{'Mat52.lengthscale':[0.01,25.],'Mat52.variance':[0.1,25.0],'white.variance':[1e-09,10.]},'hp_update_interval':10} {'type_optim':'BO','save_extra_bo':True,'kernel_type':'RBF','nb_anchors':5,'type_acq':'LCB','max_time':83000,'type_lik':None,'nb_init':150,'nb_iter':600,'num_cores':4,'optimize_restarts':1,'acquisition_weight_lindec':True,'acq_weight':5,'exploitation_steps':250,'_FLAG':'BO_LCBlindec','hp_constrains':{'rbf.lengthscale':[0.01,25.],'rbf.variance':[0.1,25.0],'white.variance':[1e-09,10.]},'hp_update_interval':10} {'type_optim':'RANDOM','maxiter':10000,'_FLAG':'RANDOM'} {'type_optim':'LBFGSB','maxiter':1500,'_FLAG':'LBFGSB'} {'type_optim':'SPSA','maxfev':200000,'max_time':83000,'_FLAG':'SPSA10K_01','maxiter':10000,'config_spsa':0,'b':0.1}