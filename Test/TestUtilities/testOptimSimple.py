#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
import copy
#sys.path.append('../../../FloquetSimulation')
sys.path.append('../../')
sys.path.append('../Utility/BayesianOptimization-master/')
from bayes_opt import BayesianOptimization

import FloquetSimulation.Simulation.ToyModelOptim as customOptim
import importlib as ilib
import pdb 

#customOptim.ToyModelOptim.helpConfiguration()

#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
paramsSim = {'eta' : 0.2, 'setup' : 2, 'tgt' : 0.1, 'tgt_type' : 'sigmaX',
             'tgt_picture': 'interaction', 'nPPP': 37, 'nPeriodEff':1,
             '_skipMotionalError':True, 'decimalRounding': 5, 'maxPopLastMotional':0.01, 
             'maxDevSumProba':0.01, 'IterMax':1, 'weights_FOM':None}

paramsOptim ={'algo': 'GP', 'name_FOM': ['NegFidelity_HeatingM1', 'Fidelity', 'HeatingM','Heating', 'Fluence']
                , 'params_init': 'simple', 'nb_params' : 3, 'gp_bounds':'range_0.25_1.75_-3_3_-3_3', 'gp_maxiter':50, 
                'gp_kappa':'linear_10', 'gp_init':'simple_noHeat'}

paramsSimTest = paramsSim.copy()
paramsSimTest['nPeriodEff'] = 20
paramsSimTest['nPPP'] = 27


#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
ilib.reload(customOptim)
oc = customOptim.ToyModelOptim(paramsSim, paramsOptim, paramsSimTest)
res = oc.run()

#pdb.run('oc.run()')


sim = oc.simulator_testing
st = sim.state_store
pb = sim.fss.probaFromState(st)
st_tgt = sim.stateTGT
pb_tgt = sim.fss.probaFromState(st_tgt)
import numpy as np
import matplotlib.pyplot as plt
st_heat = 1 - np.sum(pb[:,slice(0,30,15)], 1)

plt.plot(st_heat[sim.p.slice_time_strobo])
plt.plot(st_heat)

plt.plot(pb[:,0])
plt.plot(pb_tgt[:,0])



#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
paramsOptim2 = paramsSim.copy()
paramsOptim2['algo']='NoOptim'

#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
ilib.reload(customOptim)
oc2 = customOptim.ToyModelOptim(paramsSim, paramsOptim2, None)
res2 = oc2.run()
