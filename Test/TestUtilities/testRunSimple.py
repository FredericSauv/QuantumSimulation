#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:19:38 2018

@author: fred
"""
import sys
#sys.path.append('../../../FloquetSimulation')
sys.path.append('../../')

import FloquetSimulation.Simulation.ToyModelOptim as customOptim
import importlib as ilib
import pdb 

customOptim.ToyModelOptim.helpConfiguration()

#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
paramsSim = {'eta' : 0.2, 'setup' : 2, 'tgt' : 0.1, 'tgt_type' : 'sigmaX',
             'tgt_picture': 'interaction', 'nPPP': 17, 'nPeriodEff':1, 'nbParams':1, 
             '_skipMotionalError':True, 'decimalRounding': 5, 'maxPopLastMotional':0.01, 
             'maxDevSumProba':0.01, 'IterMax':1}

paramsOptim ={'algo': 'NoOptim', 'name_FOM': ['HeatingM', 'Fidelity'], 'params_init': 'force_[1.074]',
              'nb_params' : 1}

paramsSimTest = paramsSim


#------------------------------------------------------------------------------
#                   Parameters simulation / optim 
#------------------------------------------------------------------------------
ilib.reload(customOptim)
oc = customOptim.ToyModelOptim(paramsSim, paramsOptim, paramsSimTest)
res = oc.run()

#pdb.run('oc.run()')



sim = oc.simulator
st = sim.state_store
pb = sim.fss.probaFromState(st)
st_tgt = sim.stateTGT
pb_tgt = sim.fss.probaFromState(st_tgt)
import numpy as np
import matplotlib.pyplot as plt
st_heat = 1 - np.sum(pb[:,slice(0,30,15)], 1)
plt.plot(st_heat)

plt.plot(pb[:,0])
plt.plot(pb_tgt[:,0])


fid_t = sim.fss.fidelity_distance_t(st, st_tgt)
plt.plot(fid_t)