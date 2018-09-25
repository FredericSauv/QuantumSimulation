#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:32:34 2018

@author: fred
"""


import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import pFunc_base
from QuantumSimulation.Utility import Helper as ut
import numpy as np



L = 7
N= 7
T_long = 15
#==============================================================================
# Spectrum (linear drive)
#==============================================================================
fom = ['f2t2:neg_fluence:0.0001_smooth:0.05']
dico_simul = {'L':L, 'Nb':L,'sps':11,'mu':0, 'T':T_long, 'dt':0.001, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15',
              'kblock':0,'pblock':1}

ow = pFunc_base.OwriterYWrap(input_min = [-np.inf, T_long], input_max = [0, np.inf], output_ow =[0,1])
linear = ow * pFunc_base.LinearFunc(bias=0,w=1/T_long)
dico_simul['control_obj'] = linear
model_linear = bh1d.BH1D(**dico_simul)

mg = model_linear.FindMinDelta()
print(np.pi/mg)

# 5.125365060189967 L = 5 (16)
# 6.6735571351094105 L = 7 (133)
# 9.024327575008847 L = 10 (4752)
# 10.593133307114995 L = 12 (56822)