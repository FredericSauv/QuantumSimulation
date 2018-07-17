#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:27:34 2018

@author: fred
"""
import numpy as np
import matplotlib.pylab as plt

alpha = np.arange(0, 1.57, 0.01)
p_alpha = np.square(np.sin(alpha))
p_tgt = np.square(np.sin(0.79))
alpha_opt = 0.79

nb_repeat = 1000
max_measure = 1000 

res = np.zeros([max_measure-1, 4]) #mean, sd, min, max

for nb_meas in range(1, max_measure):
    res_tmp = np.zeros(nb_repeat)    
    for r in range(nb_repeat):
        f_drawn = np.random.binomial(nb_meas, p_alpha) / nb_meas
        abs_diff = np.abs(p_tgt - f_drawn)
        min_diff = np.min(abs_diff)
        alpha_min = [a for n, a in enumerate(alpha) if abs_diff[n] == min_diff]
        res_tmp[r] = np.average(np.abs(np.array(alpha_min) - alpha_opt))
    res[nb_meas-1, 0] = np.mean(res_tmp)
    res[nb_meas-1, 1] = np.std(res_tmp)
    res[nb_meas-1, 2] = np.min(res_tmp)
    res[nb_meas-1, 3] = np.max(res_tmp)
    
plt.scatter(x=np.arange(1, max_measure) * len(alpha), y= res[:,0])
plt.xlim(450, 15000)