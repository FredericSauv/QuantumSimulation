#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:06:12 2018

@author: fred
"""
import numpy as np
import matplotlib.pylab as plt

def transfo(x, strtch = 15):
    coeff = 1/np.tanh(strtch*0.5)
    return 0.5 + coeff/2 * np.tanh(strtch*(x-0.5))


underlying = np.arange(0, 1.1 ,0.1)

sd_original = 0.15
nb_draws = 100
draw_original = np.array([np.random.normal(underlying, sd_original) for _ in range(nb_draws)]).T
draw_transfo = transfo(draw_original)

mean = np.average(draw_transfo, 1)
yerr = np.std(draw_transfo, 1)
xerr = np.std(draw_original, 1)



plt.errorbar(underlying, mean,  yerr=yerr)


plt.plot(underlying, mean)
yerr

mean
xerr