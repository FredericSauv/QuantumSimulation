#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:25:24 2018

@author: fred
"""

import numpy as np
import numpy.random as rdm


def sampling_constrained_step(nb_samples, nb_dim, step = 0.2, init = 0.0, end = 1.0, lim_min = 0.0, lim_max = 1.0):
    """ sample nb samples (where dim of one sample is nb_dim) under constraints that:
        + change between np.abs(sample[i+1] - sample[i]) < step
        + np.abs(sample[0] - init) < step
        + np.abs(sample[N] - end) < step
        + sample[i] in [lim_min, lim_max]
    Goal: efficient sampling in the sense that we should reject as few as possible
    without introducing artificial bias
    """
    
    nb_to_gen = min(75 * nb_samples, int(100000000 / nb_dim))
    #first step
    init_samples = rdm.uniform(max(lim_min, init - step), min(lim_max, init + step), nb_to_gen)
    steps_samples = rdm.uniform(-step, step, (nb_to_gen, nb_dim -1))
    samples = np.cumsum(np.c_[init_samples, steps_samples], axis = 1)
    filt = np.product((samples>=lim_min) * (samples<=lim_max), axis =1)
    filt *= (samples[:,-1]<=(end + step)) * (samples[:,-1] >= (end- step))
    samples = samples[np.ma.make_mask(filt)]
    return samples[range(min(samples.shape[0], nb_samples))]
    

test = sampling_constrained_step(10000, 15, 0.2) 