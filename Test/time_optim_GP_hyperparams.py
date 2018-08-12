#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:13:24 2018

@author: fred
"""

## Get an idea of how long it takes to invert M times a dense NxN mat 



#Naive ...
import numpy as np
import time
N = 700
M = 1000
mat_rdm = np.random.uniform(size = [N, N])


time_start = time.time()
for _ in range(M): 
    mat_inv = np.linalg.inv(mat_rdm)

time_elapsed = time.time() - time_start


### Will need to debug into BayesianOptimization