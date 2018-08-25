#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:19:20 2018

@author: fred
"""

import numpy as np
nb_repeat = 1000
nb_obs = 100000
rdm_drawns = np.random.poisson(1, (nb_repeat, nb_obs, 5))

res = np.array([(np.average(np.squeeze(r)), np.average(np.std(np.squeeze(r), 0))) for r in rdm_drawns])
res_std = np.squeeze(res[:, 1])
res_mean = np.squeeze(res[:, 0])

np.std(res_std)