#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:13:20 2019

@author: fred
"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse

pot_SF = lambda x:  np.square(np.sin(x))
pot_MI = lambda x:  5 * np.square(np.sin(x)) - 10
mi_wf = lambda x, xc, cl: 7*np.exp(-np.square((x-xc)/cl)) - 10


n_T = 2
x = np.linspace(0, n_T * 2* np.pi, 10000)
x_dashed_1 = np.linspace(-np.pi/2, 0, 10000)
x_dashed_2 = np.linspace(0, np.pi/2, 10000) + n_T * 2* np.pi

size_mi = 0.3
x_mi_0 = size_mi * np.linspace(-np.pi, np.pi, 10000)
x_mi_1 = size_mi * np.linspace(-np.pi, np.pi, 10000) + np.pi
x_mi_2 = size_mi * np.linspace(-np.pi, np.pi, 10000) + 2 * np.pi
x_mi_3 = size_mi * np.linspace(-np.pi, np.pi, 10000) + 3 * np.pi
x_mi_4 = size_mi * np.linspace(-np.pi, np.pi, 10000) + 4 * np.pi

fig, ax = plt.subplots()
ax.plot(x, pot_MI(x), 'b')
ax.plot(x_dashed_1, pot_MI(x_dashed_1), 'b--')
ax.plot(x_dashed_2, pot_MI(x_dashed_2), 'b--')

for n, x_mi in enumerate([x_mi_0, x_mi_1, x_mi_2, x_mi_3, x_mi_4]):
    ax.plot(x_mi, mi_wf(x_mi, n* np.pi, 0.3), 'r')
    ax.fill_between(x_mi, -10 * np.ones(len(x_mi)), mi_wf(x_mi, n * np.pi, 0.3), color='red', alpha=0.5)


ax.plot(x, pot_SF(x), 'b')
ax.plot(x_dashed_1, pot_SF(x_dashed_1), 'b--')
ax.plot(x_dashed_2, pot_SF(x_dashed_2), 'b--')

sf = Ellipse((2*np.pi, 0.5), 5.2* np.pi, 0.5, facecolor='r', alpha=0.5)
ax.add_artist(sf)

ax.text(-2, 1.5, '(i)')
ax.text(-2, -3.5, '(ii)')
fig.patch.set_visible(False)
ax.axis('off')
plt.savefig("sftomi_schematic.pdf", transparent=True, pad_inches=0)