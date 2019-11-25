#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:34:55 2019

@author: fred
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import utilities as ut

#qubits
zero = qt.qubits.qubit_states(1,[0])
one = qt.qubits.qubit_states(1,[1])
I, X, Y, Z = qt.identity([2]), qt.sigmax(), qt.sigmay(), qt.sigmaz()
Rx, Ry, Rz = qt.rx, qt.ry, qt.rz

ghz1 = ut.get_ghz(2)


###########
# Decomp Hermitian operators Schmidt
###########
H0 = qt.tensor(X, X) + qt.tensor(Y, Y)
