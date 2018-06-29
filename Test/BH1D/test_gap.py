# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append("../../../")
from QuantumSimulation.ToyModels.BH1D import BH1D
from QuantumSimulation.Utility.Optim import pFunc_base as pf
    
BH1D.info()

# Create a 1D BH chain linearly driven, evolve the GS of H(t=0) to T
# observe psi(T) avg(Var(n_i)) F(psi(T), GS_MI) etc..
list_size = [5, 10, 15, 20]

# [0.64252758733132032, 0.37735824340287805]
# [8.5510983763132753, 4.6369785794814016]

T = 100
linear_ramp = pf.LinearFunc(w=1/T , bias =0)
fom_name = ['f2t2:neg_fluence:0.0001_smooth:0.0005', 'f2t2', 'varN:sqrt', 'fluence', 'smooth', 'varN']
fom_name_last = ['last:'+f for f in fom_name]

dico_simul = {'control_obj':linear_ramp, 'L':6, 'Nb':6, 'mu':0, 'T':T, 'dt':0.01, 
            'flag_intermediate':False, 'setup':'1', 'state_init':'GS_i', 
            'state_tgt':'GS_f', 'fom':fom_name_last, 'kblock':0, 'pblock':1}

list_energy_gap = []
list_ratio_gap = []

for l in list_size:
    dico_simul['L'] = l
    dico_simul['Nb'] = l
    simul = BH1D(**dico_simul)
    #res_fom = simul.Simulate(fom = fom_name, store = True, debug = False)
    energy_evol = simul.EvolutionInstEnergies(nb = 2)
    gap_1 = energy_evol[:, 2] - energy_evol[:, 1]
    ind_min = np.argmin(gap_1)
    gap_min = gap_1[ind_min]
    U_min = energy_evol[ind_min, 0]
    ratio_min = U_min / (1-U_min)        
    list_energy_gap.append(gap_min)
    list_ratio_gap.append(ratio_min)