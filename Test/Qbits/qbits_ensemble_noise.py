import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.QB import QBits, learn_QBits
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import pdb
import copy
import numpy as np

optim_type = 'BO2'

# Create a model
fid = ['f2t2'] # 'proj100:neg_fluence:0.0001_smooth:0.01'
TQSL = np.pi/2
T= 1 * TQSL


# ==================================================================
# NO NOISE
#===================================================================
dico_no_noise = {'T':T, 'dt':0.01, 'flag_intermediate':False, 'setup':'1Q1', 
              'state_init':'0', 'state_tgt':'m', 'fom':fid, 'fom_print':True, 
              'track_learning': True, 'ctl_shortcut':'owbds01_pwc5'}

dico_no_noise = learn_QBits.learnerQB._process_controler(dico_no_noise)
dico_no_noise['control_obj'] = learn_QBits.learnerQB._build_control_from_string(
                    dico_no_noise['control_obj'], None, context_dico = dico_no_noise)
model = QBits.Qubits(**dico_no_noise)
print(model._energies)
model([1.0, 1.0, 0.0, 0.0, 1.0])
model.control_fun.plot_function(np.linspace(0, T+0.01, 50))
st = model.EvolutionPopAdiab()
pop = np.square(np.abs(st))
model.plot_pop_adiab()

import matplotlib.pylab as plt
plt.plot(model.t_array, pop[0])
plt.plot(model.t_array, pop[1])

# ==================================================================
# NOISE SIMPLE
#===================================================================
dico_noise = copy.copy(dico_no_noise)
dico_noise['noise'] = {'Ez':'normal_0_0.2'}

model_noise = QBits.Qubits(**dico_noise)
print(model_noise._energies)
model_noise([1.0, 1.0, 0.0, 0.0, 1.0])

# ==================================================================
# NOISE ENSEMBLE
#===================================================================
dico_ensemble = copy.copy(dico_no_noise)
dico_ensemble['noise'] = {'Ez':'normal_0_0.2', 'nb_H_ensemble':10}
dico_ensemble['mp_obj'] = True

model_ensemble = QBits.Qubits(**dico_ensemble)
print(model_ensemble._energies)
model_ensemble([1.0, 1.0, 0.0, 0.0, 1.0])




## 15 nice
## 30
if(optim_type == 'BO2'):
    #BO
    optim_args = {'algo': 'BO2', 'maxiter':50, 'num_cores':4, 'init_obj':25, 'acq':'EI'}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    print(resBO2)
    res = resBO2

if(optim_type == 'DE'):
    optim_args = {'algo': 'DE', 'popsize':5, 'maxiter':75}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resDE = optim()
    print(resDE)
    res = resDE

if(optim_type == 'BO'):
    optim_args = {'algo': 'BO', 'maxiter':5, 'init_obj':15}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO = optim()
    resBO['last_func'] = model.control_fun
    print(resBO)
    res = resBO

if(optim_type == 'NM'):
    #NM
    optim_args = {'algo': 'NM', 'init_obj': 'uniform_-1_1', 'nfev':10000}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resNM = optim()
    print(resNM)
    res = resNM

## Create testing
dico_test = copy.copy(dico_simul)
dico_test['noise'] = {}
fom_test = ['f2t2:neg_fluence:0.0001_smooth:0.0005']  + ['f2t2', 'fluence', 'smooth']
dico_test['fom']=fom_test
dico_test['track_learning'] = False

model_test = tl.Qubits(**dico_test)
optim_params = resBO2['params']
res_test = model_test(optim_params)

optim_params_exp = resBO2['params_exp']
res_test_exp = model_test(optim_params_exp)

model_test.Simulate(store = True)
_ = model_test.EvolutionPopAdiab(nb_ev=2)
model_test.plot_pop_adiab()


#plot func optimal
func_used = model_test.control_fun
import numpy as np
x_to_plot = np.linspace(0, T, 500)
func_used.plot_function(x_to_plot)

like_function = False
if(like_function):
    pFunc_base.pFunc_base.save_a_func_to_file(func_used, 'optim_bo_owbds01_pwc15_bdaries.txt', type_write = 'w')

## Benchmarking results vs Linear
ow = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[0,1])
linear = ow * pFunc_base.LinearFunc(bias=0,w=1/T)
dico_test_linear = copy.copy(dico_test)
dico_test_linear['control_obj'] = linear
model_test_linear = tl.Qubits(**dico_test_linear)
res_test_linear = model_test_linear([])


