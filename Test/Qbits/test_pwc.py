import sys
sys.path.append("../../../")
from QuantumSimulation.Simulation.Qbits.learn_Qbits import learnerQB as learner
from QuantumSimulation.ToyModels import TwoLevels as tl
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy
import numpy as np

optim_type = 'DE'

# Create a model
fom = ['f2t2:neg'] #['f2t2:neg_fluence:0.0001_smooth:0.01']
TQSL = np.pi/2 ### That's my TQSL
T= 1 * TQSL
dico_simul = {'T':T, 'dt':0.01, 'flag_intermediate':False, 'setup':'1Q1', 
              'state_init':'0', 'state_tgt':'m', 'fom':fom, 'fom_print':True, 
              'track_learning': True, 'ctl_shortcut':'owbds01_pwc15'}
# 'owbds01_pwc15'
dico_simul = learner._process_controler(dico_simul)
dico_simul['control_obj'] = learner._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = tl.Qubits(**dico_simul)

## 15 nice
## 30
if(optim_type == 'BO2'):
    #BO
    optim_args = {'algo': 'BO2', 'maxiter':100, 'num_cores':4, 'init_obj':100, 'acq':'EI'}
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
fom_test = fom + ['f2t2', 'fluence', 'smooth']
dico_test = copy.copy(dico_simul)
dico_test['fom']=fom_test
dico_test['track_learning'] = False
model_test = tl.Qubits(**dico_test)
optim_params = resBO2['params']
res_test = model_test(optim_params)
model_test.Simulate(store = True)
_ = model_test.EvolutionPopAdiab(nb_ev=2)
model_test.plot_pop_adiab()


fom = ['f2t2:neg'] #['f2t2:neg_fluence:0.0001_smooth:0.01']
TQSL = np.pi/np.sqrt(2)
T= 1 * TQSL
dico_simul = {'T':T, 'dt':0.01, 'flag_intermediate':False, 'setup':'1Q1', 
              'state_init':'0', 'state_tgt':'m', 'fom':fom, 'fom_print':True, 
              'track_learning': True, 'ctl_shortcut':'owbds01_pwc3'}
# 'owbds01_pwc15'
dico_simul = learner._process_controler(dico_simul)
dico_simul['control_obj'] = learner._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = tl.Qubits(**dico_simul)
res = model([0.4, 0.4, 0.4])
model.Simulate(store = True)
_ = model.EvolutionPopAdiab(nb_ev=2)
model.plot_pop_adiab()


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


