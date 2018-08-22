import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy

optim_type = 'BO2'

# Create a model
fom = ['f2t2:neg_fluence:0.0001_smooth:0.05']
T = 5.136511734353454
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'sps':5, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15'}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)


if(optim_type == 'BO2'):
    #BO
    optim_args = {'algo': 'BO2', 'maxiter':200, 'num_cores':4, 'init_obj':50, 'exploit_steps':49,
                  'acq':'EI', 'optim_num_anchor':25, 'optim_num_samples':10000}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    res = resBO2
    print(res.keys())

if(optim_type == 'DE'):
    optim_args = {'algo': 'DE', 'popsize':5, 'maxiter':1}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resDE = optim()
    print(resDE)
    res = resDE

if(optim_type == 'BO'):
    optim_args = {'algo': 'BO', 'maxiter':5, 'init_obj':15}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO = optim()
    resBO['last_func'] = model.control_fun
    res = resBO
    print(res.keys())

if(optim_type == 'NM'):
    #NM
    optim_args = {'algo': 'NM', 'init_obj': 'uniform_-1_1', 'nfev':10000}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resNM = optim()
    res = resNM
    print(res.keys())

## Create testing
fom_test = fom + ['f2t2', 'fluence', 'smooth', 'varN']
dico_test = copy.copy(dico_simul)
dico_test['fom']=fom_test
dico_test['track_learning'] = False
model_test = bh1d.BH1D(**dico_test)
optim_params = res['params']
res_test = model_test(optim_params, trunc_res = False)

#plot func optimal
func_used = model_test.control_fun
import numpy as np
x_to_plot = np.linspace(-0.2, T+0.1, 500)
func_used.plot_function(x_to_plot)

## Benchmarking results vs Linear

ow = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[0,1])
linear = ow * pFunc_base.LinearFunc(bias=0,w=1/T)
dico_test_linear = copy.copy(dico_test)
dico_test_linear['control_obj'] = linear
model_test_linear = bh1d.BH1D(**dico_test_linear)
res_test_linear = model_test_linear([], trunc_res=False)



import imp
imp.reload(learner1DBH)
## Benchmarking2
dico_linear = {'L':5, 'Nb':5, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom_test, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01r_linearr'}
dico_linear = learner1DBH._process_controler(dico_linear)
dico_linear['control_obj'] = learner1DBH._build_control_from_string(
            dico_linear['control_obj'], None, context_dico = dico_linear)
model_linear = bh1d.BH1D(**dico_linear)
res_linear = model_linear([], trunc_res=False)

model_linear.EvolutionPopAdiab(nb_ev=15)
model_linear.plot_pop_adiab()

print(res_test)
print(res_test_linear)


