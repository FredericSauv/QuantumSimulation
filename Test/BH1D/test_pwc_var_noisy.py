import sys
sys.path.append("../../../")
from QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy

optim_type = 'BO2'
# Create a model
fom = ['varN:rdmtime']
T = 4.889428431607287 *1.5
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwc5', 
              'noise':{'fom':'normal_0_0.1'}}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)


if(optim_type == 'BO2'):
    #BO
    optim_args = {'algo': 'BO2', 'maxiter':75, 'num_cores':1, 'init_obj':50, 'exploit_steps':30,'acq':'EI'}
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
fom_test = ['f2t2', 'fluence', 'smooth', 'varN']
dico_test = copy.copy(dico_simul)
dico_test['fom']=fom_test
dico_test['track_learning'] = False
model_test = bh1d.BH1D(**dico_test)
optim_params = res['params_exp']
res_test = model_test(optim_params)

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
res_test_linear = model_test_linear([])


