import logging
logging.basicConfig(level = logging.INFO)
logging.getLogger(__name__)
import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy
import pdb

optim_type = 'BO2'

# TO ASSESS: BATCH vs no BATCH (for the same number of iter) precision/time
# Nb anchor 5/15/50


# Create a model
fom = ['f2t2:neg_fluence:0.0001_smooth:0.005']
T=9.778856863214575/2

dico_simul = {'L':5, 'Nb':5, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01r_pwc10'}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(
                    dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)

optim_main = {'algo': 'BO2', 'maxiter':50, 'num_cores':4, 'init_obj':30, 
              'exploit_steps':30,'acq':'EI', 'optim_num_anchor':15, 'optim_num_samples':10000}



if(optim_type == 'GP'):
    #BO
    optim_args = optim_main
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    res = resBO2
    print(res.keys())


if(optim_type == 'GP_ARD'):
    #BO
    optim_args = optim_main.update({'ARD':True})
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    res = resBO2
    print(res.keys())


if(optim_type == 'GPSparse'):
    optim_args = optim_main.update({'model_type':'sparseGP', 'number_inducing':10})
    optim = Learner.learner_Opt(model = model, **optim_args)
    resDE = optim(track_learning=True)
    print(resDE)
    res = resDE

if(optim_type == 'GPBatch'):
    optim_args = optim_main.update({'batch_method':'local_penalization', 'batch_size':4})
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO = optim(track_learning=True)
    resBO['last_func'] = model.control_fun
    res = resBO
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

print(res_test)
print(res_test_linear)


