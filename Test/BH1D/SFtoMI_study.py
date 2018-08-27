import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy
import numpy as np



##STUDY OF OPTIM WITH Fid/VarN/VarN100
#==============================================================================
# ***GEN***  
#==============================================================================
T = 5.136511734353454
optim_args = {'algo': 'BO2', 'maxiter':500, 'num_cores':4, 'init_obj':75, 'exploit_steps':49,
              'acq':'EI', 'optim_num_anchor':25, 'optim_num_samples':10000}

save = False
#==============================================================================
# ***OPTIMIZATION***  
#==============================================================================
fom = ['f2t2:neg_fluence:0.0001_smooth:0.05']
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'sps':5, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15'}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)

try:
    func_used = pFunc_base.pFunc_base.read_func_from_file("SFtoMI_0")
except:
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    res = resBO2
    func_used = model.control_fun
    func_used.theta = res['params']
    if(save):
        func_used.save_to_file("SFtoMI_0")
    
#Testing
fom_test = fom + ['f2t2', 'fluence', 'smooth', 'varN']
dico_test = copy.copy(dico_simul)
dico_test['fom'] = fom_test
dico_test['track_learning'] = False
model_test = bh1d.BH1D(**dico_test)
optim_params = func_used.theta
res_test = model_test(optim_params, trunc_res = False)

#plot func optimal
x_to_plot = np.linspace(-0.2, T+0.1, 500)
func_used.plot_function(x_to_plot)



#==============================================================================
# ***Benchmarking results vs Linear***  
#==============================================================================
T_85pct = 6 * T 
ow = pFunc_base.OwriterYWrap(input_min = [-np.inf, T_85pct], input_max = [0, np.inf], output_ow =[0,1])
linear = ow * pFunc_base.LinearFunc(bias=0,w=1/T_85pct)
dico_test_linear = copy.copy(dico_test)
dico_test_linear['T'] = T_85pct
dico_test_linear['control_obj'] = linear
model_test_linear = bh1d.BH1D(**dico_test_linear)
res_test_linear = model_test_linear([], trunc_res=False)



#==============================================================================
# ***OPTIM2 with VarN***  
#==============================================================================
fom_varN = ['varN_smooth:0.05']
dico_simul_varN = copy.copy(dico_simul)
dico_simul_varN['fom'] = fom_varN
model_varN = bh1d.BH1D(**dico_simul_varN)

try:
    func_used_varN = pFunc_base.pFunc_base.read_func_from_file("SFtoMI_varN")
except:
    optim = Learner.learner_Opt(model = model_varN, **optim_args)
    res_varN = optim(track_learning=True)
    res_varN['last_func'] = model_varN.control_fun
    func_used_varN = model_varN.control_fun
    func_used_varN.theta = res_varN['params']
    if(save):
        func_used_varN.save_to_file("SFtoMI_varN")

#Testing
dico_test_varN = copy.copy(dico_test)
model_test_varN = bh1d.BH1D(**dico_test_varN)
optim_params_varN = func_used_varN.theta
res_test_varN = model_test_varN(optim_params_varN, trunc_res = False)

func_used_varN.plot_function(x_to_plot)
func_used.plot_function(x_to_plot)


#==============================================================================
# ***OPTIM3 with VarN100***  
# test varN100
#==============================================================================
## Testing
# import imp
# imp.reload(bh1d)


## variance of VarN depeding on the number of observations
# For Sf 0 / 0.0016 / 0.004 / 0.015 / 0.05
# For MI 0 / 0.0006 / 0.002 / 0.0017 /
dico_testingvar = copy.copy(dico_test)
fom_old = dico_test['fom']
fom_old[-1] = 'varN1000'
dico_testingvar['fom'] = fom_old
model_testingvar = bh1d.BH1D(**dico_testingvar)

nb_obs = 100
res_ensemble =np.zeros(nb_obs)

opt_p = np.random.uniform(size=15)
opt_p = optim_params_varN
#opt_p = np.zeros(15)

for n in range(nb_obs):    
    res_tmp = model_testingvar(opt_p, trunc_res = False, store = True)
    res_ensemble[n] = res_tmp[-1]

np.std(res_ensemble)
np.std(res_ensemble) / np.mean(res_ensemble)






## Optim with noise
fom_varN = ['varN1000_smooth:0.05']
dico_simul_varN1000 = copy.copy(dico_simul)
dico_simul_varN10000 = copy.copy(dico_simul)
dico_simul_varN100000 = copy.copy(dico_simul)
dico_simul_varN1000['fom'] = ['varN1000_smooth:0.05']
dico_simul_varN10000['fom'] = ['varN10000_smooth:0.05']
dico_simul_varN100000['fom'] = ['varN100000_smooth:0.05']
model_varN1000 = bh1d.BH1D(**dico_simul_varN1000)
model_varN10000 = bh1d.BH1D(**dico_simul_varN10000)
model_varN100000 = bh1d.BH1D(**dico_simul_varN100000)
model_varN_noisy = model_varN100000

try:
    func_used_varN_noisy = pFunc_base.pFunc_base.read_func_from_file("SFtoMI_varN_noisy")
except:
    model_varN_noisy = model_varN100000
    optim = Learner.learner_Opt(model = model_varN_noisy, **optim_args)
    res_varN_noisy = optim(track_learning=True)
    res_varN_noisy['last_func'] = model_varN_noisy.control_fun
    func_used_varN_noisy = model_varN_noisy.control_fun
    func_used_varN_noisy.theta = res_varN_noisy['params']
    if(save):
        func_used_varN_noisy.save_to_file("SFtoMI_varN100000")

#Testing
dico_test_varN_noisy = copy.copy(dico_test)
model_test_varN_noisy = bh1d.BH1D(**dico_test_varN_noisy)
optim_params_varN_noisy = func_used_varN_noisy.theta
res_test_varN1000 = model_test_varN_noisy(optim_params_varN_noisy, trunc_res = False)



model_varN_noisy(optim_params_varN_noisy, trunc_res=False)
model_varN(optim_params_varN_noisy, trunc_res=False)

func_used_varN_noisy.plot_function(x_to_plot)
func_used_varN.plot_function(x_to_plot)
func_used.plot_function(x_to_plot)


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


