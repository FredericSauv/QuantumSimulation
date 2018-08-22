#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:34:46 2018

@author: fred
"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import sys
import matplotlib.pylab as plt
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
func = pFunc_base.pFunc_base.read_func_from_file('opimizedF_rev')
from QuantumSimulation.Utility import Helper as ut





#==============================================================================
# ***EXTRA FUNCTIONS***  
#==============================================================================
def get_var_op(O, V):
    OV = O.dot(V)
    VOOV = np.asscalar(O.matrix_ele(V, OV))
    VOV2 = O.expt_value(V) ** 2
    var = VOOV -VOV2
    assert (np.imag(var) < 1e-8), 'Imaginary part not neglectible.. pb'
    return np.abs(var)

def get_site_pop(state, basis):
    op_n_sites = [hamiltonian([['n',[[1.0, i]]]], [], basis=basis, dtype=np.float64) for i in range(basis.L)]
    mean = [op.expt_value(state) for op in op_n_sites]   
    var = [get_var_op(op, state) for op in op_n_sites]
    return mean, np.sqrt(var)

def plot_site_population_states(states, basis):
    x_sites = np.arange(basis.L)
    ns = basis.Ns
    if(np.ndim(states) == 1):
        states = np.reshape(states, (1, len(states)))
    elif(np.shape(states)[0] == ns):
        states = np.transpose(states)
    for s in states:
        mean_s, var_s = get_site_pop(s, basis)
        plt.errorbar(x_sites, mean_s, yerr=var_s, fmt='o')


def plot_pop_adiab(model, **args_pop_adiab):
    """ Plot pop adiab where each population_t is dispatched on one of the 
    three subplot
    #TODO: better plots
    """
    if(hasattr(model,'pop_adiab')):
        limit_legend = args_pop_adiab.get('lim_legend', 10)
        limit_enlevels = args_pop_adiab.get('lim_enlevels', np.inf)
        pop_adiab = model.adiab_pop #txn
        t = model.adiab_t 
        en = model.adiab_en #txn
        cf = model.adiab_cf # txcf
        nb_levels = min(pop_adiab.shape[1], limit_enlevels)   
        en_0 = en[:,0]
        #[0,0] control function
        f, axarr = plt.subplots(2,2, sharex=True)
        axarr[0,0].plot(t, cf, label = 'f(t)')
        for i in range(nb_levels):
            pop_tmp = pop_adiab[:, i]
            max_tmp = np.max(pop_tmp)
            if(i<=limit_legend):
                lbl_tmp = str(i)
            else:
                lbl_tmp = None
            if(max_tmp > 0.1):
                axarr[0,1].plot(t, pop_tmp, label = lbl_tmp)
            elif(max_tmp > 0.01):
                axarr[1,1].plot(t, pop_tmp, label = lbl_tmp)
            axarr[1,0].plot(t, en[:, i]-en_0, label = lbl_tmp)
        
        ax_tmp = axarr[0,1]
        ax_tmp.legend(fontsize = 'x-small')
        ax_tmp.set_title('main pop')
        ax_tmp.set(xlabel='t', ylabel='%')
        
        ax_tmp = axarr[1,1]
        ax_tmp.legend(fontsize = 'x-small')
        ax_tmp.set_title('sec pop')
        ax_tmp.set(xlabel='t', ylabel='%')
        
        ax_tmp = axarr[0,0]
        ax_tmp.legend()
        ax_tmp.set_title('control')
        ax_tmp.set(xlabel='t', ylabel='cf')
        
        ax_tmp = axarr[1,0]
        ax_tmp.set_title('instantaneous ein')
        ax_tmp.set(xlabel='t', ylabel='E')
        save_fig = args_pop_adiab.get('save_fig')
        if(ut.is_str(save_fig)):
            f.savefig(save_fig)
    else:
        logger.warning("pcModel_qspin.plot_pop_adiab: no pop_adiab found.. Generate it first")





#==============================================================================
# ***MODEL MI->SF***  
#==============================================================================
fom = ['f2t2:neg_fluence:0.0001_smooth:0.05'] + ['f2t2', 'fluence', 'smooth', 'varN']
T=4.889428431607287
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'sps':5, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'control_obj':func}
model = bh1d.BH1D(**dico_simul)
optim_params = func.theta
res_model = model(optim_params, trunc_res = False)

#plot func optimal
x_to_plot = np.linspace(-0.2, T+0.1, 500)


#Benchmarking Linear
ow = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[1,0])
linear = ow * pFunc_base.LinearFunc(bias=1,w=-1/T)
dico_linear = copy.copy(dico_simul)
dico_linear['control_obj'] = linear
model_test_linear = bh1d.BH1D(**dico_linear)
res_linear = model_test_linear([], trunc_res=False)

print("MI->SF: with optimized control: {}".format(res_model))
print("MI->SF: with linear ramp: {}".format(res_linear))

func.plot_function(x_to_plot)
linear.plot_function(x_to_plot)

basis = model._ss
optim_state_t = model.EvolutionPopAdiab(nb_ev=basis.Ns)
plot_pop_adiab(model, save_fig = 'MI2SF_adiabpop.pdf')


linear_state_t = model_test_linear.EvolutionPopAdiab(nb_ev=basis.Ns)
plot_pop_adiab(model_test_linear, save_fig = 'MI2SF_linear_adiabpop.pdf')

plot_site_population_states(model.state_init, basis)
plot_site_population_states(model.state_tgt, basis)

#==============================================================================
# ***MODEL SF->MI***  
#==============================================================================
dico_rev = copy.copy(dico_simul)
f_clone = func.clone()
params_rev = f_clone.get_params()
params_rev['f0__f0__output_ow'] = np.flip(params_rev['f0__f0__output_ow'], 0) 
params_rev['f0__f1__f1__F'] = np.flip(params_rev['f0__f1__f1__F'], 0)
f_clone.set_params(**params_rev)
f_clone.plot_function(x_to_plot)
dico_rev['control_obj'] = f_clone
model_rev = bh1d.BH1D(**dico_rev)

plot_site_population_states(model_rev.state_init, basis)
plot_site_population_states(model_rev.state_tgt, basis)

optim_params_rev = f_clone.theta
res_model_rev = model_rev(optim_params_rev, trunc_res = False)

#Benchmarking Linear
ow_rev = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[0,1])
linear_rev = ow_rev * pFunc_base.LinearFunc(bias=0,w=1/T)
dico_linear_rev = copy.copy(dico_rev)
dico_linear_rev['control_obj'] = linear_rev
model_test_linear_rev = bh1d.BH1D(**dico_linear_rev)
res_linear_rev = model_test_linear_rev([], trunc_res=False)

print("SF->MI: with optimized control: {}".format(res_model_rev))
print("SF->MI: with linear ramp: {}".format(res_linear_rev))

f_clone.plot_function(x_to_plot)
linear_rev.plot_function(x_to_plot)

basis_rev = model_rev._ss
optim_state_t_rev = model_rev.EvolutionPopAdiab(nb_ev=basis_rev.Ns)
plot_pop_adiab(model_rev, save_fig = 'SF2MI_adiabpop.pdf')

linear_state_t_rev = model_test_linear_rev.EvolutionPopAdiab(nb_ev=basis_rev.Ns)
plot_pop_adiab(model_test_linear_rev, save_fig = 'SF2MI_linear_adiabpop.pdf')



SFtoMI_final = model_rev.adiab_pop[-1]
MItoSF_final = model.adiab_pop[-1]

SFtoMI_final_EV = np.squeeze(model_rev.adiab_evect[-1])
MItoSF_final_EV = np.squeeze(model.adiab_evect[-1])


plt.plot(SFtoMI_final[1:], label='final, SF->MI')
plt.plot(MItoSF_final[1:], label='final, MI->SF')
plt.legend()

plot_site_population_states(SFtoMI_final_EV[:,0], basis)
plot_site_population_states(MItoSF_final_EV[:,0], basis)

index_list = np.arange(basis.Ns)

which_sign_SF2MI = index_list[SFtoMI_final>0.004]
which_sign_MI2SF = index_list[MItoSF_final>0.01]

index = 1
st_SF2MI = SFtoMI_final_EV[:,which_sign_SF2MI[index]]
st_MI2SF = MItoSF_final_EV[:,which_sign_MI2SF[index]]

plot_site_population_states(st_MI2SF, basis)
plt.plot(st_MI2SF)



basis.get_vec
