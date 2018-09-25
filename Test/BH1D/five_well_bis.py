import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
from QuantumSimulation.Utility import Helper as ut
import copy
import numpy as np
import imp


L = 5
N=5
prefix = 'five_wells'
#==============================================================================
# Spectrum (linear drive)
#==============================================================================
T = np.pi / 0.612950061075626
T_long = 10
fom = ['f2t2:neg_fluence:0.0001_smooth:0.05']
dico_simul = {'L':L, 'Nb':L,'sps':None,'mu':0, 'T':T, 'dt':0.001, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15',
              'kblock':0,'pblock':1}

ow = pFunc_base.OwriterYWrap(input_min = [-np.inf, T_long], input_max = [0, np.inf], output_ow =[0,1])
linear = ow * pFunc_base.LinearFunc(bias=0,w=1/T_long)
dico_linear = copy.copy(dico_simul)
dico_linear['control_obj'] = linear
dico_linear['T'] = T_long
model_linear = bh1d.BH1D(**dico_linear)
res_test_linear = model_linear([], trunc_res=False)
state_tmp = model_linear.EvolutionPopAdiab(nb_ev =16)
# min_gap = model_linear.FindMinDelta()
model_linear.plot_pop_adiab(plot_gap = True)
func_fivewells_linear = model_linear.control_fun
if(False):
    func_fivewells_linear.save_to_file(prefix+'func_linear')

optim_args = {'algo': 'BO2', 'maxiter':200, 'num_cores':4, 'init_obj':100, 'exploit_steps':49,
              'acq':'EI', 'optim_num_anchor':25, 'optim_num_samples':10000}



import matplotlib.pylab as plt
en = model_linear.adiab_en #txn
cf = model_linear.adiab_cf # txcf
#[0,0] control function
f, axarr = plt.subplots(1,1, sharex=True)
for i in range(16):
    axarr.plot(cf, en[:, i], color = 'black')
axarr.set_ylabel(r"$E$", fontsize=16)
axarr.set_xlabel(r"$\Gamma(t)$", fontsize=16)

diff_01 = en[:, 1] - en[:, 0]
index_min = np.argmin(diff_01)
E0_min = en[index_min, 0]
cf_min = cf[index_min]
y_min= diff_01[index_min]
axarr.arrow(cf_min, E0_min, 0, y_min)
axarr.text(cf_min, E0_min - 1, r"$\Delta = %.2f$"%(y_min), fontsize=12)
f.savefig('spectrum_LN5_2.pdf', bbox_inches='tight', transparent=True, pad_inches=0)



#==============================================================================
# Try to reach GS at the end  
#==============================================================================
fom_GS = ['f2t2:neg_fluence:0.0001_smooth:0.05']
dico_GS = {'L':L, 'Nb':L, 'mu':0,'sps':None, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom_GS, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15',
              'kblock':0,'pblock':1}

dico_GS = learner1DBH._process_controler(dico_GS)
dico_GS['control_obj'] = learner1DBH._build_control_from_string(
        dico_GS['control_obj'], None, context_dico = dico_GS)
model_GS = bh1d.BH1D(**dico_GS)

optim_GS = Learner.learner_Opt(model = model_GS, **optim_args)
res_GS = optim_GS(track_learning=True)
model_GS.control_fun.plot_function(np.arange(-0.01, T+0.01, 0.01))
state_tmp = model_GS.EvolutionPopAdiab(nb_ev =3)
model_GS.plot_pop_adiab(plot_gap = True)
func_threewells_GS = model_GS.control_fun
if(False):
    func_threewells_GS.save_to_file('func_threewells_GS')


#==============================================================================
# Try to reach ES at the end  
#==============================================================================
fom_firstE = ['projSS:neg_fluence:0.0001_smooth:0.05']
dico_firstE = {'L':L, 'Nb':L, 'sps':None, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'ESS_1_10.0', 'fom':fom_firstE, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15', 
              'kblock':0,'pblock':1}
dico_firstE = learner1DBH._process_controler(dico_firstE)
dico_firstE['control_obj'] = learner1DBH._build_control_from_string(
            dico_firstE['control_obj'], None, context_dico = dico_firstE)
model_firstE = bh1d.BH1D(**dico_firstE)
optim_firstE = Learner.learner_Opt(model = model_firstE, **optim_args)
res_sym = optim_firstE(track_learning=True)
model_firstE.control_fun.plot_function(np.arange(-0.01, T+0.01, 0.01))
state_tmp = model_firstE.EvolutionPopAdiab(nb_ev =3)
model_firstE.plot_pop_adiab(plot_gap = True)
func_threewells_E1 = model_firstE.control_fun
if(False):
    func_threewells_E1.save_to_file('func_threewells_E1')








#Testing
fom_test = fom + ['f2t2', 'fluence', 'smooth', 'varN']
dico_test = copy.copy(dico_simul)
dico_test['fom'] = fom_test
dico_test['track_learning'] = False
model_test = bh1d.BH1D(**dico_test)
optim_params = func_used.theta
res_test = model_test(optim_params, trunc_res = False)


if(False):
    
    test = model.state_tgt
    test2 = np.square(np.abs(test))
    index = np.arange(len(test2))[test2>0.05]
    np.array([op.expt_value(test) for op in model._op_n_sites])
    [model._h_variance(op, test) for op in model._op_n_sites]
    import matplotlib.pylab as plt
    plt.plot(test2)
    model._get_basis_to_fock(92)
    

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










import matplotlib.pylab as plt
plt.rc('text', usetex=False)

#plt.rc('text', usetex=True) 

def plot_pop_adiab(model, **args_pop_adiab):
    """ Plot pop adiab where each population_t is dispatched on one of the 
    three subplot
    #TODO: better plots
    """
    col_list = ['b', 'g', 'r', 'c', 'm', 'k','C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] * 10
    if(hasattr(model,'pop_adiab')):
        limit_legend = args_pop_adiab.get('lim_legend', 15)
        limit_enlevels = args_pop_adiab.get('lim_enlevels', np.inf)
        pop_adiab = model.adiab_pop #txn
        t = model.adiab_t 
        en = model.adiab_en #txn
        cf = model.adiab_cf # txcf
        nb_levels = min(pop_adiab.shape[1], limit_enlevels)    
        #[0,0] control function
        f, axarr = plt.subplots(2,2, sharex=True)
        axarr[0,0].plot(t, cf, label = 'f')#r"$\Gamma(t)$")
        for i in range(nb_levels):
            col = col_list[i]
            pop_tmp = pop_adiab[:, i]
            max_tmp = np.max(pop_tmp)
            if(i<=limit_legend):
                lbl_tmp = str(i)
            else:
                lbl_tmp = None
            if(max_tmp > 0.1):
                axarr[0,1].plot(t, pop_tmp, label = lbl_tmp, color = col)
            elif(max_tmp > 0.01):
                axarr[1,1].plot(t, pop_tmp, label = lbl_tmp, color = col)
            if(i<10):
                axarr[1,0].plot(t, en[:, i] - en[:, 0], label = lbl_tmp, color = col)
        ax_tmp = axarr[0,1]
        ax_tmp.legend(fontsize = 'x-small')
        ax_tmp.set_title('Population', fontsize = 8)
        ax_tmp.set(xlabel='t')
        
        ax_tmp = axarr[1,1]
        ax_tmp.legend(fontsize = 'x-small')
        ax_tmp.set(xlabel='t')
        
        ax_tmp = axarr[0,0]
        ax_tmp.legend()
        ax_tmp.set(xlabel='t', ylabel='cf')
        
        ax_tmp = axarr[1,0]
        ax_tmp.set(xlabel='t', ylabel='E') #r"$E_i - E_0$"
    
        save_fig = args_pop_adiab.get('save_fig')
        if(ut.is_str(save_fig)):
            f.savefig(save_fig)

    else:
        print("pcModel_qspin.plot_pop_adiab: no pop_adiab found.. Generate it first")

