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
T=4.889428431607287
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'sps':5, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01r_pwlr15'}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(
dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)


#==============================================================================
# ***OPTIMIZATION***  
#==============================================================================
optim_args = {'algo': 'BO2', 'maxiter':500, 'num_cores':4, 'init_obj':75, 'exploit_steps':49,
              'acq':'EI', 'optim_num_anchor':25, 'optim_num_samples':10000}
optim = Learner.learner_Opt(model = model, **optim_args)
resBO2 = optim(track_learning=True)
resBO2['last_func'] = model.control_fun
res = resBO2
print(res.keys())



#==============================================================================
# ***TESTING***  
#==============================================================================
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

#Benchmarking Linear
ow = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[1,0])
linear = ow * pFunc_base.LinearFunc(bias=1,w=-1/T)
dico_test_linear = copy.copy(dico_test)
dico_test_linear['control_obj'] = linear
model_test_linear = bh1d.BH1D(**dico_test_linear)
res_test_linear = model_test_linear([], trunc_res=False)

print("with optimized control: {}".format(res_test))
print("with linear inverted ramp: {}".format(res_test_linear))


#==============================================================================
# ***Evol under optimized more info***  
#==============================================================================
basis = model_test._ss
optim_state_t = model_test.EvolutionPopAdiab(nb_ev=basis.Ns)
plot_pop_adiab(model_test, save_fig = 'test_2.pdf')







#==============================================================================
# ***Reverseit / test it for the SF to MI transit***  
#==============================================================================
import imp
imp.reload(pFunc_base)
f_clone = func_used.clone()

f_rev = pFunc_base.pFunc_fromcallable(lambda t: f_clone(T-t))

## optim_reverse
dico_opt_rev = {'L':5, 'Nb':5, 'mu':0, 'sps':5, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_pwl15'}
dico_opt_rev = learner1DBH._process_controler(dico_opt_rev)
dico_opt_rev['control_obj'] = learner1DBH._build_control_from_string(
dico_opt_rev['control_obj'], None, context_dico = dico_opt_rev)

model_opt_rev = bh1d.BH1D(**dico_opt_rev)



#Benchmarking rev Linear
ow_r = pFunc_base.OwriterYWrap(input_min = [-100, T], input_max = [0, 100+T], output_ow =[0,1])
linear_r = ow * pFunc_base.LinearFunc(bias=0,w=1/T)
dico_test_linear_r = copy.copy(dico_opt_rev)
dico_test_linear_r['control_obj'] = linear_r
model_test_linear_r = bh1d.BH1D(**dico_test_linear_r)
res_test_linear_r = model_test_linear_r([], trunc_res=False)

print(res_test)
print(res_test_linear)






#-----------------------------------------------------------------------------#
# plot capabilities
#-----------------------------------------------------------------------------#
import matplotlib.pylab as plt
from QuantumSimulation.Utility import Helper as ut
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