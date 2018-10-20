import sys, pdb, copy, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.QB import QBits, learn_QBits
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import numpy as np

# ==================================================================
# Parameters
#===================================================================
nb_parameters = 9
nb_meas = 10
smooth_coeff = 0.002

TQSL = np.pi/2
T= 1 * TQSL
x_plot = np.linspace(-0.01, T+0.01)

do_base, do_proj10, do_proj10_b = True, True, True
do_base_smooth, do_proj10_smooth, do_proj10_b_smooth = True, True, True
do_base_smoothconstr, do_proj10_smoothconstr, do_proj10_b_smoothconstr = True, True, True
do_base_stepconstr, do_proj10_stepconstr, do_proj10_b_stepconstr = True, True, True

# ==================================================================
# Models
#===================================================================
proj10 = ['last:proj10:neg:{0}'.format(nb_meas), 'smooth', 'f2t2'] 
fid = ['last:f2t2:neg', 'smooth']
proj10_smooth = ['last:proj10:neg_smooth:{0}:{1}'.format(smooth_coeff, nb_meas), 'smooth', 'f2t2']
fid_smooth = ['last:f2t2:neg_smooth:'+str(smooth_coeff), 'smooth', 'f2t2']

dico_base = {'T':T, 'dt':0.01, 'flag_intermediate':False, 'setup':'1Q1', 
             'state_init':'0', 'state_tgt':'m', 'fom':fid, 'fom_print':True, 
             'track_learning': True, 'ctl_shortcut':'owbds01_pwc'+ str(nb_parameters)}
dico_base = learn_QBits.learnerQB._process_controler(dico_base)
dico_base['control_obj'] = learn_QBits.learnerQB._build_control_from_string(
                    dico_base['control_obj'], None, context_dico = dico_base)
func_base = dico_base['control_obj'].clone()
dico_proj10 = copy.copy(dico_base)
dico_proj10.update({'fom': proj10})
dico_proj10_smooth = copy.copy(dico_base)
dico_proj10_smooth.update({'fom': proj10_smooth})
dico_base_smooth = copy.copy(dico_base)
dico_base_smooth.update({'fom': fid_smooth})

## Models
model_base = QBits.Qubits(**dico_base)
model_proj10 = QBits.Qubits(**dico_proj10)
model_base_smooth = QBits.Qubits(**dico_base_smooth)
model_proj10_smooth = QBits.Qubits(**dico_proj10_smooth)



## Test
perfect = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0]
model_base(perfect)
model_proj10(perfect)
model_proj10_smooth(perfect)

# ==================================================================
# Optim parameters
# Simple ('_base') or using binomial likelihood ('_b')
# with constraints on the space of accessible parameters
#
#===================================================================
optim_base = {'algo': 'BO2', 'maxiter':50, 'num_cores':2, 'init_obj':35, 
              'acq':'LCB', 'optim_num_anchor':15,  'optim_num_samples':10000, 
              'acquisition_weight':5,'acquisition_weight_lindec':True, 
              'initial_design_type':'random'}

optim_max_step = copy.copy(optim_base)
optim_max_step['constraints'] = 'step_0.4_0_1'
optim_max_smooth = copy.copy(optim_base)
optim_max_smooth['constraints'] = 'smoothlin_0.1_0_1'

optim_b = copy.copy(optim_base)
optim_b.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                'likelihood':'Binomial_'+str(nb_meas), 'normalize_Y':False})
optim_b_max_step = copy.copy(optim_b)
optim_b_max_step['constraints'] = 'step_0.4_0_1'
optim_b_max_smooth = copy.copy(optim_b)
optim_b_max_smooth['constraints'] = 'smoothlin_0.1_0_1'
    
# ==================================================================
# Without constraints
# with binomial observations ()
# 
#===================================================================
if(do_base):
    # perfect measurement
    optim = Learner.learner_Opt(model = model_base, **optim_base)
    res_base = optim(track_learning=True)
    p_base = res_base['params']
    p_base_exp = res_base['params_exp']
    func_base.theta = p_base
    func_base.plot_function(x_plot)
    model_base(p_base)

if(do_proj10):    
    # binomial measurement, gaussian likelihood
    optim = Learner.learner_Opt(model = model_proj10, **optim_base)
    res_proj10 = optim(track_learning=True)
    p_proj10 = res_proj10['params']
    p_proj10_exp = res_proj10['params_exp']
    func_base.theta = p_proj10
    func_base.plot_function(x_plot)
    model_base(p_proj10)
    model_base(p_proj10_exp)

if(do_proj10_b):    
    # binomial measurement, binomial likelihood
    optim = Learner.learner_Opt(model = model_proj10, **optim_b)
    res_proj10_b = optim(track_learning=True)
    p_proj10_b = res_proj10_b['params']
    p_proj10_b_exp = res_proj10_b['params_exp']
    func_base.theta = p_proj10_b_exp
    func_base.plot_function(x_plot)
    model_base(p_proj10_b)
    model_base(p_proj10_b_exp)
    


    
# ==================================================================
# With smoothness penalization
#===================================================================
if(do_base_smooth):
    optim = Learner.learner_Opt(model = model_base_smooth, **optim_base)
    res_base_smooth = optim(track_learning=True)
    p_base_smooth = res_base_smooth['params']
    p_base_smooth_exp = res_base_smooth['params_exp']
    func_base.theta = p_base_smooth
    func_base.plot_function(x_plot)
    model_base_smooth(p_base_smooth)

if(do_proj10_smooth):    
    optim = Learner.learner_Opt(model = model_proj10_smooth, **optim_base)
    res_proj10_smooth = optim(track_learning=True)
    p_proj10_smooth = res_proj10_smooth['params']
    p_proj10_smooth_exp = res_proj10_smooth['params_exp']
    func_base.theta = p_proj10_smooth
    func_base.plot_function(x_plot)
    model_base_smooth(p_proj10_smooth)
    model_base_smooth(p_proj10_smooth_exp)

if(do_proj10_b_smooth):    
    optim = Learner.learner_Opt(model = model_proj10_smooth, **optim_b)
    res_proj10_b_smooth = optim(track_learning=True)
    p_proj10_b_smooth = res_proj10_b_smooth['params']
    p_proj10_b_smooth_exp = res_proj10_b_smooth['params_exp']
    func_base.theta = p_proj10_b_smooth_exp
    func_base.plot_function(x_plot)
    model_base_smooth(p_proj10_b_smooth)
    model_base_smooth(p_proj10_b_smooth_exp)
    


  
# ==================================================================
# With   smoothness constraints
#==================================================================    
if(do_base_smoothconstr):
    optim = Learner.learner_Opt(model = model_base, **optim_max_smooth)
    res_base_smoothconstr = optim(track_learning=True)
    p_base_smoothconstr = res_base_smoothconstr['params']
    p_base_smoothconstr_exp = res_base_smoothconstr['params_exp']
    func_base.theta = p_base_smoothconstr
    func_base.plot_function(x_plot)
    model_base(p_base_smoothconstr)

if(do_proj10_smoothconstr):    
    optim = Learner.learner_Opt(model = model_proj10, **optim_max_smooth)
    res_proj10_smoothconstr = optim(track_learning=True)
    p_proj10_smoothconstr = res_proj10_smoothconstr['params']
    p_proj10_smoothconstr_exp = res_proj10_smoothconstr['params_exp']
    func_base.theta = p_proj10_smoothconstr_exp
    func_base.plot_function(x_plot)
    model_base(p_proj10_smoothconstr)
    model_base(p_proj10_smoothconstr_exp)

if(do_proj10_b_smoothconstr):    
    optim = Learner.learner_Opt(model = model_proj10, **optim_b_max_smooth)
    res_proj10_b_smoothconstr = optim(track_learning=True)
    p_proj10_b_smoothconstr = res_proj10_b_smoothconstr ['params']
    p_proj10_b_smoothconstr_exp = res_proj10_b_smoothconstr ['params_exp']
    func_base.theta = p_proj10_b_smoothconstr_exp
    func_base.plot_function(x_plot)
    model_base(p_proj10_b_smoothconstr)
    model_base(p_proj10_b_smoothconstr_exp)
    

    
    
    

# ==================================================================
# With max_step_size constraints
#==================================================================   
if(do_base_stepconstr):
    optim = Learner.learner_Opt(model = model_base, **optim_max_step)
    res_base_stepconstr = optim(track_learning=True, debug = True)
    p_base_stepconstr = res_base_stepconstr['params']
    p_base_stepconstr_exp = res_base_smoothconstr['params_exp']
    func_base.theta = p_base_stepconstr
    func_base.plot_function(x_plot)
    model_base(p_base_stepconstr)

if(do_proj10_stepconstr):    
    optim = Learner.learner_Opt(model = model_proj10, **optim_max_step)
    res_proj10_stepconstr = optim(track_learning=True)
    p_proj10_stepconstr = res_proj10_stepconstr['params']
    p_proj10_stepconstr_exp = res_proj10_stepconstr['params_exp']
    func_base.theta = p_proj10_stepconstr_exp
    func_base.plot_function(x_plot)
    model_base(p_proj10_stepconstr)
    model_base(p_proj10_stepconstr_exp)

if(do_proj10_b_stepconstr):    
    optim = Learner.learner_Opt(model = model_proj10, **optim_b_max_step)
    res_proj10_b_stepconstr = optim(track_learning=True)
    p_proj10_b_stepconstr = res_proj10_b_stepconstr['params']
    p_proj10_b_stepconstr_exp = res_proj10_b_stepconstr['params_exp']
    func_base.theta = p_proj10_b_stepconstr_exp
    func_base.plot_function(x_plot)
    model_base(p_proj10_b_stepconstr)
    model_base(p_proj10_b_stepconstr_exp)
    




# ==================================================================
# OPTIM WITH CONSTRAINTS: NEW DESIGN_SPACE step = 0.15 // 15 params
#===================================================================



# ==================================================================
# BERNOUILLY + Batch
#===================================================================

# ==================================================================
# Traget = 0.5
#===================================================================


# ==================================================================
# OPTIM 1: 
# is there a better way
#===================================================================
optim_with_noise = True
optim_ideal = True
optim_with_noise_custom = True
optim_args = {'algo': 'BO2', 'maxiter':50, 'num_cores':4, 'init_obj':25, 'acq':'EI'}
func_test = dico_no_noise['control_obj']

if(optim_with_noise):
    optim = Learner.learner_Opt(model = model_ensemble, **optim_args)
    res_ensemble = optim(track_learning=True)
    params_noise = res_ensemble['params']
    params_noise_exp = res_ensemble['params_exp']

if(optim_with_noise_custom):
    optim = Learner.learner_Opt(model = model_ensemble_custom, **optim_args)
    res_ensemble_custom = optim(track_learning=True)
    params_noise_custom = res_ensemble_custom['params']
    params_noise_custom_exp = res_ensemble_custom['params_exp']


if(optim_ideal):
    optim = Learner.learner_Opt(model = model_no_noise, **optim_args)
    res_no_noise = optim(track_learning=True)
    params_no_noise = res_no_noise['params']

## plot the different functions
func_test.theta = params_noise
func_test.plot_function(x_plot)
func_test.theta = params_noise_exp
func_test.plot_function(x_plot)
func_test.theta = params_no_noise
func_test.plot_function(x_plot)
func_test.theta = params_noise_custom
func_test.plot_function(x_plot)


# TESTING
dico_test_noise = copy.copy(dico_ensemble)
dico_test_noise['noise'] = {'Ez':'normal_0_0.2', 'nb_H_ensemble':1000}
model_test_noise = QBits.Qubits(**dico_test_noise)

dico_test_no_noise = copy.copy(dico_no_noise)
model_test_no_noise = QBits.Qubits(**dico_test_no_noise)

dico_test_noise_custom = copy.copy(dico_ensemble_custom)
dico_test_noise_custom['noise'] = {'Ez':'normal_0_0.2', 'nb_H_ensemble':1000}
model_test_noise_custom = QBits.Qubits(**dico_test_noise_custom)

#Trained with no noise, tested with no noise
res_trainNoNoise_testNoNoise = model_test_no_noise(params_no_noise)
res_trainNoise_testNoNoise = model_test_no_noise(params_noise)
res_trainNoiseExp_testNoNoise = model_test_no_noise(params_noise_exp)

res_trainNoNoise_testNoise = model_test_noise(params_no_noise)
res_trainNoise_testNoise = model_test_noise(params_noise, update_H=False)
res_trainNoiseExp_testNoise = model_test_noise(params_noise_exp, update_H=False)


res_trainNoNoise_testNoiseCustom = model_test_noise_custom(params_no_noise)
res_trainNoise_testNoiseCustom = model_test_noise_custom(params_noise, update_H=False)
res_trainNoiseCustom_testNoiseCustom = model_test_noise_custom(params_noise_custom, update_H=False)


# ==================================================================
# OPTIM 2: 
# fom_ensemble
#===================================================================
optim_with_noise = True
optim_ideal = True
optim_args = {'algo': 'BO2', 'maxiter':50, 'num_cores':4, 'init_obj':25, 'acq':'EI'}
func_test = dico_no_noise['control_obj']


