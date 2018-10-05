import sys, pdb, copy, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.QB import QBits, learn_QBits
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import numpy as np

# Create a model
fid = ['last:f2t2:neg'] # 'proj100:neg_fluence:0.0001_smooth:0.01'
energy = ['last:energyinf']
TQSL = np.pi/2
T= 1 * TQSL
x_plot = np.linspace(-0.01, T)

# ==================================================================
# NO NOISE
#===================================================================
dico_no_noise = {'T':T, 'dt':0.01, 'flag_intermediate':False, 'setup':'1Q1', 
              'state_init':'0', 'state_tgt':'m', 'fom':energy, 'fom_print':True, 
              'track_learning': True, 'ctl_shortcut':'owbds01_pwc5'}
dico_no_noise = learn_QBits.learnerQB._process_controler(dico_no_noise)
dico_no_noise['control_obj'] = learn_QBits.learnerQB._build_control_from_string(
                    dico_no_noise['control_obj'], None, context_dico = dico_no_noise)

model_no_noise = QBits.Qubits(**dico_no_noise)
model_no_noise([1.0, 1.0, 0.0, 0.0, 0.6781])


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
dico_ensemble['fom'] = ['last:f2t2:neg']
model_ensemble = QBits.Qubits(**dico_ensemble)
print(model_ensemble._energies)
model_ensemble([1.0, 1.0, 0.0, 0.0, 1.0])


# ==================================================================
# NOISE ENSEMBLE + fom_ensemble
#===================================================================
dico_ensemble_custom = copy.copy(dico_ensemble)
dico_ensemble_custom['fom_ensemble'] = 'avgens_stdens:1'
model_ensemble_custom = QBits.Qubits(**dico_ensemble_custom)
print(model_ensemble_custom._energies)
model_ensemble_custom([1.0, 1.0, 0.0, 0.0, 1.0])


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


