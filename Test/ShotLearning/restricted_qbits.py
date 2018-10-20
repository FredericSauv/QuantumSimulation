import sys, pdb, copy, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.ModelExamples import restricted_qubit
from QuantumSimulation.Utility.Optim import Learner
import numpy as np



model_1shot = restricted_qubit(1)
model_10shots = restricted_qubit(10)



optim_base = {'algo': 'BO2', 'maxiter':100, 'num_cores':2, 'init_obj':50, 
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



