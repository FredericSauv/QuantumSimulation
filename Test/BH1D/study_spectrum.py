import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
sys.path.append("../../../QuantumSimulation")
from QuantumSimulation.ToyModels.BH.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels.BH import BH1D as bh1d
# import copy
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import matplotlib.pylab as plt
import numpy as np
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis

### Prepare the model
T = 9.778856863214575/2
L = 5
sps = 5
fom = ['f2t2:neg_fluence:0.0001_smooth:0.005', 'f2t2', 'fluence', 'smooth', 'varN']
## Benchmarking2
dico_model = {'L':L, 'Nb':L, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 'sps':sps, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_linear'}
dico_model = learner1DBH._process_controler(dico_model)
dico_model['control_obj'] = learner1DBH._build_control_from_string(dico_model['control_obj'], None, context_dico = dico_model)

### Run it
model = bh1d.BH1D(**dico_model)
res = model([], trunc_res=False)
basis = model._ss
state_t = model.EvolutionPopAdiab(nb_ev=basis.Ns)


### Get data
t = model.t_array
st = state_t
en = model.adiab_en
cf = model.adiab_cf
EV = model.adiab_evect

print("Min neutral energy gap:{0}".format(np.min(en[:, 1]-en[:, 0])))
print("T_QSL asociated:{0}".format(np.pi/np.min(en[:, 1]-en[:, 0])))

t_mask = t < T * 0.9
t_filt, st_filt, en_filt, cf_filt = t[t_mask], st[:, t_mask], en[t_mask], cf[t_mask]
U_filt = cf_filt / (1-cf_filt)
E_filt = en_filt/ (1-cf_filt)


plt.plot(U_filt, E_filt, color = 'black')
plt.xlabel('U/J')
plt.ylabel('E/J')
plt.title('L=N=5')

plt.plot(cf, en, color = 'black')
plt.xlabel('E')
plt.ylabel('t')
plt.title('L=N=5')

### 
plt.plot(U_filt, E_filt, color = 'black')
plt.plot(U_filt, E_filt)


eigvects_0 = np.squeeze(EV[0])
eigvects_F = np.squeeze(EV[-1])
eigvect_0_0 = eigvects_0[:,0]
eigvect_0_1 = eigvects_0[:,1]
eigvect_0_2 = eigvects_0[:,2]
eigvect_0_3 = eigvects_0[:,3]

eigvect_F_0 = eigvects_F[:,0]
eigvect_F_1 = eigvects_F[:,1]
eigvect_F_2 = eigvects_F[:,2]
eigvect_F_3 = eigvects_F[:,3]

eigvect_F_13 = eigvects_F[:,13]


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


plot_site_population_states(eigvect_F_13, basis)





