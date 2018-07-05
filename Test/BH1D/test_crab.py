import sys
sys.path.append("../../../")
from QuantumSimulation.Simulation.BH1D.learn_1DBH import learner1DBH
from QuantumSimulation.ToyModels import BH1D as bh1d
from QuantumSimulation.Utility.Optim import Learner, pFunc_base
import copy

optim_type = 'BO2'

# Create a model
fom = ['f2t2:neg_fluence:0.0001']
T=10
dico_simul = {'L':5, 'Nb':5, 'mu':0, 'T':T, 'dt':0.01, 'flag_intermediate':False, 
              'setup':'1', 'state_init':'GS_i', 'state_tgt':'GS_inf', 'fom':fom, 
              'fom_print':True, 'track_learning': True, 'ctl_shortcut':'owbds01_crab4'}
dico_simul = learner1DBH._process_controler(dico_simul)
dico_simul['control_obj'] = learner1DBH._build_control_from_string(dico_simul['control_obj'], None, context_dico = dico_simul)
model = bh1d.BH1D(**dico_simul)


if(optim_type == 'BO2'):
    #BO
    optim_args = {'algo': 'BO2', 'maxiter':200, 'num_cores':6, 'init_obj':15, 'acq':'EI'}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO2 = optim(track_learning=True)
    resBO2['last_func'] = model.control_fun
    print(resBO2)
    res = resBO2
    resBO2['opt_more']

if(optim_type == 'DE'):
    optim_args = {'algo': 'DE', 'popsize':5, 'maxiter':75}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resDE = optim()
    print(resDE)
    res = resDE

if(optim_type == 'BO'):
    optim_args = {'algo': 'BO', 'maxiter':250}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resBO = optim()
    resBO['last_func'] = model.control_fun
    print(resBO)
    res = resBO

if(optim_type == 'NM'):
    #NM
    optim_args = {'algo': 'NM', 'init_obj': 'uniform_-1_1', 'nfev':10000}
    optim = Learner.learner_Opt(model = model, **optim_args)
    resNM = optim()
    print(resNM)
    res = resNM

## Create testing
fom_test = fom + ['f2t2', 'fluence', 'smooth', 'varN']
dico_test = copy.copy(dico_simul)
dico_test['fom']=fom_test
dico_test['track_learning'] = False
model_test = bh1d.BH1D(**dico_test)
optim_params = resBO2['params']
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



## Looking at convergenge
if(optim_type == 'BO2'):
    import matplotlib.pylab as plt
    def dist(x, y):
        d = np.squeeze(x)-np.squeeze(y)
        return np.dot(d, d)
            
    def get_dist_successive(X, n_ev = None):
        distance = [dist(x_n, X[n-1]) for n, x_n in enumerate(X[1:])]
        
        if(n_ev is None):
            n_ev = np.arange(1, len(X)+1)
        return [np.array([0]+n_ev), np.array([0]+distance)]
        
    def get_best_so_far(Y, n_ev=None):
        best_tmp = np.Inf
        n_best = []
        y_best = []
        if(n_ev is None):
            n_ev = np.arange(1, len(Y)+1)
        for n, y_n in enumerate(Y):
            if (y_n < best_tmp):
                best_tmp = y_n
                n_best.append(n_ev[n])
                y_best.append(y_n)
        return [np.array(n_best), np.array(y_best)]
    
    def study_convergence(X, Y, end = 0, beg = 0):
        if(len(X) != len(Y)):
            SystemError("X and Y should have the same length")
        nb_obs = len(X)
        nev, dist = get_dist_successive(X)
        nevbest, Ybest = get_best_so_far(Y)
        distbest = dist[np.array([n in nevbest for n in nev])]
        fig, ax = plt.subplots()
        plt.plot(nev, dist, 'b')
        plt.scatter(nevbest, distbest, color = 'b')
        ax.axvspan(0, beg, alpha=0.5, color='grey')
        ax.axvspan(nb_obs-end, nb_obs, alpha=0.5, color='green')

    X_tmp = resBO2['opt_more']['X_evol']
    Y_tmp = resBO2['opt_more']['Y_evol']
    beg = optim_args['init_obj']
       
    study_convergence(X_tmp, Y_tmp, beg=beg, end = 15)

