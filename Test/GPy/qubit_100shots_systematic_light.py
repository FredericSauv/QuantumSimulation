#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import sys
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
sys.path.insert(0, '/Users/frederic/Desktop/GPyOpt') #Laptop's loc
sys.path.insert(0, '/home/fred/Desktop/WORK/GIT/GPyOpt') #Desktop's loc
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')

import GPyOpt


### ============================================================ ###
# Preparation:
### ============================================================ ###
x_range = (0, np.pi)
def proba(x, noise=0, model = 0):
    #generate underlying proba p(|1>)
    n = 3
    s = 2
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    if model == 0:
        p = np.square(np.sin(x))
    elif model == 1:    
        p = np.abs(np.sin(n * x_noise) * np.exp(- np.square((x_noise-np.pi/2) / s)))
    else:
        raise NotImplementedError()
    return p

def measure(x, nb_measures = 1, verbose = False, model = 0, noise_input = 0):
    p = proba(x = x, noise= noise_input, model = model)
    if(nb_measures == np.inf):
        res = p
    elif(nb_measures < 1):
        res = p +np.random.normal(0, nb_measures, size = np.shape(p))
    else:
        res = rdm.binomial(nb_measures, p)/nb_measures
        
    if(verbose):
        print(np.around(np.c_[res, p, x], 3))
    return res

def c_model(m=np.inf, mod=0, noise = 0):
    """ Create a model"""
    if(m == np.inf):
        model = lambda x: measure(x, nb_measures = m, verbose = True, model = mod, noise_input = noise)
    else:
        model = lambda x: m * (1 - measure(x, nb_measures = m, verbose = True, model = mod, noise_input = noise))
    return model

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)

def get_bests_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y and best expt for
    X already visited"""
    y_seen = np.min(bo.Y)
    x_seen = bo.X[np.argmin(bo.Y)]
    Y_pred = bo.model.predict(bo.X)
    y_exp = np.min(Y_pred[0])
    x_exp = bo.X[np.argmin(Y_pred[0])]
    
    return (x_seen, y_seen), (x_exp, y_exp)

def do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', X_init = None, Y_init = None, 
                    cost = None, cost_test = None, x_range = (0, np.pi), 
                    nb_iter = 50, nb_measures = None):
    """ Wrap up a BO optimization 
    """
    # basis parameter of the BO optimizer
    bounds_bo = [{'name': '0', 'type': 'continuous', 'domain': x_range}]
    if type_acq == 'EI':
        base_dico = {'acquisition_type':'EI', 'domain': bounds_bo, 
                     'optim_num_anchor':15, 'optim_num_samples':10000} 
    elif type_acq == 'LCB':
        base_dico = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                     'optim_num_anchor':15, 'optim_num_samples':10000, 
                     'acquisition_weight':5, 'acquisition_weight_lindec':True} 
    
    base_dico['X'] = X_init
    base_dico['Y'] = Y_init

    if type_gp == 'binomial':
        base_dico.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 
                            'likelihood':'Binomial_' + str(nb_measures), 
                            'normalize_Y':False})

    BO = GPyOpt.methods.BayesianOptimization(cost, **base_dico)
    BO.run_optimization(max_iter = nb_iter, eps = 0)
    
    xy, xy_exp = get_bests_from_BO(BO)
    test = cost_test(xy[0])
    test_exp = cost_test(xy_exp[0])
    
    return test, test_exp, BO

def test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures):
    """ Run four flavors of BO Gaussian/Binomial likelihood x EI/LCB"""
    res = np.zeros((2))
    x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
    y_init = model(x_init)
    
    test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)

    res[0] = test_exp
    
    if(nb_measures != np.inf):
        test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='binomial', 
            X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
            x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)
    
        res[1] = test_exp
             
    else:
        na = 0
        res[1] = na
    
    return res



def plot_mean(model, transfo = lambda x: x, predict_kw = None):
    """self, Xgrid, plot_raw, apply_link, percentiles, which_data_ycols, predict_kw, samples=0):
    """
    Xgrid = np.linspace(0, np.pi, 200)[:, np.newaxis]
    # Put some standards into the predict_kw so that prediction is done automatically:
    if predict_kw is None:
        predict_kw = {}
    if 'Y_metadata' not in predict_kw:
        predict_kw['Y_metadata'] = {}
    if 'output_index' not in predict_kw['Y_metadata']:
        predict_kw['Y_metadata']['output_index'] = Xgrid[:,-1:].astype(np.int)

    mu, _ = model.predict(Xgrid, **predict_kw)
    X_data, Y_data = model.X, model.Y

    x = np.squeeze(Xgrid)    
    m = np.squeeze(transfo(mu))
    xd = np.squeeze(X_data)
    yd = np.squeeze(transfo(Y_data))
    
    plt.plot(x, m, label = 'proxy')
    plt.scatter(xd, yd, c='black', marker='x', label = 'observations')


def plot_link(model, transfo = lambda x:x, percentiles = None, link = True,
                likelihood = False, n_measures = None):
    Xgrid = np.linspace(0, np.pi, 200)[:, np.newaxis]
    mu_f, var_f = model._raw_predict(Xgrid, full_cov=False, kern=None)
    
    s_f = np.random.randn(mu_f.shape[0], 1000) * np.sqrt(var_f) + mu_f
    s_f = transfo(s_f)
    if(link):
        if(likelihood):
            s_p = model.likelihood(s_f, Y_metadata={'trials':n_measures})
        else:
            s_p = model.likelihood.gp_link.transf(s_f)
    else:
        s_p = s_f
    mu_p = np.mean(s_p, axis = 1)
    if(percentiles is None):
        var_p = np.std(s_p, axis = 1)
        muplus = mu_p + 1.96 * var_p
        muminus = mu_p - 1.96 * var_p
    else:
        muminus, muplus = np.percentile(s_p, percentiles, axis = 1)
    
    X_data, Y_data = model.X, model.Y
    x = np.squeeze(Xgrid)    
    xd = np.squeeze(X_data)
    yd = np.squeeze(transfo(Y_data))
    
    plt.plot(x, mu_p, 'b', label = 'proxy')
    plt.fill_between(x, muminus, muplus, color = 'blue', alpha = 0.5, label = 'confidence')
    plt.scatter(xd, yd, c='black', marker='x', label = 'observations')
    
def get_init(n_init, model, eps=0):
    if(eps == 0):
        x_range = (0, np.pi)
        x = rdm.uniform(*x_range, n_init)[:, np.newaxis]
    else:
        half_init = int(n_init/2)
        x_range_first = (0, np.pi/2 - eps)
        x_range_last = (np.pi/2 + eps, np.pi)
        first = rdm.uniform(*x_range_first, half_init)[:, np.newaxis]
        last = rdm.uniform(*x_range_last, half_init)[:, np.newaxis]
        x = np.r_[first, last]
    y = model(x)
    return x, y

### ============================================================ ###
# Main results
### ============================================================ ###
# ------------------------------------------ #
# models to simulate
# ------------------------------------------ #
f0_perfect = c_model()
f1_perfect = c_model(mod = 1)
# model, nb_measures, model_test, nb_init, nb_iter, name
list_to_simul_f0_1 = [(c_model(m=1), 1, f0_perfect, 50, 50, 'f0_bin1')] # OK
list_to_simul_f0_3 = [(c_model(m=3), 3, f0_perfect, 16, 17, 'f0_bin3')] # OK                
list_to_simul_f0_1_noise_5 = [(c_model(m=1, noise=0.05), 1, f0_perfect, 50, 50, 'f0_bin1_noise5')] # OK
list_to_simul_f0_3_noise_5 = [(c_model(m=3, noise=0.05), 3, f0_perfect, 16, 17, 'f0_bin3_noise5')] # OK
list_to_simul_f0_1_noise_10 = [(c_model(m=1, noise=0.1), 1, f0_perfect, 50, 50, 'f0_bin1_noise10')]
list_to_simul_f1_1 = [(c_model(m=1, mod=1), 1, f1_perfect, 50, 50, 'f1_bin1')] #OK
list_to_simul_f1_3 = [(c_model(m=3, mod=1), 3, f1_perfect, 16, 17, 'f1_bin3')] #ok
list_to_simul_f0_50 = [(c_model(m=50), 50, f0_perfect, 16, 84, 'f0_bin50')]

# ------------------------------------------ #
# Interlude PLOTTING
# ------------------------------------------ #
model, nb_measures, model_test, nb_init, nb_iter, name = list_to_simul_f1_1[0]
#nb_init = 30
#nb_iter = 70
x_init, y_init = get_init(nb_init, model, eps=0.3)
x_plotting = np.linspace(0, np.pi, 200)
y_plotting = model_test(x_plotting)
# plt.scatter(x_init, y_init)
#BEFORE OPTIM
test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = 0, nb_measures = nb_measures)

#plot_mean(BO.model.model,transfo = lambda x: 1-x, predict_kw = {'Y_metadata':{'trials':nb_measures}})
plot_link(BO.model.model, transfo = lambda x: 1-x, percentiles = [5, 95])
plt.plot(x_plotting, y_plotting, 'r--', label = 'p')
plt.xlim([0, np.pi])
plt.ylim([-0.05, 1.05])
plt.legend(loc = 4)
#plt.savefig('nomiddle_before_optim_bin1_a.pdf', bbox_inches = 'tight')



#OPTIM
test, test_exp, BO = do_one_BO_optim(type_acq = 'LCB', type_gp='binomial', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)

#plot_mean(BO.model.model,transfo = lambda x: 1-x, predict_kw = {'Y_metadata':{'trials':nb_measures}})
plot_link(BO.model.model, transfo = lambda x: 1-x, percentiles = [5, 95])
plt.plot(x_plotting, y_plotting, 'r--', label = 'p')
plt.xlim([0, np.pi])
plt.ylim([-0.05, 1.05])
plt.legend(loc = 4)
plt.savefig('nomiddle_after_optim_bin1_a.pdf', bbox_inches = 'tight')
#plt.savefig('after_optim_bin1_c.pdf', bbox_inches = 'tight')



# ------------------------------------------ #
# GOGOGOGOGOGOGOGOGGOGOGHOGGOOG
# ------------------------------------------ #
nb_repetitions = 10
dico_res = {}
# simple rabbi dynamics

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_1:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp
    
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_3:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_1_noise_5:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1_1:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_3_noise_5:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp


for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_perfect:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp    
    
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1_perfect:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp   
    
for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_50:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp   
    
    

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1_3:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f1_1000:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp

for model, nb_measures, model_test, nb_init, nb_iter, name in list_to_simul_f0_1_noise_5:
    res_tmp = np.array([test_two_BOs(x_range, nb_init, model, model_test, nb_iter, nb_measures) for _ in range(nb_repetitions)])   
    dico_res[name] = res_tmp



for k, v in dico_res.items():
    print(k)
    print('avg')
    print(np.average(v, 0))
    print('std')
    print(np.std(v, 0))
    print('min')
    print(np.min(v, 0))
    print('max')
    print(np.max(v, 0))
    
### ============================================================ ###
# Process results get graphs
# 0 - plot data used
# 1 - plot gaussian processes
### ============================================================ ###
# ------------------------------------------ #
# 0 Preparation for f0 et f1
# ------------------------------------------ #


# ------------------------------------------ #
# 1 Gaussian Noisey0_EI_gaussian_test = measure(xy0_EI_gaussian[0], nb_measures = np.inf, verbose = True)
# ------------------------------------------ #


# ------------------------------------------ #
# 2 Binomial obs with a classical BO
# ------------------------------------------ #
