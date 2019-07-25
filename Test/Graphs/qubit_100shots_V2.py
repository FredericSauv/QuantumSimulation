#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import copy , sys
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
sys.path.append('/home/fred/Desktop/GPyOpt/')
import GPyOpt


### ============================================================ ###
# Preparation:
### ============================================================ ###
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
        x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
        p = np.square(np.sin(x_noise))
        #p = np.abs(np.sin(n * x_noise) * np.exp(- np.square((x_noise-np.pi/2) / s)))
    elif model == 2:
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
                     'optim_num_anchor':15, 'optim_num_samples':10000, 'maximize':True} 
    elif type_acq == 'LCB':
        base_dico = {'acquisition_type':'LCB', 'domain': bounds_bo, 
                     'optim_num_anchor':15, 'optim_num_samples':10000, 
                     'acquisition_weight':4, 'acquisition_weight_lindec':True,'maximize':True} 
    
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

### ============================================================ ###
# Main results
### ============================================================ ###
x_range = (0, np.pi)
# ------------------------------------------ #
# models to simulate
# ------------------------------------------ #
verbose = True

## perfect measurement
f1_perfect = lambda x: measure(x, nb_measures = np.inf, verbose = verbose, model = 1, noise_input = 0)
f1_perfect_neg = lambda x: measure(x, nb_measures = np.inf, verbose = verbose, model = 1, noise_input = 0)
f1_gaussian = lambda x: measure(x, nb_measures = 0.2, verbose = verbose, model = 1, noise_input = 0)
f1_bin1 = lambda x: measure(x, nb_measures = 1, verbose = verbose, model = 1, noise_input = 0)

# model, nb_measures, model_test, nb_init, nb_iter, name
list_to_simul_f1_1 = [(f1_bin1, 1, f1_perfect, 100, 50, 'f1_bin1')] #OK
list_to_simul_f1_perfect = [(f1_perfect, np.inf, f1_perfect, 1000, 85, 'f1_perfect')]

nb_repetitions = 10
dico_res = {}


# ------------------------------------------ #
# Interlude PLOTTING
# ------------------------------------------ #
#model, nb_measures, model_test, nb_init, nb_iter, name = list_to_simul_f1_perfect[0]
#res = np.zeros((2))
#x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
#y_init = model(x_init)

#test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='gaussian', 
#        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
#        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)


model, nb_measures, model_test, nb_init, nb_iter, name = list_to_simul_f1_1[0]
res = np.zeros((2))
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = model(x_init)

test, test_exp, BO = do_one_BO_optim(type_acq = 'EI', type_gp='binomial', 
        X_init = x_init, Y_init = y_init, cost = model, cost_test = model_test, 
        x_range = (0, np.pi), nb_iter = nb_iter, nb_measures = nb_measures)


x_test = np.linspace(0, np.pi, 1000)[:, np.newaxis]
y_test = f1_perfect(x_test)
BO.model.model.plot(lower = 15, upper = 85, predict_kw = {'Y_metadata':{'trials':nb_measures}}, plot_density=True)
plt.plot(x_test, 1 - y_test, 'r--', label = '1 - p')
plt.xlim([0, np.pi])
plt.ylim([-0.05, 1.05])
plt.legend()
#plt.savefig('gaussian_likelihoop_optim.pdf', bbox_inches = 'tight')



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
    
    plt.plot(x, m)
    plt.scatter(xd, yd, c='black', marker='x')

plot_mean(BO.model.model,transfo = lambda x: 1-x, predict_kw = {'Y_metadata':{'trials':nb_measures}})
plt.plot(x_test, y_test, 'r--', label = '1 - p')
plt.xlim([0, np.pi])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.savefig('gaussian_likelihoop_optim.pdf', bbox_inches = 'tight')



# ------------------------------------------ #
# GOGOGOGOGOGOGOGOGGOGOGHOGGOOG
# ------------------------------------------ #

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
