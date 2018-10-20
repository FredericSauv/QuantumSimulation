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
# MODEL: restricted qubits with projective measurement
# |phi(x)> = sin(x) |1> + cos(x) |0>
### ============================================================ ###
def proba(x, noise=0):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    return np.square(np.sin(x_noise))

def measure(x, underlying_p = proba, nb_measures = 1, verbose = False):
    p = underlying_p(x)
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



### ============================================================ ###
# Regression task
# 1- Simple GP with heteroskedastic noise - onem
# 2- Use of a bernouilli Likelihood
### ============================================================ ###
# ------------------------------------------ #
# 0 - preparation: data + kernels
# ------------------------------------------ #
nb_measures = 5
x_range = (0, np.pi)
nb_init = 20
x_test = np.linspace(*x_range)[:, np.newaxis]
p_test = proba(x_test)
x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = measure(x_init, nb_measures = nb_measures)

plt.plot(x_test, p_test, label = 'real proba')
plt.scatter(x_init, y_init, label='one shot')
plt.legend()

#Same kernel for all
k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_binomial = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)

# ------------------------------------------ #
# 1 Classic set-up
# ------------------------------------------ #
m_reg = GPy.models.GPRegression(X = x_init, Y=y_init, kernel=k)
m_reg.optimize_restarts(num_restarts = 10)
yp_reg, _ = m_reg.predict(x_test)
error_reg = squerror(yp_reg, p_test)
print(error_reg)
m_reg.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.legend()


# ------------------------------------------ #
# 2 Binomial likelihood
# ------------------------------------------ #
i_meth = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Binomial()
Y_metadata={'trials':np.ones_like(y_init) * nb_measures}
m_classi = GPy.core.GP(X=x_init, Y=y_init * nb_measures, kernel=k_binomial, 
                inference_method=i_meth, likelihood=lik, Y_metadata = Y_metadata)
_ = m_classi.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters


y2, v2 = m_classi._raw_predict(x_test) ## Before link
y3, v3 = m_classi.predict_noiseless(x_test) ## same
y4 = m_classi.likelihood.gp_link.transf(y3) 

y = y4
v = v2

plt.plot(x_test, y)
plt.plot(x_test, y+1.96*v)
plt.plot(x_test, y-1.96*v)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.scatter(x_init, y_init, label = 'obs')
plt.ylim([0.0,1.3])
plt.legend()

error_reg = squerror(y4, p_test)
print(error_reg)

m_classi.plot(predict_kw = {'Y_metadata':{'trials':nb_measures}})
m_classi.plot_f()

plt.plot(x_test, nb_measures*p_test, 'r', label = 'real p')

#### WHAT DO WE SEE
#### WHAT ABOUT THE VARIANCE
plt.plot(x_test, y, label = 'binomial')
plt.plot(x_test, yp_reg, label= 'classical')
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.scatter(x_init, y_init, label = 'obs')
plt.legend()






### ============================================================ ###
# BINOMIAL OBSERVATIONS
# 1
# 2- Simple GP 
# 3- Binomial Likelihood
### ============================================================ ###
# ------------------------------------------ #
# 0 Preparation
# ------------------------------------------ #
bounds_bo = [{'name': '0', 'type': 'continuous', 'domain': x_range}]                    
args_EI = {'acquisition_type':'EI', 'domain': bounds_bo, 'optim_num_anchor':15, 
           'optim_num_samples':10000} 
args_LCB = {'acquisition_type':'LCB', 'domain': bounds_bo, 'optim_num_anchor':15, 
           'optim_num_samples':10000, 'acquisition_weight':4,
           'acquisition_weight_lindec':True} 

def get_bests_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y and best expt for
    X already visited"""
    y_seen = np.min(bo.Y)
    x_seen = bo.X[np.argmin(bo.Y)]
    Y_pred = bo.model.predict(bo.X)
    y_exp = np.min(Y_pred[0])
    x_exp = bo.X[np.argmin(Y_pred[0])]
    
    return (x_seen, y_seen), (x_exp, y_exp)


nb_iter = 16
nb_measures = 3
nb_init = 17

cost = lambda x: nb_measures * (1- measure(x, nb_measures = nb_measures, verbose = True))
cost_gaussian = lambda x: 1 - measure(x, nb_measures = 0.1, verbose = True)

x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = cost(x_init)
y_init_gaussian = cost_gaussian(x_init)
args_EI['X'] = x_init
args_LCB['X'] = x_init
args_EI['Y'] = y_init
args_LCB['Y'] = y_init
args_EI_bin = copy.copy(args_EI)
args_EI_bin.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bin = copy.copy(args_LCB)
args_LCB_bin.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bin['acquisition_weight'] = 2
args_EI_gaussian = copy.copy(args_EI)
args_EI_gaussian['Y'] = y_init_gaussian
args_LCB_gaussian = copy.copy(args_LCB)
args_LCB_gaussian['Y'] = y_init_gaussian




# ------------------------------------------ #
# 1 Gaussian Noise
# ------------------------------------------ #
BO_EI_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_EI_gaussian)
BO_EI_gaussian.run_optimization(max_iter = nb_iter)
xy0_EI_gaussian, xy1_EI_gaussian = get_bests_from_BO(BO_EI_gaussian)
y0_EI_gaussian_test = measure(xy0_EI_gaussian[0], nb_measures = np.inf, verbose = True)
y1_EI_gaussian_test = measure(xy1_EI_gaussian[0], nb_measures = np.inf, verbose = True)
BO_EI_gaussian.model.model.plot()


BO_LCB_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_LCB_gaussian)
BO_LCB_gaussian.run_optimization(max_iter = nb_iter)
xy0_LCB_gaussian, xy1_LCB_gaussian = get_bests_from_BO(BO_LCB_gaussian)
y0_LCB_gaussian_test = measure(xy0_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
y1_LCB_gaussian_test = measure(xy1_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
BO_LCB_gaussian.model.model.plot()


# ------------------------------------------ #
# 2 Binomial obs with a classical BO
# ------------------------------------------ #
BO_EI = GPyOpt.methods.BayesianOptimization(cost, **args_EI)
BO_EI.run_optimization(max_iter = nb_iter)
xy0_EI, xy1_EI = get_bests_from_BO(BO_EI)
y0_EI_test = measure(xy0_EI[0], nb_measures = np.inf, verbose = True)
y1_EI_test = measure(xy1_EI[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)

BO_LCB = GPyOpt.methods.BayesianOptimization(cost, **args_LCB)
BO_LCB.run_optimization(max_iter = nb_iter)
xy0_LCB, xy1_LCB = get_bests_from_BO(BO_LCB)
y0_LCB_test = measure(xy0_LCB[0], nb_measures = np.inf, verbose = True)
y1_LCB_test = measure(xy1_LCB[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)


# ------------------------------------------ #
# 2 Binomial obs with a binomial BO
# ------------------------------------------ #
BO_EI_bin = GPyOpt.methods.BayesianOptimization(cost, **args_EI_bin)
BO_EI_bin.run_optimization(max_iter = nb_iter)
xy0_EI_bin, xy1_EI_bin = get_bests_from_BO(BO_EI_bin)
y0_EI_bin_test = measure(xy0_EI_bin[0], nb_measures = np.inf, verbose = True)
y1_EI_bin_test = measure(xy1_EI_bin[0], nb_measures = np.inf, verbose = True)
BO_EI_bin.model.model.plot(lower=10, upper=90, predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bin.model.model.plot_f()

BO_LCB_bin = GPyOpt.methods.BayesianOptimization(cost, **args_LCB_bin)
BO_LCB_bin.run_optimization(max_iter = nb_iter)
xy0_LCB_bin, xy1_LCB_bin = get_bests_from_BO(BO_LCB_bin)
y0_LCB_bin_test = measure(xy0_LCB_bin[0], nb_measures = np.inf, verbose = True)
y1_LCB_bin_test = measure(xy1_LCB_bin[0], nb_measures = np.inf, verbose = True)
BO_EI_bin.model.model.plot(predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bin.model.model.plot_f()





### ============================================================ ###
# BERNOUILLI OBSERVATIONS
# 1
# 2- Simple GP 
# 3- Binomial Likelihood
### ============================================================ ###
# ------------------------------------------ #
# 0 Preparation
# ------------------------------------------ #
nb_iter = 50
nb_measures = 1
nb_init = 50

cost = lambda x: nb_measures * (1- measure(x, nb_measures = nb_measures, verbose = True))
cost_gaussian = lambda x: 1 - measure(x, nb_measures = 0.1, verbose = True)

x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = cost(x_init)
y_init_gaussian = cost_gaussian(x_init)
args_EI['X'] = x_init
args_LCB['X'] = x_init
args_EI['Y'] = y_init
args_LCB['Y'] = y_init
args_EI_bern = copy.copy(args_EI)
args_EI_bern.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bern = copy.copy(args_LCB)
args_LCB_bern.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bern['acquisition_weight'] = 2
args_EI_gaussian = copy.copy(args_EI)
args_EI_gaussian['Y'] = y_init_gaussian
args_LCB_gaussian = copy.copy(args_LCB)
args_LCB_gaussian['Y'] = y_init_gaussian




# ------------------------------------------ #
# 1 Gaussian Noise
# ------------------------------------------ #
BO_EI_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_EI_gaussian)
BO_EI_gaussian.run_optimization(max_iter = nb_iter)
xy0_EI_gaussian, xy1_EI_gaussian = get_bests_from_BO(BO_EI_gaussian)
y0_EI_gaussian_test = measure(xy0_EI_gaussian[0], nb_measures = np.inf, verbose = True)
y1_EI_gaussian_test = measure(xy1_EI_gaussian[0], nb_measures = np.inf, verbose = True)
BO_EI_gaussian.model.model.plot()


BO_LCB_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_LCB_gaussian)
BO_LCB_gaussian.run_optimization(max_iter = nb_iter)
xy0_LCB_gaussian, xy1_LCB_gaussian = get_bests_from_BO(BO_LCB_gaussian)
y0_LCB_gaussian_test = measure(xy0_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
y1_LCB_gaussian_test = measure(xy1_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
BO_LCB_gaussian.model.model.plot()


# ------------------------------------------ #
# 2 bernomial obs with a classical BO
# ------------------------------------------ #
BO_EI = GPyOpt.methods.BayesianOptimization(cost, **args_EI)
BO_EI.run_optimization(max_iter = nb_iter)
xy0_EI, xy1_EI = get_bests_from_BO(BO_EI)
y0_EI_test = measure(xy0_EI[0], nb_measures = np.inf, verbose = True)
y1_EI_test = measure(xy1_EI[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)

BO_LCB = GPyOpt.methods.BayesianOptimization(cost, **args_LCB)
BO_LCB.run_optimization(max_iter = nb_iter)
xy0_LCB, xy1_LCB = get_bests_from_BO(BO_LCB)
y0_LCB_test = measure(xy0_LCB[0], nb_measures = np.inf, verbose = True)
y1_LCB_test = measure(xy1_LCB[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)


# ------------------------------------------ #
# 2 bernomial obs with a bernomial BO
# ------------------------------------------ #
BO_EI_bern = GPyOpt.methods.BayesianOptimization(cost, **args_EI_bern)
BO_EI_bern.run_optimization(max_iter = nb_iter)
xy0_EI_bern, xy1_EI_bern = get_bests_from_BO(BO_EI_bern)
y0_EI_bern_test = measure(xy0_EI_bern[0], nb_measures = np.inf, verbose = True)
y1_EI_bern_test = measure(xy1_EI_bern[0], nb_measures = np.inf, verbose = True)
BO_EI_bern.model.model.plot(lower=10, upper=90, predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bern.model.model.plot_f()

BO_LCB_bern = GPyOpt.methods.BayesianOptimization(cost, **args_LCB_bern)
BO_LCB_bern.run_optimization(max_iter = nb_iter)
xy0_LCB_bern, xy1_LCB_bern = get_bests_from_BO(BO_LCB_bern)
y0_LCB_bern_test = measure(xy0_LCB_bern[0], nb_measures = np.inf, verbose = True)
y1_LCB_bern_test = measure(xy1_LCB_bern[0], nb_measures = np.inf, verbose = True)
BO_EI_bern.model.model.plot(predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bern.model.model.plot_f()



### ============================================================ ###
# Binomial with a l9ot
# 1
# 2- Simple GP 
# 3- Binomial Likelihood
### ============================================================ ###
# ------------------------------------------ #
# 0 Preparation
# ------------------------------------------ #
nb_iter = 16
nb_measures = 1000
nb_init = 84

cost = lambda x: nb_measures * (1- measure(x, nb_measures = nb_measures, verbose = True))
cost_gaussian = lambda x: 1 - measure(x, nb_measures = 0.1, verbose = True)

x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
y_init = cost(x_init)
y_init_gaussian = cost_gaussian(x_init)
args_EI['X'] = x_init
args_LCB['X'] = x_init
args_EI['Y'] = y_init
args_LCB['Y'] = y_init
args_EI_bern = copy.copy(args_EI)
args_EI_bern.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bern = copy.copy(args_LCB)
args_LCB_bern.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(nb_measures), 'normalize_Y':False})
args_LCB_bern['acquisition_weight'] = 2
args_EI_gaussian = copy.copy(args_EI)
args_EI_gaussian['Y'] = y_init_gaussian
args_LCB_gaussian = copy.copy(args_LCB)
args_LCB_gaussian['Y'] = y_init_gaussian




# ------------------------------------------ #
# 1 Gaussian Noise
# ------------------------------------------ #
BO_EI_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_EI_gaussian)
BO_EI_gaussian.run_optimization(max_iter = nb_iter)
xy0_EI_gaussian, xy1_EI_gaussian = get_bests_from_BO(BO_EI_gaussian)
y0_EI_gaussian_test = measure(xy0_EI_gaussian[0], nb_measures = np.inf, verbose = True)
y1_EI_gaussian_test = measure(xy1_EI_gaussian[0], nb_measures = np.inf, verbose = True)
BO_EI_gaussian.model.model.plot()


BO_LCB_gaussian = GPyOpt.methods.BayesianOptimization(cost_gaussian, **args_LCB_gaussian)
BO_LCB_gaussian.run_optimization(max_iter = nb_iter)
xy0_LCB_gaussian, xy1_LCB_gaussian = get_bests_from_BO(BO_LCB_gaussian)
y0_LCB_gaussian_test = measure(xy0_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
y1_LCB_gaussian_test = measure(xy1_LCB_gaussian[0], nb_measures = np.inf, verbose = True)
BO_LCB_gaussian.model.model.plot()


# ------------------------------------------ #
# 2 bernomial obs with a classical BO
# ------------------------------------------ #
BO_EI = GPyOpt.methods.BayesianOptimization(cost, **args_EI)
BO_EI.run_optimization(max_iter = nb_iter)
xy0_EI, xy1_EI = get_bests_from_BO(BO_EI)
y0_EI_test = measure(xy0_EI[0], nb_measures = np.inf, verbose = True)
y1_EI_test = measure(xy1_EI[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)

BO_LCB = GPyOpt.methods.BayesianOptimization(cost, **args_LCB)
BO_LCB.run_optimization(max_iter = nb_iter)
xy0_LCB, xy1_LCB = get_bests_from_BO(BO_LCB)
y0_LCB_test = measure(xy0_LCB[0], nb_measures = np.inf, verbose = True)
y1_LCB_test = measure(xy1_LCB[0], nb_measures = np.inf, verbose = True)
BO_EI.model.model.plot(resolution = 2000)


# ------------------------------------------ #
# 2 bernomial obs with a bernomial BO
# ------------------------------------------ #
BO_EI_bern = GPyOpt.methods.BayesianOptimization(cost, **args_EI_bern)
BO_EI_bern.run_optimization(max_iter = nb_iter)
xy0_EI_bern, xy1_EI_bern = get_bests_from_BO(BO_EI_bern)
y0_EI_bern_test = measure(xy0_EI_bern[0], nb_measures = np.inf, verbose = True)
y1_EI_bern_test = measure(xy1_EI_bern[0], nb_measures = np.inf, verbose = True)
BO_EI_bern.model.model.plot(lower=10, upper=90, predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bern.model.model.plot_f()

BO_LCB_bern = GPyOpt.methods.BayesianOptimization(cost, **args_LCB_bern)
BO_LCB_bern.run_optimization(max_iter = nb_iter)
xy0_LCB_bern, xy1_LCB_bern = get_bests_from_BO(BO_LCB_bern)
y0_LCB_bern_test = measure(xy0_LCB_bern[0], nb_measures = np.inf, verbose = True)
y1_LCB_bern_test = measure(xy1_LCB_bern[0], nb_measures = np.inf, verbose = True)
BO_EI_bern.model.model.plot(predict_kw = {'Y_metadata':{'trials':nb_measures}})
BO_EI_bern.model.model.plot_f()



