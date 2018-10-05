#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
import sys, copy
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

def measure(x, underlying_p = proba, nb_measures = 1, verbose=True):
    real_p = underlying_p(x)
    if(nb_measures == np.inf):
        res = real_p
    else:
        res = rdm.binomial(nb_measures, underlying_p(x)) / nb_measures
    if(verbose):
        print(res)
        print(x)
    return res

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)


### ============================================================ ###
# Data training / test
### ============================================================ ###
x_range = (0, np.pi)
nb_meas = 20
x_best = np.pi/2
nb_params = 1
#init
nb_measures_init = 100
nb_x_init_onem = nb_measures_init
nb_x_init_nm = int(nb_measures_init / nb_meas)
x_init_onem = rdm.uniform(*x_range, nb_x_init_onem)[:, np.newaxis]
y_init_onem = measure(x_init_onem)
x_init_nm = rdm.uniform(*x_range, nb_x_init_nm)[:, np.newaxis]
y_init_nm = measure(x_init_nm, nb_measures = nb_meas)
cost_onem = measure
cost_nm = lambda x: measure(x, nb_measures = nb_meas)
#iter
nb_iter_onem = 900
nb_iter_nm = int(nb_iter_onem / nb_meas)


### ============================================================ ###
# Regression task
# 1- Simple GP with heteroskedastic noise - onem
# 1b- Simple GP with heteroskedastic noise - nm
# 2- Learn warping - onem
# 2b- Learn warping - nm
# 3- As a classification task
### ============================================================ ###
#Same kernel for all
k_one = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_warp_one = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_warp_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_classi_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)



acq = 'EI'
name_params = [str(i) for i in range(nb_params)]
bounds_bo = [{'name': name_params[i], 'type': 'continuous', 'domain': x_range} for i in range(nb_params)]                    
args_BO = {'acquisition_type':'EI', 'domain': bounds_bo, 'optim_num_anchor':25, 'optim_num_samples':10000} 

args_BO_ber = copy.copy(args_BO)
args_BO_ber.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Bernouilli', 'initial_design_numdata':50, 'normalize_Y':False})
args_BO_bin = copy.copy(args_BO)
args_BO_bin.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_'+str(nb_meas),'initial_design_numdata':50, 'normalize_Y':False})


def get_best_seen_from_BO(bo):
    """ From BO optimization extract X giving the best seen Y"""
    return bo.X[np.argmin(bo.Y)]
    
def get_best_exp_from_BO(bo):
    """ From BO optimization extract X giving the best predicted Y (only amongst 
    the tested values)"""
    Y_pred = bo.model.predict(bo.X)
    return bo.X[np.argmin(Y_pred[0])]


#Initialization phase
#BO_binomial
cost_bin = lambda x: 1- measure(x, nb_measures = nb_meas)
BO_bin = GPyOpt.methods.BayesianOptimization(cost_bin, **args_BO_bin)
BO_bin.run_optimization(max_iter = 100)
get_best_exp_from_BO(BO_bin)

x_test = np.linspace(0,np.pi)[:, np.newaxis]
m = BO_bin.model.model
yp_classi_n_tmp, _ = m.predict_noiseless(x_test)
yp_classi_n = m.likelihood.gp_link.transf(yp_classi_n_tmp) 

plt.plot(x_test, yp_classi_n)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])
error_classi_n = squerror(yp_classi_n, p_test)
m_classi_n.plot_f()




## BO simple
cost = lambda x: - measure(x, nb_measures = 100)
BO = GPyOpt.methods.BayesianOptimization(cost, **args_BO)
get_best_seen_fom_BO(BO)


#BO_bernouilli
cost_ber = lambda x: 1 - measure(x)
BO_ber = GPyOpt.methods.BayesianOptimization(cost_ber, **args_BO_ber)
BO_ber.run_optimization(max_iter = 1000)
get_best_exp_from_BO(BO_ber)





# Exploration-Exploitation phase
(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', 
                 max_iters=1000, optimize_restarts=5, sparse = False, num_inducing = 10,  
                 verbose=True, ARD=False, likelihood = None, inf_method = None)
bo = GPyOpt.methods.BayesianOptimization(cost, **args_BO)
bo.run_optimization(max_iter = options['maxiter'], max_time = options['max_time'])
        


# ------------------------------------------ #
# 1
# ------------------------------------------ #
m_reg_one = GPy.models.GPRegression(X = x_train_onem, Y=y_train_onem, kernel=k_one)
m_reg_one.optimize_restarts(num_restarts = 10)
yp_reg_one, _ = m_reg_one.predict(x_test)
error_reg_one = squerror(yp_reg_one, p_test)
print(error_reg_one)
m_reg_one.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])



m_reg_n = GPy.models.GPRegression(X = x_train_nm, Y=y_train_nm, kernel=k_n)
m_reg_n.optimize_restarts(num_restarts = 10)
yp_reg_n, _ = m_reg_n.predict(x_test)
error_reg_n = squerror(yp_reg_n, p_test)
print(error_reg_n)
m_reg_n.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])



# 3a Bernouilly
i_meth = GPy.inference.latent_function_inference.expectation_propagation.EP()
lik = GPy.likelihoods.Bernoulli()
m_classi = GPy.core.GP(X=x_train_onem, Y=y_train_onem, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik)
_ = m_classi.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters

yp_classi, _ = m_classi.predict(x_test)
test, _ = m_classi.predict_noiseless(x_test)
plt.plot(x_test, test)


error_classi = squerror(yp_classi, p_test)
print(error_classi)

m_classi.plot_f()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])



# 3b Binomial
i_meth_LP = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Binomial()
Y_metadata={'trials':np.ones_like(y_train_nm) * nb_meas}
m_classi_n = GPy.core.GP(X=x_train_nm, Y=y_train_nm * nb_meas, kernel=k_classi_n, 
                inference_method=i_meth_LP, likelihood=lik, Y_metadata=Y_metadata)
_ = m_classi_n.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters

yp_classi_n_tmp, _ = m_classi_n.predict_noiseless(x_test)
yp_classi_n = m_classi_n.likelihood.gp_link.transf(yp_classi_n_tmp) 

plt.plot(x_test, yp_classi_n)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])
error_classi_n = squerror(yp_classi_n, p_test)
print(error_classi_n)






### ============================================================ ###
# Testingg
### ============================================================ ###
X = (2 * np.pi) * np.random.random(151) - np.pi
Y = np.sin(X) + np.random.normal(0,0.2,151)
Y = np.array([np.power(abs(y),float(1)/3) * (1,-1)[y<0] for y in Y])
X = X[:, None]
Y = Y[:, None]

warp_k = GPy.kern.RBF(1)
warp_f = GPy.util.warping_functions.TanhFunction(n_terms=2)
warp_m = GPy.models.WarpedGP(X, Y, kernel=warp_k, warping_function=warp_f)
warp_m['.*\.d'].constrain_fixed(1.0)
m = GPy.models.GPRegression(X, Y)
m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=10)
warp_m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=10)
#m.optimize(max_iters=max_iters)
#warp_m.optimize(max_iters=max_iters)

print(warp_m)
print(warp_m['.*warp.*'])

warp_m.predict_in_warped_space = False
warp_m.plot(title="Warped GP - Latent space")
warp_m.predict_in_warped_space = True
warp_m.plot(title="Warped GP - Warped space")
m.plot(title="Standard GP")
warp_m.plot_warping()
pb.show()
