#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:43:25 2019
@author: fred
"""
import pdb
pdb.run("a=3")

import sys
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/')
import GPyOpt
from qutip import sigmax, sigmaz, sigmay, mesolve, Qobj, Options, fidelity
import numpy as np
from numpy import random
import matplotlib.pylab as plt
from scipy.special import erfinv
## TODO
## a. Add BlochSphere capa
## c. wrap f to take only one argument first DONE
## d. create bo_args DONE
## e. debug DONE
## f. implement logic if nothing has changed just return last results (NOT NOW)
## g. find old simulations with target
## h. find a state target DONE
## i. test EI acquisition_function
## j. Incorporate to batch_few_shots
## k. Profiling (Add more timers)
## l. Optim 
## m. bch: average
## e. run it on previous examples
#def run_model(args, phi0, phitgt, T = 1.):

#=======================================#
# HELPER FUNCTIONS
#=======================================#
class qubit_mo:
    """ Simple parametrized qubit model 
    H = alpha *  [beta * Z + (1 - beta^2)^0.5 * Y] """
    def __init__(self, phi_init, phi_tgt, args=[0., 0.]): 
        """  """
        self.options_evolve = Options(store_states=True)
        self.phi_tgt = phi_tgt
        self.phi_init = phi_init
        self.T = 1
        self.set_H([0., 0.])
        self.all_e = [sigmax(), sigmay(), sigmaz()]

    def set_H(self, args):
        """ setup the parametrized Hamiltonian """
        self.args = args        
        alpha = args[0]
        beta = args[1]
        self.H = alpha * beta* sigmaz() + sigmay() * alpha * np.sqrt(1. - np.square(beta))

    def run(self, args, meas = None, N=np.inf, verbose = True):
        """ """
        self.set_H(args)
        if meas is None:
            e = self.all_e
        else:
            e = [self.all_e[meas]]

        
        evol = mesolve(self.H, self.phi_init, tlist = [0., self.T], e_ops=e, options = self.options_evolve)
        self.final_state = evol.states[-1]
        self.final_expect = [e[-1] for e in evol.expect]
        proba = [(1 + e)/2 for e in self.final_expect]
        if(N == np.inf):
            meas = proba
        else:
            meas = random.binomial(N, proba)
        if verbose:
            print(args, meas)
        return meas

    def get_fid(self, args = None, phi_tgt = None):
        if args is not None:
            self.run(args)
        if phi_tgt is not None:
            self.phi_tgt = phi_tgt
        self.fid = np.square(fidelity(self.final_state, self.phi_tgt))
        return self.fid

def probit(p):
    return np.sqrt(2) * erfinv(2 * p -1)



def get_bests_from_BO_mo(bo, target = None):
    """ From BO optimization extract X giving the best seen Y and best expected 
    for Xs already visited"""
    Y_pred = bo.model.get_model_predict()
    target = bo.acquisition  if hasattr(bo.acquisition, 'target') else None
        
    if(target is None):
        y_seen = np.min(bo.Y)
        x_seen = bo.X[np.argmin(bo.Y)]    
        y_exp = np.min(Y_pred[0])
        x_exp = bo.X[np.argmin(Y_pred[0])]
    
    else:
        if(bo.normalize_Y):
            target = (target - bo.Y.mean())/bo.Y.std() 
        y_seen = np.min(np.abs(bo.Y - target))
        x_seen = bo.X[np.argmin(np.abs(bo.Y - target))]
        y_exp = np.min(np.abs(Y_pred[0] - target))
        x_exp = bo.X[np.argmin(np.abs(Y_pred[0] - target))]

    return (x_seen, y_seen), (x_exp, y_exp)

def gen_f(model, N = np.inf, beta = 0):
    def f(X, verbose = True):
        res = [model.run([x[0], beta], N=N, verbose=verbose) for x in X]
        return np.array(res)
    return f

def gen_f_2args(model, N = np.inf):
    def f(X, verbose = True):
        res = [model.run(x, N=N, verbose=verbose) for x in X]
        return np.array(res)
    return f



#=======================================#
# INIT 
#=======================================#
# Create system
phi0 = Qobj(np.array([1., 0.]))
phiT = Qobj(1/np.sqrt(2) * np.array([1., -1.j]))
qb = qubit_mo(phi_init = phi0, phi_tgt = phiT)

args = [np.pi/4, 0.]
qb.run(args)
qb.final_state
qb.final_expect

#Wrap 
f_perfect = gen_f(qb, beta= 0.5)
f2_perfect = gen_f_2args(qb)
f_1s = gen_f(qb, beta= 0.5, N = 1)
f2_1s = gen_f_2args(qb, N=1)

# Domain of the parameters
domain_alpha = (0, np.pi)
domain_beta = (0, 1)



#=======================================#
# MAIN -- 1 args // Perfect measurements
#=======================================#
xtgt = np.array([[np.random.uniform(*domain_alpha)]])
print("ideal x is {}".format(xtgt))
ftgt = np.squeeze(f_perfect(xtgt))
phitgt = qb.final_state
print("ideal p is {}".format(ftgt))

# init
nb_init = 20
X_init = np.random.uniform(*domain_alpha, (nb_init, 1))
Y_init = f_perfect(X_init, verbose = False)
plt.scatter(np.squeeze(X_init), Y_init[:,0], label = 'X')
plt.scatter(np.squeeze(X_init), Y_init[:,1], label = 'Y')
plt.scatter(np.squeeze(X_init), Y_init[:,2], label = 'Z')
plt.legend()


## Try to set up optimizer 1 args
# later on binomial
# bo_args.update({'model_type':'GP_CUSTOM_LIK', 'inf_method':'Laplace', 'likelihood':'Binomial_' + str(self.n_meas), 'normalize_Y':False})
# later on EI 
mo_dict = {'output_dim':3, 'rank':0, 'missing':False}
bounds = [{'name': 'alpha', 'type': 'continuous', 'domain': domain_alpha}]
bo_args = {'domain': bounds, 'optim_num_anchor':5, 'optim_num_samples':10000, 
           'acquisition_type':'LCB_target', 'acquisition_weight':4, 
           'acquisition_weight_lindec':True, 'acquisition_ftarget': ftgt,
           'X': X_init, 'Y': Y_init, 'mo':mo_dict}

BO = GPyOpt.methods.BayesianOptimization(f_perfect, **bo_args)
BO.run_optimization(max_iter = 10, eps = 0)
(x0, y0), (x1,y1) = BO.get_best()

qb.get_fid([1.47379736, 0.5], phitgt)




#=======================================#
# MAIN - 2args // Perfect measurements
#=======================================#
xtgt2 = np.array([[np.random.uniform(*domain_alpha), np.random.uniform(*domain_beta)]])
print("ideal x is {}".format(xtgt2))
tgt2 = np.squeeze(f2_perfect(xtgt2))
phitgt2 = qb.final_state
print("ideal p is {}".format(tgt2))

# tgt = probit(ptgt) ####################### USE IF NON GAUSSIAN LIKELIHOOD

# init
nb_init = 20
X_init2 = np.c_[(np.random.uniform(*domain_alpha, (nb_init, 1)), np.random.uniform(*domain_beta, (nb_init, 1)))]
Y_init2 = f2_perfect(X_init2)

mo_dict2 = {'output_dim':3, 'rank':0, 'missing':False}
bounds2 = [{'name': 'alpha', 'type': 'continuous', 'domain': domain_alpha}, {'name': 'beta', 'type': 'continuous', 'domain': domain_beta}]
bo_args2 = {'domain': bounds2, 'optim_num_anchor':5, 'optim_num_samples':10000, 
           'acquisition_type':'LCB_target', 'acquisition_weight':4, 
           'acquisition_weight_lindec':True, 'acquisition_ftarget': tgt2,
           'X': X_init2, 'Y': Y_init2, 'mo':mo_dict2}

BO2 = GPyOpt.methods.BayesianOptimization(f2_perfect, **bo_args2)
BO2.run_optimization(max_iter = 10, eps = 0)
(x0_2, y0_2), (x1_2,y1_2) = BO2.get_best()
qb.get_fid(x1_2, phitgt2)



#=======================================#
# MAIN - 1arg // Single shot measurement
#=======================================#
xtgt_1s = np.array([[np.random.uniform(*domain_alpha)]])
print("ideal x is {}".format(xtgt_1s))
tgt_1s = np.squeeze(f_perfect(xtgt_1s))
phitgt_1s = qb.final_state
print("ideal p is {}".format(tgt_1s))
f_tgt_1s = probit(tgt_1s) ####################### USE IF NON GAUSSIAN LIKELIHOOD


# init
nb_init = 50
X_init_1s = np.c_[(np.random.uniform(*domain_alpha, (nb_init, 1)))]
Y_init_1s = f_1s(X_init_1s, verbose = False)
mo_dict_1s = {'output_dim':3, 'rank':0, 'missing':False}
bounds_1s = [{'name': 'alpha', 'type': 'continuous', 'domain': domain_alpha}]
bo_args_1s = {'domain': bounds_1s, 'optim_num_anchor':5, 'optim_num_samples':10000, 
           'acquisition_type':'LCB_target', 'acquisition_weight':2, 
           'acquisition_weight_lindec':True, 'acquisition_ftarget': f_tgt_1s,
           'X': X_init_1s, 'Y': Y_init_1s, 'mo':mo_dict_1s, 'model_type':'GP_CUSTOM_LIK', 
           'inf_method':'Laplace', 'likelihood':'Binomial_' + str(1), 'normalize_Y':False}


BO2 = GPyOpt.methods.BayesianOptimization(f_1s, **bo_args_1s)
BO2.run_optimization(max_iter = 10, eps = 0)
(x0_2, y0_2), (x1_2,y1_2) = BO2.get_best()
qb.get_fid(x1_2, phitgt2)


qb.get_fid([1.9592188905711554, 0.5], phitgt_1s)

# Try to do some plotting:
x_plot = np.linspace(*domain_alpha, 1000)[:,np.newaxis]
y_plot = f_perfect(x_plot, verbose = False)
mu_plot, sd_plot = BO2.model.predict(x_plot)
mu_msd_plot = mu_plot - sd_plot
mu_psd_plot = mu_plot + sd_plot
fy_plot = probit(y_plot)


mu_p_plot = BO2.model.model.likelihood.gp_link.transf(mu_plot)
mu_p_msd_plot = BO2.model.model.likelihood.gp_link.transf(mu_msd_plot)
mu_p_psd_plot = BO2.model.model.likelihood.gp_link.transf(mu_psd_plot)

labels = ['X', 'Y', 'Z']

## In p space
for i in range(3):
    plt.figure()
    plt.plot(x_plot, y_plot[:, i], 'r')
    plt.plot(x_plot, mu_p_plot[:, i], 'b--')
    plt.fill_between(np.squeeze(x_plot), mu_p_msd_plot[:, i], mu_p_psd_plot[:, i], alpha= 0.2, facecolor='blue')
    plt.hlines(tgt_1s[i], *domain_alpha)
    plt.title(labels[i] + 'pspace')

#in f space
for i in range(3):
    plt.figure()
    plt.plot(x_plot, fy_plot[:, i], 'r')
    plt.plot(x_plot, mu_plot[:, i], 'b--')
    plt.fill_between(np.squeeze(x_plot), mu_msd_plot[:, i], mu_psd_plot[:, i], alpha= 0.2, facecolor='blue')
    plt.hlines(f_tgt_1s[i], *domain_alpha)
    plt.title(labels[i] + 'fspace')



#=======================================#
# MAIN - 2args // Single shot measurement
#=======================================#
xtgt2_1s = np.array([[np.random.uniform(*domain_alpha), np.random.uniform(*domain_beta)]])
print("ideal x is {}".format(xtgt2_1s))
tgt2_1s = np.squeeze(f2_perfect(xtgt2_1s))
phitgt2_1s = qb.final_state
print("ideal p is {}".format(tgt2_1s))

f_tgt2_1s = probit(tgt2_1s) ####################### USE IF NON GAUSSIAN LIKELIHOOD

# init
nb_init = 50
X_init2_1s = np.c_[(np.random.uniform(*domain_alpha, (nb_init, 1)), np.random.uniform(*domain_beta, (nb_init, 1)))]
Y_init2_1s = f2_1s(X_init2_1s, verbose = False)
mo_dict2_1s = {'output_dim':3, 'rank':0, 'missing':False, 'kappa_fix':True}
bounds2_1s = [{'name': 'alpha', 'type': 'continuous', 'domain': domain_alpha}, {'name': 'beta', 'type': 'continuous', 'domain': domain_beta}]
bo_args2_1s = {'domain': bounds2_1s, 'optim_num_anchor':5, 'optim_num_samples':10000, 
           'acquisition_type':'LCB_target', 'acquisition_weight':2, 
           'acquisition_weight_lindec':True, 'acquisition_ftarget': f_tgt2_1s,
           'X': X_init2_1s, 'Y': Y_init2_1s, 'mo':mo_dict2_1s, 'model_type':'GP_CUSTOM_LIK', 
           'inf_method':'Laplace', 'likelihood':'Binomial_' + str(1), 'normalize_Y':False}


BO2 = GPyOpt.methods.BayesianOptimization(f2_1s, **bo_args2_1s)
BO2.run_optimization(max_iter = 10, eps = 0)
(x0_2, y0_2), (x1_2,y1_2) = BO2.get_best()
qb.get_fid(x1_2, phitgt2_1s)



#=======================================#
# MAIN - 2args // Single shot measurement // EI
#=======================================#
xtgt2_1s = np.array([[np.random.uniform(*domain_alpha), np.random.uniform(*domain_beta)]])
print("ideal x is {}".format(xtgt2_1s))
tgt2_1s = np.squeeze(f2_perfect(xtgt2_1s))
phitgt2_1s = qb.final_state
print("ideal p is {}".format(tgt2_1s))

f_tgt2_1s = probit(tgt2_1s) ####################### USE IF NON GAUSSIAN LIKELIHOOD

# init
nb_init = 50
X_init2_1s = np.c_[(np.random.uniform(*domain_alpha, (nb_init, 1)), np.random.uniform(*domain_beta, (nb_init, 1)))]
Y_init2_1s = f2_1s(X_init2_1s, verbose = False)
mo_dict2_1s = {'output_dim':3, 'rank':0, 'missing':False}
bounds2_1s = [{'name': 'alpha', 'type': 'continuous', 'domain': domain_alpha}, {'name': 'beta', 'type': 'continuous', 'domain': domain_beta}]
bo_args2_1s = {'domain': bounds2_1s, 'optim_num_anchor':5, 'optim_num_samples':10000, 
           'acquisition_type':'EI_target', 'acquisition_weight':4, 
           'acquisition_weight_lindec':True, 'acquisition_ftarget': f_tgt2_1s,
           'X': X_init2_1s, 'Y': Y_init2_1s, 'mo':mo_dict2_1s, 'model_type':'GP_CUSTOM_LIK', 
           'inf_method':'Laplace', 'likelihood':'Binomial_' + str(1), 'normalize_Y':False}


BO2 = GPyOpt.methods.BayesianOptimization(f2_1s, **bo_args2_1s)
BO2.run_optimization(max_iter = 50, eps = 0)
(x0_2, y0_2), (x1_2,y1_2) = BO2.get_best()
qb.get_fid(x1_2, phitgt2_1s)

