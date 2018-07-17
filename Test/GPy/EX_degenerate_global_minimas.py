#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:12:14 2018

@author: fred
"""
import sys
sys.path.append('/home/fred/Desktop/GPyOpt')
sys.path.append('/home/fred/anaconda3/envs/py36q/lib/python3.6/site-packages')
import GPyOpt
import numpy as np
import pdb
from functools import partial
import matplotlib.pylab as plt

# ------------------------------------------------------
# Utility: 0
# 
#
#
# ------------------------------------------------------
def genFunction(noise = 0, tilt = 0, degen = 1, dim = 1, local_degen = 3):
    def f(X):
        """generate f(X=[x1,.., xd]) = f(x1) x ... x f(x2)
        with f(x) = sin^2((x-0.1)*degen*pi) + cos^2((x-0.1) * 3 * degen * pi) + tilt * x 
        
        X = ((1+0.2)/2degen,...,1/2degen) is a minimum with f(x) = (-1+titl/2degen)^d
        there is degen^d (quasi) degenerate mimimas and ((local_degen-1) * degen) ^d local minimas
        """
        X = np.array(X).reshape((len(X),dim))
        freq_ppal = np.pi * degen
        freq_scnd = local_degen * freq_ppal
        res = 1
        for i in range(dim):
            XX = X[:,i] + 0.1 /degen 
            res *= tilt * XX - np.square(np.sin(XX * freq_ppal)) + 0.4 * np.square(np.cos(XX*freq_scnd))
        noise_to_add = np.random.normal(0,noise, len(X))
        return res + noise_to_add

    minimas = [[(0.4 + n) / (degen) for n in range(int(degen))] for d in range(dim)]
    return f, minimas


def get_best_seen_from_BO(bo):
    return bo.X[np.argmin(bo.Y)]
    
def get_best_exp_from_BO(bo):
    Y_pred = bo.model.predict(bo.X)
    return bo.X[np.argmin(Y_pred[0])]

def get_dist_to_a_min(real_mins, estimated_min):
    return np.min(np.abs(real_mins-estimated_min))

# ------------------------------------------------------
# Study 1: 0
# plot function
# ------------------------------------------------------
degen, local_degen, dim, tilt, noise = 1, 3, 1, 0, 0
bounds = [{'name': 'var_'+str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(dim)]
myf, minimas = genFunction(noise, tilt, degen, dim, local_degen)
if(dim == 1):
    x_plot = np.linspace(0,1,1000)
    y_plot = myf(x_plot)
    plt.plot(x_plot, y_plot)



# ------------------------------------------------------
# Study 1: how do we choose the optimal x: is it based 
# on the best y seen or on the best expected y (i.e. fm the surr model)
# still only consider only x explored
# RES better to use expected values
# ------------------------------------------------------
study_1 = False
if study_1:
    list_iter = [10, 25, 50]
    list_noise = [0, 0.1, 0.2]
    nb_repeat = 10
    
    res = np.zeros([2, len(list_iter), len(list_noise), nb_repeat])
    
    for n, it in enumerate(list_iter):
        budget = it
        init = 5
        maxiter = budget - init
        
        for p, noise in enumerate(list_noise):
            degen, local_degen, dim, tilt = 3, 3, 1, 0
            bounds = [{'name': 'var_'+str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(dim)]
            myf, minimas = genFunction(noise, tilt, degen, dim, local_degen)
            
            for r in range(nb_repeat):
                myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, 
                        acquisition_type ='EI', optim_num_samples = 100000, optim_num_anchor = 50,
                        initial_design_type = 'random', num_cores = 4)
                myOpt.run_optimization(maxiter)
            
                if(False):
                    myOpt.plot_acquisition()
                    myOpt.plot_convergence()
            
                x_best_seen = get_best_seen_from_BO(myOpt)
                x_best_exp = get_best_exp_from_BO(myOpt)
                d_seen = get_dist_to_a_min(minimas, x_best_seen)
                d_exp = get_dist_to_a_min(minimas, x_best_exp)
                res[0, n, p, r] = d_seen
                res[1, n, p, r] = d_exp
            
        
    #no_noise    
    i_iter = 2
                
    mean_diff = np.mean(np.squeeze(res[0,i_iter,:,:]), axis = 1)
    std_diff = np.std(np.squeeze(res[0,i_iter,:,:]), axis = 1)
    mean_diff_exp = np.mean(np.squeeze(res[1,i_iter,:,:]), axis = 1)
    std_diff_exp = np.std(np.squeeze(res[1,i_iter,:,:]), axis = 1)
    
    print(mean_diff)
    print(mean_diff_exp)



# ------------------------------------------------------
# Study 2: In the case of a lot of degenerate minimas is it better to spend 
# more time in the exploitation phases
# ------------------------------------------------------
study_2 = False
if study_2:
    noise = 0
    list_degen = [1, 3, 5, 10]
    list_iter = [25, 50, 100]
    list_pct_exploit = [0, 0.25, 0.50, 1]
    nb_repeat = 10
    
    res = np.zeros([2, len(list_degen), len(list_iter), len(list_pct_exploit), nb_repeat])
    
    for d, deg in enumerate(list_degen):
        print('d='+str(d))
        degen, local_degen, dim, tilt = deg, 3, 1, 0
        bounds = [{'name': 'var_'+str(i), 'type': 'continuous', 'domain': (0, 1)} for i in range(dim)]
                
        for i, it in enumerate(list_iter):
            print('i='+str(i))
            budget = it
            init = 5
            
            for p, pct in enumerate(list_pct_exploit):
                nb_exploit = int(pct * budget /100)
                nb_explor = budget - nb_exploit
                print('p='+str(p))
                
                myf, minimas = genFunction(noise, tilt, degen, dim, local_degen)
                
                for r in range(nb_repeat):
                    myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, 
                            acquisition_type ='EI', optim_num_samples = 100000, optim_num_anchor = 50,
                            initial_design_type = 'random', num_cores = 4)
                    myOpt.run_optimization(nb_explor)
                
                    bo_new = GPyOpt.methods.BayesianOptimization(myf, bounds, 
                                acquisition_type = 'LCB', X = myOpt.X, Y =myOpt.Y, 
                                acquisition_weight = 0.00001, optim_num_anchor = 50, 
                                optim_num_samples = 100000, num_cores = 4)
                    
                    bo_new.run_optimization(nb_exploit)

                
                    x_best_seen = get_best_seen_from_BO(bo_new)
                    x_best_exp = get_best_exp_from_BO(bo_new)
                    d_seen = get_dist_to_a_min(minimas, x_best_seen)
                    d_exp = get_dist_to_a_min(minimas, x_best_exp)
                    res[0, d, i, p, r] = d_seen
                    res[1, d, i, p, r] = d_exp
        
    
    #no_noise    
    i_iter = 2
                
    mean_diff = np.mean(np.squeeze(res[0,i_iter,:,:]), axis = 1)
    std_diff = np.std(np.squeeze(res[0,i_iter,:,:]), axis = 1)
    mean_diff_exp = np.mean(np.squeeze(res[1,i_iter,:,:]), axis = 1)
    std_diff_exp = np.std(np.squeeze(res[1,i_iter,:,:]), axis = 1)
    
    print('MEAN')
    print(mean_diff)
    print(mean_diff_exp)
    print('STD')
    print(std_diff)
    print(std_diff_exp)







# ------------------------------------------------------
# Study 3: In the case of a lot of degenerate minimas is it better to spend 
# more time in the exploitation phases
# ------------------------------------------------------
def local_BO_grad(bo, myf, nbiter_max= 50, v_momentum=0.5, alpha=0.1):
    v = 0
    for it in range(nbiter_max):
        Xbest, Ybest, grad, length = get_info_from_BO(bo)
        v = v_momentum * v+ alpha * grad 
        Xtest = Xbest - v
        Ytest = myf(Xtest)
        bo.add(Xtest, Ytest)
        if(Ytest < Ybest):
            print(Ytest)
    


def get_info_from_bo(bo, type_best):
    
    



#Strat2
maxiter = budget - init - final
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, acquisition_type ='EI')
myOpt.run_optimization(maxiter)
X_already, Y_already = myOpt.X, myOpt.Y
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, X= X_already, Y = Y_already, acquisition_type ='LCB',exploration_weight =0.000001)
myOpt.run_optimization(5)
myOpt.plot_acquisition()
myOpt.plot_convergence()
print(myOpt.fx_opt)
print(myOpt.x_opt)



#Strat3
import copy
percentage = 0.1
init = 5
final = final = int(max(5, budget* percentage))
maxiter = budget - init - final
before_final = budget - final 
restrict = int(max(5, before_final * percentage))
def c_diff(x1, x2):
    diff = np.squeeze(x1) - np.squeeze(x2)
    return np.sqrt(np.dot(diff, diff))

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 4)}]
myOpt = GPyOpt.methods.BayesianOptimization(myf, bounds, initial_design_numdata = 5, 
                    acquisition_type ='EI',num_cores=4)
myOpt.run_optimization(maxiter)


X_already, Y_already = myOpt.X, myOpt.Y
index_best = np.argmin(myOpt.Y)
best_Y = Y_already[index_best]
best_X = X_already[index_best]
dist = [c_diff(best_X, x) for n, x in enumerate(X_already)]
index_closest = np.argsort(dist)
index_relevant = index_closest[: restrict]
X_relevant = X_already[index_relevant]
Y_relevant = Y_already[index_relevant]
new_limits = [np.min(X_relevant,0), np.max(X_relevant, 0)]
new_bounds = [copy.copy(b) for n, b in enumerate(bounds)]
for n, m in enumerate(new_bounds):
    m.update({'domain':(new_limits[0][n], new_limits[1][n])})

myOpt2 = GPyOpt.methods.BayesianOptimization(myf, new_bounds, X= X_relevant, 
                    Y = Y_relevant, acquisition_type ='EI'BayesianOptimization,
                    num_cores=4)
myOpt2.run_optimization(final)
myOpt2.plot_acquisition()
myOpt2.plot_convergence()
print(myOpt.fx_opt)
print(myOpt.x_opt)




    

myOpt2 = GPyOpt.methods.BayesianOptimization(myf, bounds, X= X_already, 
        Y = Y_already, acquisition_type ='LCB',exploration_weight =0.000001,
        num_cores=4)
myOpt2.run_optimization(5)


#learning process
myOpt2.plot_acquisition()
myOpt2.plot_convergence()

print(myOpt.fx_opt)
print(myOpt.x_opt)



