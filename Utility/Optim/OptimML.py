#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:03:37 2017

@author: fs
"""
if __name__ == "__main__":
    import sys
    sys.path.append("../../")
    from Utility.Optim import Optim as optim
    from Utility.BayesOptim import BayesianOptimization
    from Utility import Helper as ut
else:
    from . import Optim as optim
    from .. import Helper as ut
    from ..BayesOptim import BayesianOptimization 


import scipy.optimize as sco
import pdb as pdb
import numpy as np
import importlib as ilib
ilib.reload(optim)


#==============================================================================
#                   CUSTOMOPTIM CLASS 
# 
# Implementation of some optimization procedures in an abstract class.
#
# MAIN METHODS IMPLEMENTED
#    init
#    run
#
#==============================================================================
# TODO: Implement gaussian Processes // BayesianOptim // BFGS // GRID


class AbstractOptimML(optim.AbstractOptim):
# ------------------------------------------------------
# MAIN METHODS:
#   __init__
#   run
#   save     
# ------------------------------------------------------
    
    def __init__(self, paramsOptim = None, paramsSimulation = None, paramsSimulationTest = None):
        optim.AbstractOptim.__init__(self, paramsOptim, paramsSimulation, paramsSimulationTest)
        self.ALGO_AVAILABLE['DE'] = self.runDE
        self.ALGO_AVAILABLE['GP'] = self.runGP


# ------------------------------------------------------
# ML based OPTIMIZATION PROCEDURES :
# runDE - Differential Evolution
# runGP - GP
# ------------------------------------------------------
    def runDE(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Implementation of a differential evolution optimizer
            (need bounds)

        """
        if arg2costFun is None:
            arg2costFun = self.arg2costFun
        cost = ut.reduceFunToFirstOutput(arg2costFun) 

        #Optim setup and run
        bounds= argsOptim['params_bound']
        popsize = argsOptim['de_popsize']
        maxiter =argsOptim['de_maxiter']
        disp = argsOptim['disp'] 
        
        resultOptim = sco.differential_evolution(cost, bounds, popsize = popsize, maxiter = maxiter, disp = disp)
        return resultOptim

    
    def runGP(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Run a bayesian optimization using the library Bayesian Optimization
            (https://github.com/fmfn/BayesianOptimization) built on 
            
        Comment:
        #TODO: tuning of the kernel in particular implement ARD
        #TODO: Transfert Learning
        #TODO: REMBO
        #TODO: Prior on the measurement noise
        #TODO: parrallelization of the acq_max
        #TODO: Final optim
        #TODO: Bigger alphabet
        """        
        
        bo, bo_args, name_params, nb_points_init = self.InitializeGP(argsOptim, arg2costFun)
        bo.maximize(init_points=0, **bo_args)

        # Exploitation phase
        bo_args_Maximize = {'n_iter':15, 'acq': 'ucb', 'kappa':0.0001} 
        bo.maximize(init_points=0, **bo_args_Maximize)
        
        # Close pool of processors used (if it exists)
        nb_workers_used = bo._nb_workers
        bo.close_mp_pool()

        #Plot if 1D
        if(self._nb_params == 1):
            xmin = argsOptim.get('params_bound')[0][0]
            xmax = argsOptim.get('params_bound')[0][1]
            x = np.linspace(xmin, xmax, 1000).reshape(-1, 1)
            y = np.zeros_like(x)
            
            plot_gp(bo, x, y)

       
        #Create and return results      
        index_max = np.argmax(bo.Y)
        optim_params = bo.X[index_max]
        optim_value = arg2costFun(optim_params)
        niter = bo_args['n_iter']
        nfev = niter + nb_points_init
        resultTest = {'x': optim_params, 'fun': optim_value, 'nfev':nfev, 'nit':niter, 'sucess':True}
        resultTest['gp_kernel_init'] = str(bo.gp.kernel)
        resultTest['gp_kernel_optim'] = str(bo.gp.kernel_)
        resultTest['gp'] = bo.gp.kernel_
        resultTest['nb_processes'] = nb_workers_used
        return resultTest


# ------------------------------------------------------
# Support function
# ------------------------------------------------------
    def InitializeGP(self, argsGP, arg2costFun):
        """ Create the GP model with 
        """
        pdb.set_trace()
        acq = argsGP.get('gp_acq') #acquisition function
        niter = argsGP.get('gp_maxiter') # Number of data to get
        kappa = argsGP.get('gp_kappa')
        verbose = argsGP.get('gp_verbose')
        init_type = argsGP.get('gp_init') #    
        kernel = argsGP.get('gp_kernel')
        white_noise = argsGP.get('gp_wnoise')
        scaling_ker = argsGP.get('gp_scaling')
        flag_MP = argsGP.get('flag_MP', False)
        
        dico_gp = {'kernel': kernel, 'whiteNoise':white_noise, 'scalingKer':scaling_ker, 'flag_MP': flag_MP}
        
        #static kappa (parameters)
        if(isinstance(kappa, float)):
            kappa = np.repeat(kappa, niter)
        #Dynamic kappa
        else:
            bits = ut.splitString(kappa)
            if(bits[0] == 'linear'):
                kappa = float(bits[1]) * (1 - np.arange(niter) / (niter - 1))
            #kappa = kappa[0] #TODO: CHANGEEE
        
        # name the parameters in a consistent way
        name_params = [str(i) for i in range(self._nb_params)]
        boundsDico = {name_params[i]: argsGP.get('params_bound')[i] for i in range(self._nb_params)}
        cost = ut.ArrayToDicoInputWrapper(arg2costFun)
        
        #Create the optimizer
        bo = BayesianOptimization(cost, boundsDico, verbose = verbose, **dico_gp)
        
        # Custom Init
        if (init_type is None):
            # Init is managed in the library BayesianOptim by def require 2 * nb_params init point
            bo.maximize(init_points= (self._nb_params *2), n_iter=0, acq=acq, kappa=0)  
            nb_points_init = self._nb_params *2
            
        elif (isinstance(init_type, int)):
            # Init is managed in the library BayesianOptim
            bo.maximize(init_points= init_type, n_iter=0, acq=acq, kappa=0)
            nb_points_init = init_type
            
        elif (ut.is_matrix_float(init_type)):
            init_matrix = ut.matrixFloatFromString(init_type)
            nb_points_provided = len(init_matrix)
            to_explore = {name_params[i]: [init_matrix[j][i] for j in nb_points_provided] for i in range(self._nb_params)}
            bo.explore(to_explore)
            nb_points_still_needed = max(0, self._nb_params *2 - nb_points_provided)
            bo.maximize(init_points = nb_points_still_needed, n_iter = 0, acq = acq, kappa = 0)
            nb_points_init = nb_points_provided + nb_points_still_needed
                    
        else:
            bits = ut.splitString(init_type)
            #FirstImplementation of latinHypercube sampling
            if(bits[0] == 'lhs'):
                if len(bits) == 2:
                    size_pop = [int(bits[1]), self._nb_params]
                else:
                    size_pop = [self._nb_params * 2, self._nb_params]
                limits = np.array(argsGP.get('params_bound')).T
                init_matrix = self._rdm_gen.init_population_lhs(size_pop, limits)
            else:
                init_matrix = [self.getInitParams(b) for b in bits]
                
            nb_points_provided = len(init_matrix)
            to_explore = {name_params[i]: [init_matrix[j][i] for j in range(nb_points_provided)] for i in range(self._nb_params)}
            bo.explore(to_explore)
            nb_points_still_needed = max(0, self._nb_params *2 - nb_points_provided)
            if(nb_points_still_needed > 0):
                bo.maximize(init_points = nb_points_still_needed, n_iter = 0, acq = acq, kappa = 0)
            nb_points_init = nb_points_provided + nb_points_still_needed
            
        bo_args = {'n_iter':niter, 'acq': acq, 'kappa':kappa}    
        return bo, bo_args, name_params, nb_points_init
    
    
    
    
    def GenDefaultOptimParameters(self):
        """
        Purpose:
            Generate default parameters for the optimizer
        """
        dico = optim.AbstractOptim.GenDefaultOptimParameters(self)

        # Differential Evolution algorithm
        dico['de_maxiter'] = 50 
        dico['de_popsize'] = 10  #pop = nbParams * popsize  
        dico['de_tol'] = 0.01
    
        # Gaussian Processes
        dico['gp_acq'] ='ucb'
        dico['gp_kappa'] = 5.0
        dico['gp_maxiter'] = 150
        dico['gp_init'] = None
        dico['gp_verbose'] = False
    
        return dico
    
    
    def PrintDefaultOptimParameters(self):
        dicoDef = self.GenDefaultOptimParameters()
        print(dicoDef)
    
    
    def helpConfiguration(self):
        optim.AbstractOptim.helpConfiguration()
        print('New configuration parameters for the ML OPTIMIZATION:')
        print('Optim_mandatory:')
        ut.printDicoLineByLine(self.LIST_PARAMS_OPTIM_ML_NEEDED)
        print('')
        print('Optim_optional:')
        ut.printDicoLineByLine(self.LIST_PARAMS_OPTIM_ML_OPT)
        print('')

    #TODO: tofill
    @staticmethod    
    def helpParametersOptimization(self):
        dico_nm = {}
        dico_de = {}
        dico_gp = {}
        dico_bh = {}
        ut.printDicoLineByLine(dico_nm)
        ut.printDicoLineByLine(dico_de)
        ut.printDicoLineByLine(dico_gp)
        ut.printDicoLineByLine(dico_bh)
 

# =============================================================================
# Some plotting function for Gaussian Optim stuff
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib import gridspec
def posterior(bo, x, xmin=-2, xmax=10):
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(bo, x, y):
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = posterior(bo, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    xmin = np.min(x)
    xmax = np.max(x)
    axis.set_xlim((-xmin, xmax))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility = bo.util.utility(x, bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
