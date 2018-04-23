#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:03:37 2017

@author: fs
"""
from .. import Helper as ut
from . import RandomGenerator as rdmg

import scipy.optimize as optim
import numpy as np
import pdb as pdb
import importlib as ilib
ilib.reload(rdmg)

# TODO: Implement RandomSearch // BFGS // GRID
#==============================================================================
#                   CUSTOMOPTIM CLASS 

# 
# Implementation of some optimization procedures in an abstract class.
#
# MAIN METHODS IMPLEMENTEDs
#    init
#    run
#
#==============================================================================
class AbstractOptim:
    
    # parameters MANDATORY to initialise the optimizer (should be maintained)
    LIST_PARAMS_OPTIM_NEEDED = {}
    LIST_PARAMS_OPTIM_NEEDED['algo'] =  "Type of optimization performed - 'DE' differential evolution, 'NM' nedler mead, 'BH' basin hopping, 'NoOPtim' (still can be some testing)"
    LIST_PARAMS_OPTIM_NEEDED['nb_params'] =  "Nb of parameters to be optim"
    
    # parameters OPTIONAL to initialise the optimizer (should be maintained)
    LIST_PARAMS_OPTIM_OPT = {}
    LIST_PARAMS_OPTIM_OPT['params_bound'] =  'How to set up bounds on parameters - Mandatory for DE'
    LIST_PARAMS_OPTIM_OPT['params_init'] =  'How to set up the initialization of the parameters - Mandatory if NM / Simple / BH algos are used'
 

# ------------------------------------------------------
# MAIN METHODS:
#     __init__
#     run
#     save     
# ------------------------------------------------------
    def __init__(self, paramsOptim = None, paramsSimulation = None, paramsSimulationTest = None):
        self.ALGO_AVAILABLE = {'NM': self.runNM, 'NoOptim':self.runTest, 'BH':self.runBH}

        # Prepare the parameters needed for the OPTIMIZERS
        self.params_optim = self.InitOptimizer(paramsOptim)
        self._name_algo = self.params_optim['algo']
        self._nb_params = self.params_optim['nb_params']
        self._seed = self.params_optim.get('seed', None)
        self._rdm_gen = rdmg.RandomGenerator()
                
        #Init the simulator
        self.params_sim = ut.getStringOrDico(paramsSimulation)
        self.simulator, self.sim_args = self.initSimulator(self.params_sim)  

        #Init the simulator for TESTING
        if(paramsSimulationTest not in [None, {}]):
            self.params_sim_test = ut.getStringOrDico(paramsSimulationTest)
            self.simulator_testing, self.test_args = self.initSimulator(self.params_sim_test) 

        # generate init parameters and boundaries
        self.params_optim['params_bound'], self.params_optim['params_init'] = self.GenInitAndBoundaries()
        self.best = {}



    def Run(self, simulator = None, writeLogs = False):
        """
        Purpose (should be general enough not to be re-implemented in the SubClasses):
            (1) Wrap the simulator into a function such that it takes only the arguments 
            to be optimized as input and returns the relevant FOM.
            (2) Run the relevant optimization procedures
            (3) Test the optimal arguments (optional)
            
            (0) Doesn't have a saving capability - could 
     
        Arguments:
            writeLogs
        """
        if(simulator is None):
            simulator = self.simulator
        
        
        # Wrap the simulator into a function taking args to optim as its only 
        self.sim_args['write_logs'] = writeLogs
        self.arg2costFun = self.optimWrapper(simulator, self.sim_args)      

        # Run optimizations
        algo2run = self.ALGO_AVAILABLE[self._name_algo]
        self.resOptim = algo2run(self.arg2costFun, self.params_optim, writeLogs)
        #pdb.set_trace()
        
        # Gather Results (optimal parameters) into a dictionary
        name_res = self.GenNameSimul(self.params_optim['res_name_method'], self.sim_args, self.params_optim)
        res= {'name': name_res}
        self.AddResultsOptim(res, self.resOptim, self.sim_args, 'opt_')

        if(writeLogs):
            self._WriteLogs(self.resOptim)        
        
        # run the testing (if parameters have been provided) with same FOM as for
        # the optim if not o.w. specified and optimal_params found during the optimizations
        if(hasattr(self, 'simulator_testing')):
            #pdb.set_trace()
            simulator = self.simulator_testing
            cost_test = self.optimWrapper(simulator, self.test_args, flagTesting = True)
            init = res['opt_params']
            if(init is None):
                print('No Optimal Parameters found, testing is done based on init parameters')
                init = self.params_init
            argsRunTest = {'init':init}
            self.resTest = self.runTest(cost_test, argsRunTest, writeLogs)
            self.AddResultsOptim(res, self.resTest, self.test_args, 'test_')


        return res

    def Save(self, pathFile = None, funWrite = None):
        """
        Purpose:
            save results (res - a dictionnary) as a text file
        Arguments:
            path: where to save the file by default will gen a random integer
            funWrite: function which can process object.res and save it, by default
                      ut.
        """
        assert hasattr(self, 'res'), 'Abstarct.OPtim.save : no res to save'
        
        if(funWrite is None):
            funWrite = ut.dico2text
        funWrite(self.res, pathFile)
            
            
# ------------------------------------------------------
# OPTIMIZATION PROCEDURES IMPLEMENTATION:
#   runNM - Nedler Mead
#   runBH - Basin Hopping
#   runTest - No Optim 
# ------------------------------------------------------
    def runNM(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Implementation of a Nedler Meat optimization
            (need initial parameters)

        """
        if arg2costFun is None:
            arg2costFun = self.arg2costFun
        
        maxfev = argsOptim['nm_maxfev']
        maxiter = argsOptim['nm_maxiter']
        ftol  = argsOptim['nm_ftol']
        init = argsOptim['params_init']
        disp = argsOptim['disp']
        options={'disp': disp, 'maxiter': maxiter, 'maxfev': maxfev, 'ftol': ftol}    
        if(len(np.shape(init)) > 1):
            options['initial_simplex'] = init
            init = init[0,:]
        cost = ut.reduceFunToFirstOutput(arg2costFun) 
        # Set up and run optimizer
        resultOptim = optim.minimize(cost, init, method='Nelder-Mead', options = options)
        return resultOptim


    def runBH(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Implementation of a Basin Hopping optimization
            (need initial parameters)

        """

        if arg2costFun is None:
            arg2costFun = self.arg2costFun
            
        niter_success = argsOptim['bh_niter_success']
        stepsize = argsOptim['bh_stepsize']   
        T = argsOptim['bh_T']
        init = argsOptim['params_init'] 
        disp = argsOptim['disp'] 
        
        cost = ut.reduceFunToFirstOutput(arg2costFun) 
        # Set up and run optimizer
        resultOptim = optim.basinhopping(cost, init, T = T, stepsize = stepsize, disp = disp, niter_success = niter_success)
        return resultOptim
    
 
    def runTest(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Simple run of the simulator on a set of parameters (no Optim involved,
            one call of the simulator)
        Comment:
            Slightly diff behavior: cost can return a list of FOM
            Is it a good workaround??
        """
       

        #TODO: will probably evolved to contain more details abt the optim 
        params = argsOptim['init']
        simRun = arg2costFun(params)
        resultTest = {'x': params, 'fun': simRun, 'nfev':1, 'nit':1, 'sucess':True}
        return resultTest


# ---------------------------
# To be implemented in the subclass
# ---------------------------        
    def initSimulator(self, paramsSimul):
        """
        What it should do:
            Initialize the simulator (a.k.a the model) 

        """
        raise NotImplementedError()

    def GenNameSimul(self, name_method, args_simul):
        """
        What it should do:
            generate a name for the simulation (used to tag/store the results) 

        """
        raise NotImplementedError()

       
    def optimWrapper(self, simCall, args):
        """
        What it should do:
            Wrap the simulator to take arguments to be optim and return the figure of merit

        """
        raise NotImplementedError()
    

    def _WriteLogs():
        pass



# ------------------------------------------------------
#           INITIALIZATION FUNCTIONS:
#     InitOptimizer   
#     GenInitAndBoundaries
#        
# ------------------------------------------------------
    def InitOptimizer(self, params_optim):
        """
        Purpose
        """
        params_optim = ut.getStringOrDico(params_optim)
        def_params_optim = self.GenDefaultOptimParameters()
        if(def_params_optim is not None):
            params_optim = ut.merge_dico(def_params_optim, params_optim, update_type = 0)    
        return params_optim


    def GenInitAndBoundaries(self):
        """
        Purpose:
            Generate the boundaries and inital values of the parameters 
            (to be optimized)
        """
        bds = self.getBoundsParams(self.params_optim.get('params_bound', None))
        init = self.getInitParams(self.params_optim.get('params_init', None))
        return bds, init
    
    
    def getBoundsParams(self, method):
        """
        Purpose:
                generate bounds for the parameters to be optimized
        e.g. range_1_2            >>  [(1,2), (1,2), ..., (1,2)}]
             range_1_2_0_3_4_5    >>  [(1,2), (0,3), (4,5)}]
        """
        
        if method is None:
            return None
        else:
            nb_params = self._nb_params
            args = ut.splitString(method)
            if(args[0] == 'range'):
                if(len(args) == 3):
                    #same bounds for all the params
                    bounds = [(float(args[1]), float(args[2])) for _ in range(nb_params)]
                else:
                    nbParamsFound = int((len(args)-1)/2)
                    assert nbParamsFound == nb_params, 'issue in the generation of the bounds'
                    bounds = [(float(args[1 + 2*i]), float(args[2 + 2*i])) for i in range(nbParamsFound)]
            else:
                raise NotImplementedError()
        return bounds

 
    def getInitParams(self, method = None):
        """
        Purpose:
                Generate initial parameters for the optimization 
        Methods:
            + 'zero' for each params 0
            + 'force' LIST is directly used as the init parameters
            + 'random'
            + 'nm_guess'
        """        
        if method is None:
            init_args = None
            
        else:
            args = ut.splitString(method)
            nb_params = self._nb_params
            
            # zero everywhere
            if(args[0] == 'zero'):
                init_args = np.zeros_like(nb_params)

            # Force init 
            elif(args[0] == 'force'):
                if(ut.is_list_float(args[1])):
                    init_args = np.array(ut.listFloatFromString(args[1]))
                elif(ut.is_matrix_float(args[1])):
                    init_args = np.array(ut.matrixFloatFromString(args[1]))
                else:
                    raise NotImplementedError()
            
            # Random initialization
            elif(args[0] in ['uniform','normal', 'lhs']):
                if(self._name_algo == "NM"):  
                    dim = [self._nb_params+1, self._nb_params]
                else:
                    dim = self._nb_params
                init_args = self._rdm_gen.gen_rdmnb_from_string(method, dim)                    

            # For the initialization of NM based on one guess
            # for the NM algo we need to come up with N+1 vertices arround a guess
            # Based on Matlab routine
            # guess[0] = guess, guess[i] = guess + epsilon * n_i
            # where epsilon can be different when guess_i = 0 and n_i is one 
            # basis unit vector
            # e.g. nmguess_[3,0,1]_0.5_1 >> [[3,0,1], [4, 0, 1], [3,0.5,1], [3,0,2]]
            elif(args[0] == 'nmguess'):
                assert (len(args)>=2), 'nmguess, not the right format'
                if(len(args) == 2):
                    step0 = args[1]
                    stepNon0 = args[1]
                elif(len(args) == 3):
                    step0 = args[1]
                    stepNon0 = args[2]
                    
                init_args = np.zeros_like([nb_params + 1, nb_params])
                guess = ut.recastString(args[1])
                if(isinstance(guess, str)):
                    guess_vector = self.getInitParams(guess)
                
                elif(isinstance(guess, list)):
                    guess_vector = np.array(guess)
                    
                init_args[0,:] = guess_vector
                for i in range(nb_params):
                    perturb = np.zeros(nb_params)
                    if(guess[i] == 0):
                        perturb[i] = step0
                    else:
                        perturb[i] = stepNon0 
                    init_args[(i+1), :] = init_args[0,:] + perturb
                
                return init_args
            
            print(init_args)
            
            return init_args




# ============================================================================#
#                        MISCELLANEOUS
#            
# addResultsOptim - Differential Evolution
# GenDefaultOptimParameters
# runTest - No optimization performed, just run the simulation
# ============================================================================#
    def AddResultsOptim(self, resInit, resOptim, argsOptim, prefix = ''):
        """
        Purpose:
            fill the (dico) resInit with some entries from resOptim an other dictionary
            typically return by scipy.minimize
        Arguments:
            + argsOptim 
            + prefix
        """
        resInit[prefix +'params'] = resOptim.get('x')
        fom_tmp = resOptim.get('fun')
        FOM_name = argsOptim['fom_name']
        
        if(isinstance(fom_tmp, list)):
            for i in range(len(fom_tmp)):
                resInit[prefix + FOM_name[i]] = fom_tmp[i]
        else:
            if(isinstance(FOM_name, list)):
                FOM_name = FOM_name[0]
            resInit[prefix + FOM_name] = fom_tmp
                    
        resInit[prefix +'nbEvalF'] = resOptim.get('nfev') 
        resInit[prefix +'nbIt'] = resOptim.get('nit')
        resInit[prefix +'Sucess'] = resOptim.get('sucess')


    def GenDefaultOptimParameters(self):
        """
        Purpose:
            Generate default parameters for the optimizer
        """
        dico={}
        dico['disp'] = True
        dico['maxiter'] = 500
        dico['res_name_method'] = None
        
        #Nedler Mead algo
        dico['nm_maxiter'] = 200 
        dico['nm_ftol'] = 0.00001 # tolerance 
        dico['nm_maxfev'] = 200

    
        # Bassin Hoping algorithm
        dico['bh_niter_success'] = 20 
        dico['bh_stepsize'] = 0.5  #pop = nbParams * popsize   
        dico['bh_T'] = 1.0
        return dico
    
    def PrintDefaultOptimParameters(self):
        dicoDef = self.GenDefaultOptimParameters()
        print(dicoDef)


    @classmethod
    def helpConfiguration(self):
        print('Configuration arguments for the OPTIMIZATION:')
        print('Optim_mandatory:')
        ut.printDicoLineByLine(self.LIST_PARAMS_OPTIM_NEEDED)
        print('')
        print('Optim_optional:')
        ut.printDicoLineByLine(self.LIST_PARAMS_OPTIM_OPT)
        print('')

    #TODO: tofill  
    def helpParametersOptimization(self):
        dico_nm = {}
        dico_de = {}
        dico_gp = {}
        dico_bh = {}
        ut.printDicoLineByLine(dico_nm)
        ut.printDicoLineByLine(dico_de)
        ut.printDicoLineByLine(dico_gp)
        ut.printDicoLineByLine(dico_bh)
        





