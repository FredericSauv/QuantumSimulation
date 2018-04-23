#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
#
#============================================================================== 
from ...ToyModels import ControlledSpin as cspin
from ...Utility import Helper as ut
from ...Utility.Optim import OptimML as opt
from ...Utility.Optim import ParametrizedFunctionFactory as pfunc

import numpy as np
import pdb as pdb #debugger 
import importlib as ilib
import time
ilib.reload(opt)
ilib.reload(cspin)


#TODO: Deal with setup2
#TODO: Test setup 3
class ToyModelOptim(opt.AbstractOptimML):
    """ 
    Implementation of the abstract optimization for the toyModel (simulated via
    the lightSimulation class)
    
    BUGS:
        Testing works as long as some params are not changed between simul and testing
    
    """
    # parameters MANDATORY to initialise the simulator (should be maintained)
    LIST_PARAMS_SIM_NEEDED = {}
    LIST_PARAMS_SIM_NEEDED['setup'] = "1: f(t) sig_X + (1-f(t)) sig_Z //2: f(t) sig_X + g(t) sig_Z // 3: 2qbits"
    LIST_PARAMS_SIM_NEEDED['target_state_name'] = "name of the target state"
    LIST_PARAMS_SIM_NEEDED['init_state_name'] = "initial state provided by its "
    LIST_PARAMS_SIM_NEEDED['control'] = "parametrization type of the system i.e guess_linear_function_Fourier"    
    LIST_PARAMS_SIM_NEEDED['fom_name'] = "Figure Of Merit"
    
    # parameters OPTIONAL to initialise the simulator (should be maintained)  
    LIST_PARAMS_SIM_OPT = {}
    LIST_PARAMS_SIM_OPT['guess'] = "initial state"
    LIST_PARAMS_SIM_OPT['dt'] = "Time step for the simulations, default = 0.01"
    LIST_PARAMS_SIM_OPT['T'] = "Total time of the simulation, default = 1"    

    # available PARAMETRIZATION of the control function available
    LIST_PARAMETRIZATION = {}
    LIST_PARAMETRIZATION['sineFixedTrend'] = ""
    LIST_PARAMETRIZATION['fourFixedCt'] = ""
    LIST_PARAMETRIZATION['sineTrendFixedCt'] = ""
    LIST_PARAMETRIZATION['sine'] = ""
    LIST_PARAMETRIZATION['chebyFixedCt'] = ""
    LIST_PARAMETRIZATION['chebyFixedTrend'] = ""
    LIST_PARAMETRIZATION['cheby'] = ""    
    
    DIC_NAME2FUN = {}
    DIC_NAME2FUN['fourFixedCt'] = pfunc.cFourierAmplitudesFixedCt
    DIC_NAME2FUN['sineFixedTrend'] = pfunc.cSineFixedTrend
    DIC_NAME2FUN['sineTrendFixedCt'] = pfunc.cSineTrendFixedCt
    DIC_NAME2FUN['sine'] = pfunc.cSine
    DIC_NAME2FUN['chebyFixedCt'] = pfunc.cChebyshevFixedCt
    DIC_NAME2FUN['chebyFixedTrend'] = pfunc.cChebyshevFixedTrend
    DIC_NAME2FUN['cheby'] = pfunc.cChebyshev    
    DIC_NAME2FUN['chebyOdd'] = pfunc.cChebyshevOdd
    DIC_NAME2FUN['step'] = pfunc.StepFunc
    DIC_NAME2FUN['squareexp'] = pfunc.SquareExponential
    # DIC_NAME2FUN['sumsquareexp'] = pfunc.SumSquareExponential


    def __init__(self, paramsSimulation = None, paramsOptim = None, paramsSimulationTest = None):
        # call of self.initSimulator from the parent class
        self.nbControlFunPerSetUp = {'1':1, '2':2, '3':1}
        opt.AbstractOptimML.__init__(self, paramsOptim, paramsSimulation, paramsSimulationTest)
        self.ALGO_AVAILABLE['D-CRAB'] = self.runDCRAB ## Almost surely the wrongplace!!!
        
        
    @classmethod    
    def helpConfiguration(cls):
        print('Configuration arguments for the SIMULATIONS:')
        print('Simulation_mandatory:')
        ut.printDicoLineByLine(cls.LIST_PARAMS_SIM_NEEDED)
        print('')
        print('Simulation_optional:')
        ut.printDicoLineByLine(cls.LIST_PARAMS_SIM_OPT)
        print('')
        opt.AbstractOptim.helpConfiguration(cls)
        print('')


# ---------------------------
# Implementation of the methods required (but not Implemented) from the abstract class 
# ---------------------------
    def initSimulator(self, p):
        """
        Purpose:
            Custom initialization of the simulator based on the paramsSimulation
            provided. 
        Input:
            p : (dico/textfile) which contains the config of the simulation
        
        Output:
            Simulator (model we want to simulate which has been initialized)
            p: 
            
         Comments:
            Extra infos are generated
        """
        # Extract setup variables from the dico
        setup = p.get('setup')
        target_state_name = p.get('target_state_name')
        init_state_name = p.get('init_state_name')
        p['method_simul'] = p.get('method_simul', 'PWCH')
        custom_control, args_control = self.GenerateCustomControlFunction(p)
        sim_args = ut.merge_dico(p, args_control, update_type = 3)
        noise = p.get('noise')
        simulator = cspin.ControlledSpin(controlFun = custom_control, setup = setup, 
                                         state_init = init_state_name, state_target = target_state_name, noise = noise)

        self._args_control = args_control
        return simulator, sim_args



    def GenNameSimul(self, name_method = None, args_simul = None, args_optim = None):
        """
        What it should do:
            generate a name for the simulation (used to tag/store the results) 
        """
        setup = str(args_simul['setup'])
        fom_name = args_simul['fom_name']
        
        if (name_method is None):
            if (isinstance(fom_name, list)):
                fom_name = fom_name[0] # use first one as that is the one used for optim
            name = ut.concat2String(self._name_algo, 'Setup', setup)
            
        else:
            raise NotImplementedError()
            
        return name


    
    def optimWrapper(self, simulator, args, flagTesting = False):
        """Wrap the simulator to take arguments to be optim and return the figure of merit

        Output:
            funCost: Function <argsToOptim> -> FOM

        """
        write_logs = args.get('write_logs', False)
        fom_name = args.get('fom_name')
        method = args.get('method_simul')
        dt = args.get('dt')
        T = args.get('T')
        printVal = args.get('print', False)
        argsUpdateParams = {'parametrization': self._args_control.get('function'), 
                     'indexFun':self._args_control.get('indexFunToUpdate')}
        
        if(write_logs):
            # Some infos are obtained directly here
            self._nb_call_fev = 0
            self._timer = time
            self._time_init = self._timer.time()
            self._bestfom_nbev = []
            self._bestfom_time = []
            self._bestfom_func = []
            
            def arg2costFun(params):
                self.CustomUpdateControlParameters(params, simulator, **argsUpdateParams)
                self._randomize_simulator(simulator)
                
                fom = simulator.Simulate(T = T, time_step = dt, method = method, fom = fom_name, store = False)
                if(printVal):
                    print(params, fom)
                
                self._nb_call_fev += 1
                fom_best_so_far = self.best.get('fom')
                
                if(not(flagTesting)):
                    if (fom_best_so_far is None) or (fom_best_so_far > fom):
                        self.best['fom'] = fom
                        self.best['params'] = params
                        self.best['omegas'] = self._args_control.get('omegas')
                        self._time_elapsed = self._timer.time() - self._time_init
                        self._bestfom_nbev.append([self._nb_call_fev, fom])
                        self._bestfom_time.append([self._time_elapsed, fom])
                        self._bestfom_func.append([repr(simulator.controlFun), fom])
    
                return fom
            
            
        else:
            def arg2costFun(params):
                self.CustomUpdateControlParameters(params, simulator, **argsUpdateParams)
                self._randomize_simulator(simulator)
                
                fom = simulator.Simulate(T = T, time_step = dt, method = method, fom = fom_name, store = False)
                if(printVal):
                    print(params, fom)
                
                fom_best_so_far = self.best.get('fom')
                if(not(flagTesting)):
                    if (fom_best_so_far is None) or (fom_best_so_far > fom):
                        self.best['fom'] = fom
                        self.best['params'] = params
                        self.best['omegas'] = self._args_control.get('omegas')
                
    
                return fom
        
        return arg2costFun


    def _WriteLogs(self, dicoRes):
        """Add extra info to the dico of results
        """
        dicoRes['time_fom'] = self._bestfom_time
        dicoRes['fev_fom']= self._bestfom_nbev
        dicoRes['func_fom'] = self._bestfom_func

    def _randomize_simulator(self, simulator):
        """ Randomize the simulators
        """
        simulator.gen_energy_scaling()
        
# ---------------------------
# CUSTOM functions to initialize the model
# ---------------------------
    def GenerateCustomControlFunction(self, params):
        """
        PURPOSE: 
            generate (potentially several) custom parametrized control functionS
        """
        
        #Guess function
        
        dicoGuess = params.get('guess', None)
        dicoControl = params.get('control', None)
        dicoOverall = params.get('overall', None)
        T = params['T']
        
        if isinstance(dicoControl, list):
            nb_fun = len(dicoControl)
            control_func = []
            control_info = []
            
            if(not(isinstance(dicoGuess, list))):
                dicoGuess = [dicoGuess for _ in range(nb_fun)]
            
            if(not(isinstance(dicoOverall, list))):
                dicoOverall = [dicoOverall for _ in range(nb_fun)]
            
            for i in range(len(dicoControl)):
                ctl_tmp = dicoControl[i]
                guess_tmp = dicoGuess[i]
                overall_tmp = dicoOverall[i]
                fun_tmp, info_tmp = self.GenerateOneCustomControlFunction(guess_tmp, ctl_tmp, overall_tmp, T =T)
                control_func.append(fun_tmp)
                control_info.append(info_tmp)
            
            control_info= ut.concat_dico(control_info, global_name=None, suffix = 'f')
                
        
        else:
            control_func, control_info = self.GenerateOneCustomControlFunction(dicoGuess, dicoControl, dicoOverall, T =T)

        return control_func, control_info

    def GenerateOneCustomControlFunction(self, params1guess, params1control, params1overall, **args):
        """
            Generate one CustomControl Function
            f(t) = overall(guess(t) * fParam(t))
            where overall implement potential boundaries/rescaling
        """
        # Guess
        if(params1guess is not None):
            params1guess['T'] = args['T']
        fGuess, self.info_guess = self._GenCustomParametrizedFunction(params1guess)
        fGuess.FixParams()

        #Parametrized Function
        params1control['T'] = args['T']
        if(self._name_algo == 'D-CRAB'):
            fParam, info_control = self._GenCustomParametrizedFunction(params1control, split = True)
        else:
            fParam, info_control = self._GenCustomParametrizedFunction(params1control, split = False)
        
        # overall
        if (params1overall is not None):
            bds = params1overall.get('bounds')
            ct = params1overall.get('constraints')
            ct_type = params1overall.get('constraints_type')
            controlFun = pfunc.Times([fParam, fGuess], ct, ct_type, bds)
        else:
            controlFun = fParam

        return controlFun, info_control
    
    
    def CustomUpdateControlParameters(self, params, simulator = None,  **args):
        """
        Purpose:
            Update the paramaters of the control function, nested in the simulator
        """
        if simulator is None:
            simulator = self.simulator
        simulator.UpdateControlParameters(params, **args)

            
    def _GenCustomParametrizedFunction(self, args, split = False):
        """
        PURPOSE: 
            generate custom parametrized functions for the model:

        Parametrization type:
            + linear
            + Fourier
        Split:
            used in the case of DCRAB
        """
        if (args in [None, {}]):
            res_func = pfunc.ConstantFunc(1)
            res_dico_args = {'type':'constant', 'c0':1}

        else:
            T = args['T']
            ctts = args.get('constraints')
            ctts_type = args.get('constraints_type')
            bdries = args.get('hard_wall')
            name_function = args.get('func')
            res_dico_args = {'func': name_function, 'constraints':ctts, 
                             'constraints_type':ctts_type, 'T':T, 'hard_wall': bdries, 'indexFunToUpdate':0}
            
            if(name_function == 'constant'):
                 c0 = args.get('c0')
                 res_func = pfunc.ConstantFunc(c0)
                 res_dico_args = {'type':'constant', 'c0':c0}
                 
            # Linear function
            elif(name_function == 'linear'):
                if((ctts is not None) and (len(ctts) == 2)):
                    dico_params = ConstraintsToLinearCoeff(ctts, dico = True)
                else:
                    a = args.get('a')
                    b = args.get('b')
                    dico_params = {'a':a, 'b':b}
                    res_dico_args = ut.add_dico(res_dico_args, dico_params)
                res_func = pfunc.LinearFunc(dico_params)                    
            
            # Exponential function
            elif(name_function == 'exp'):
                raise NotImplementedError()
            
            # Fourier series type parametrization
            elif(name_function[:4] in ['four', 'sine']):
                function_class = self.DIC_NAME2FUN[name_function]
                nbH = function_class.GetNbH(self._nb_params)
                om, dico_freq = self.__GenFrequencies(T, nbH, args)
                phi = args.get('phi', 0)
                a = args.get('a', 0)
                b = args.get('b', 0)
                c0 = args.get('c0', 0)
                dico_params = {'nbP':self._nb_params, 'c0':c0, 'phi':phi, 'om':om, 'a':a, 'b':b}
                # Workaround for DCRAB optim
                if(split):
                    dico_params['c0'] = 0
                    fs = function_class(dico_params, None, None, None)                  
                    ct = pfunc.ConstantFunc(args.get('c0', 0))
                    res_func = pfunc.Plus([fs, ct], ctts, ctts_type, bdries)
                    res_dico_args['indexFunToUpdate'] = [0,0]
                else:
                    res_func = function_class(dico_params, ctts, ctts_type, bdries)
                    
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params, dico_freq)
                                    
                    
            # Chebyshev type parametrization
            elif(name_function[:5] == 'cheby'):   
                function_class = self.DIC_NAME2FUN[name_function]
                domain = args.get('domain', [0, T])
                c0 = args.get('c0', 0)
                dico_params = {'c0':c0, 'nbP': self._nb_params, 'domain':domain}
                a = args.get('a')
                b = args.get('b')
                if(a is not None):
                    dico_params['a'] = a
                if(a is not None):
                    dico_params['c0'] = b

                if(split):
                    dico_params['c0'] = 0
                    ch = function_class(dico_params, None, None, None)
                    ct = pfunc.ConstantFunc(c0)
                    res_func = pfunc.Plus([ch, ct], ctts, ctts_type, bdries) 
                    res_dico_args['indexFunToUpdate'] = [0,0]
                else:       
                    res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)
            
            #Step function
            elif(name_function[:4] == 'step'):   
                function_class = self.DIC_NAME2FUN[name_function]
                step_dt = T / self._nb_params
                step_T = np.arange(0, T, step_dt)
                dico_params = {'Tstep':step_T}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)
                
            elif(name_function[:9] == 'squareexp'):  
                function_class = self.DIC_NAME2FUN[name_function]
                dico_params = {'sigma':1.0, 'mu':T/2, 'l':T/4}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)

            else:
                raise NotImplementedError()
            res_dico_args['nb_effective_params'] = res_func._nbTotalParams
        return res_func, res_dico_args
    

    def runDCRAB(self, arg2costFun = None, argsOptim = None, writeLogs = False):
        """ 
        What it does:
            Implementation of DCRAB superoptim
            (need bounds)

        """
        nb_superit = argsOptim.get('nb_superiter', 50)
        tol_superit = argsOptim.get('tol_superiter', 0.000001)
        tol_nbNotBetter = argsOptim.get('tol_superiter', 10)
        algo_superit = argsOptim.get('algo_superiter', 'NM')
        superit_best = argsOptim.get('superit_start', 0.35)
        
        if writeLogs:
            raise NotImplementedError()
            
        if arg2costFun is None:
            arg2costFun = self.arg2costFun
        
        improvement = 0
        nb_nobetter = 0
        
        self.trace_best = []
        p_newfunctions = self._args_control.copy()
        p_newfunctions['norm'] = None
        p_newfunctions['c0'] = 0
        i2upd = p_newfunctions['indexFunToUpdate']
        
        for super_i in np.arange(nb_superit):
            
            # Time to stop
            if(nb_nobetter >= tol_nbNotBetter):
                print('D-CRAB: finish optim - last imporvement ' + str(improvement) + 'after nb_superit=' + str(super_i))
                break
            
            resTmp = self.ALGO_AVAILABLE[algo_superit](arg2costFun, argsOptim, writeLogs)
            superit_tmp = resTmp['fun']
            print(superit_tmp)
            self.trace_best.append(superit_tmp)
            improvement = superit_best - superit_tmp 
            
            # pdb.set_trace()
            # No improvement
            if(improvement < tol_superit):
                print('bad')
                nb_nobetter +=1
                fNew, _ = self._GenCustomParametrizedFunction(p_newfunctions)
                self.simulator.controlFun.ReplaceFunction(fNew, i2upd)
            
            #Improvement
            else:
                superit_best = superit_tmp
                nb_nobetter = 0
                fNew, _ = self._GenCustomParametrizedFunction(p_newfunctions)
                self.simulator.controlFun.AddNewFunctions(fNew, i2upd[0])
                
                
            print(len(self.simulator.controlFun.listFunc[0].listFunc))  

            
        resTmp['superit_nb'] = super_i
        return resTmp




    def __GenFrequencies(self, T, nb_freq = 1, params = None):
        """
        Purpose:
            Generate (potentially randomized) frequencies
        + CRAB:
        + DCRAB: Uniformly drawn in
        + None: gen frequencies k 2Pi/T
        """
        om_ref = 2 * np.pi / T
        name_rdm = params.get('name_rdm')
        seed_rdm = params.get('seed_rdm')
        dico_args = {}
        dico_args['name_rdm'] = name_rdm
        dico_args['seed_rdm'] = seed_rdm
        
        if(name_rdm is not None):
            args_rdm = ut.splitString(name_rdm)
            if(args_rdm[0] == 'CRAB'):
                if(len(args_rdm) == 1):
                    rdv_method = 'uniform_-0.5_0.5'  
                elif(len(args_rdm) == 2):
                    val = str(args_rdm[1])
                    rdv_method = ut.concat2String('uniform', -val, val)
                elif(len(args_rdm) == 3):
                    distrib = val = str(args_rdm[1])
                    val = str(args_rdm[2])
                    rdv_method = ut.concat2String(distrib, -val, val)
                elif(len(args_rdm) == 4):
                    rdv_method = ut.concat2String(args_rdm[1], args_rdm[2], args_rdm[3])                        
                rdvgen = ut.getRandomGenerator(method = rdv_method, nb = nb_freq, seed = seed_rdm)
                om = (1 + np.arange(nb_freq) + rdvgen()) * om_ref
                    
            elif(args_rdm[0] == 'DCRAB'):
                if(len(args_rdm)>1):
                    Nmax = int(args_rdm[1])
                else:
                    Nmax = nb_freq
                wmax = Nmax * om_ref
                rdv_method = ut.concat2String('uniform', 0, wmax)  
                rdvgen = ut.getRandomGenerator(method = rdv_method, nb = nb_freq, seed = seed_rdm)
                om = rdvgen()                            
                dico_args['flag'] = 'DCRAB'
        else:
            om = (1 + np.arange(nb_freq)) * om_ref

        dico_args['omegas'] = om
        return om, dico_args

#==============================================================================
#                   Auxilliary function
#============================================================================== 
def ConstraintsToLinearCoeff(constraints, dico = False):
    if(len(constraints) == 2):
        x0, y0 = constraints[0]
        x1, y1 = constraints[1]
        
        a = (y1 - y0)/(x1 - x0)
        b = y0 - a*x0
    else:
        raise NotImplementedError()
    
    if(dico):
        res = {'a':a, 'b':b}
    else:
        res = [a, b]
    
    return res

#==============================================================================
#                   DEPRECIATED
#============================================================================== 

