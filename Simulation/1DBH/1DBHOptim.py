#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
#
#============================================================================== 
import sys
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import ParametrizedFunctionFactory as pfunc
    from QuantumSimulation.Utility.Optim import OptimML as opt
    from QuantumSimulation.ToyModels import BH1D as bh
    
else:
    from ...ToyModels import BH1D as bh
    from ...Utility import Helper as ut
    from ...Utility.Optim import OptimML as opt
    from ...Utility.Optim import ParametrizedFunctionFactory as pfunc


import numpy as np
import pdb as pdb #debugger 
import importlib as ilib
import time
import copy as cp
ilib.reload(opt)
ilib.reload(bh)
ilib.reload(ut)


#TODO: Deal with setup2
#TODO: Test setup 3
class BH1DOptim(opt.AbstractOptimML):
    """ 
    Implementation of the abstract optimization for the toyModel (simulated via
    the lightSimulation class)
    
    BUGS:
        Testing works as long as some params are not changed between simul and testing
    
    """
    # parameters MANDATORY to initialise the simulator (should be maintained)
    LIST_PARAMS_SIM_NEEDED = bh.BH1D.MODEL_PARAMS_DEF
    

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
        """ Gen the control function and the 1D Bose Hubbard model chain
        Input:
            p : (dico/textfile) which contains the config of the simulation        
        Output:
            Simulator (model we want to simulate which has been initialized)
            p: 
            

        """
        # pdb.set_trace()
        # Build the parametrized control function 
        custom_control, args_control = self.GenerateCustomControlFunction(p)
        sim_args = ut.merge_dico(p, args_control, update_type = 3)
        # Initializethe simulator        
        simulator = bh.BH1D(controlFun = custom_control, model_params = p)
        self._args_control = args_control
        return simulator, sim_args


    def GenNameSimul(self, name_method = None, args_simul = None, args_optim = None):
        """ Generate a name for the simulation (used to tag/store the results) 
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
                # self._randomize_simulator(simulator)
                fom = simulator.Simulate(fom = fom_name)
                
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
                # self._randomize_simulator(simulator)
                
                fom = simulator.Simulate(fom = fom_name)
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
# CUSTOM control function
# ---------------------------
    def GenerateCustomControlFunction(self, params):
        """
        PURPOSE: 
            generate (potentially several) custom parametrized control functionS
        """
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
            ow_X = params1overall.get('ow_X')
            ow_Y = params1overall.get('ow_Y')
            controlFun = pfunc.Times([fParam, fGuess], ct, ct_type, bds, ow_X, ow_Y)
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
            
            elif(name_function == 'expramp'):
                a = args['ampl']
                T = args['T']
                l = args['l']
                dico_params = {'ampl':a, 'T':T, 'l':l}
                res_func = pfunc.cFixedExpRamp()
            
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
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    flag_PW = True
    flag_CRAB_CL = False
    flag_CRAB_OL = False
    
    T = 5
    L=5
    N=L
    name_fom_main = ['f2t2:neg_fluenceNorm:0.0001_smooth:0.01']
    name_fom_testing = ['f2t2:neg_fluenceNorm:0.0001_smooth:0.01', 'f2t2', 'fluenceNorm', 'smooth2', 'varN']
    name_fom_re = ['last:f2t2:neg_fluenceNorm:0.0001_smooth:0.01', 'last:f2t2', 'fluenceNorm', 'smooth2', 'last:varN']

    paramsSim = {'L':L, 'N':N, 'mu':0, 'T':T, 'dt':0.1, 'flagInterTime':False,
                     'setup':'1', 'state_init':'GS_i', 'state_tgt':'11111',
                     'fom_name':name_fom_main, 'noise':None}

        

    if(flag_CRAB_CL):
        paramsSim['overall'] = {'bounds':[0,1], 'ow_X':[0,T], 'ow_Y':[0,1]}
        paramsSim['guess'] = {}
        paramsSim['control'] = {'func':'step'}        
        
    if(flag_PW):
        paramsSim['overall'] = {'bounds':[0,1], 'ow_X':[0,T], 'ow_Y':[0,1]}
        paramsSim['guess'] = None
        paramsSim['control'] = {'func':'step'}
    
    
        paramsOptim_DE =  {'algo':'DE','name_res':None,'nb_params':30,'params_init':'uniform_0_1',
                           'params_bound':'range_0_1','de_maxiter':100,'_FLAG_NAME':'DE', 'de_popsize':2} 
    
        paramsOptim_GP = {'algo':'GP', 'name_res':None,'nb_params':15,'params_init':'uniform_0_1',
                          'params_bound':'range_0_1','gp_init’:40,’gp_acq':'ei','gp_kernel':None,
                          'gp_wnoise':0.01,'gp_maxiter':300,'_FLAG_NAME':'GP','flag_MP':7} 
        
        paramsOptim_NM = {'algo':'NM','name_res':None,'nb_params':15,'params_init':'uniform_-2_2', 
                          'params_bound':'range_0_1','nm_maxiter':5000,'nm_maxfev':15000,'nm_ftol':1e-6,
                          '_FLAG_NAME':'NM','flag_MP':7}
    
        paramsTesting = cp.copy(paramsSim)
        paramsTesting['fom_name'] = name_fom_testing

    
        # runOptim
        optim_DE = BH1DOptim(paramsSim, paramsOptim_DE, paramsTesting)
        res_DE = optim_DE.Run()
        
        #look at f(t)
        cfBest = optim_DE.simulator.controlFun
        x = np.arange(0, T +0.01 , 0.01)
        cfBest.PlotFunction(x)
        params_ottimo = res_DE['opt_params']
    
        
        #Recreate testing condition 1 
        import copy 
        p1 = copy.copy(paramsSim)
        p1['flagInterTime'] = True
        cf1, _ = optim_DE.GenerateCustomControlFunction(p1)
        s1 = bh.BH1D(controlFun = cf1, model_params = p1)
        s1.UpdateControlParameters(params_ottimo)
        s1.state_t = s1.EvolutionPopAdiab(nb_ev = 10)
        s1.plot_pop_adiab()
        s1.Simulate(fom = name_fom_re, store= True)
        st1 = s1.state_t
        
        p2 = copy.copy(paramsSim)
        p2['flagInterTime'] = False
        s2 = bh.BH1D(controlFun = cf1, model_params = p2)
        s2.UpdateControlParameters(params_ottimo)
        s2.Simulate(fom = name_fom_testing, store=True)
        st2 = s2.state_t
        
        p3 = copy.copy(paramsSim)
        p3['flagInterTime'] = True
        p3['dt'] = 0.01
        s3 = bh.BH1D(controlFun = cf1, model_params = p3)
        s3.UpdateControlParameters(params_ottimo)
        s3.Simulate(fom = name_fom_re, store=True)
        st3 = s3.state_t
    
        # Recreate testing condition 2
        # s2 = optim_DE.simulator_testing
        # s2.state_t = s1.EvolutionPopAdiab(nb_ev = 10)
        # s2.plot_pop_adiab()