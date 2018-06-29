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


class BH1DOptim(opt.AbstractOptimML):
    """ Implementation of the abstract optimization for the toyModel (simulated via
    the lightSimulation class)
    
    BUGS:
        Testing works as long as some params are not changed between simul and testing
    
    """
    # parameters MANDATORY to initialise the simulator (should be maintained)
    LIST_PARAMS_SIM_NEEDED = bh.BH1D.LIST_ARGS
    
    def __init__(self, paramsSimulation = None, paramsOptim = None, paramsSimulationTest = None):
        # call of self.initSimulator from the parent class
        self.nbControlFunPerSetUp = {'1':1}
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
        simulator = bh.BH1D(**p)
        sim_args = ut.merge_dico(p, simulator.info_control, update_type = 3)
        return simulator, sim_args


    def GenNameSimul(self, name_method = None, args_simul = None, args_optim = None):
        """ Generate a name for the simulation (used to tag/store the results)
        based on the args of the simulations and optimizations
        """
        setup = str(args_simul['setup'])
        fom_name = args_simul['fom_name']
        algo = args_optim['algo']
        
        if (name_method is None):
            if (isinstance(fom_name, list)):
                fom_name = fom_name[0] 
            name = ut.concat2String(algo, 'Setup', setup)
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
        argsUpdateParams = {'index_fun':args.get('indexFunToUpdate')}
        
        if(write_logs):
            # Some infos are obtained directly here
            self._nb_call_fev = 0
            self._timer = time
            self._time_init = self._timer.time()
            self._bestfom_nbev = []
            self._bestfom_time = []
            self._bestfom_func = []
            
            def arg2costFun(params):
                simulator.update_control_parameters(params, **argsUpdateParams)
                self._randomize_simulator(simulator)
                fom = simulator.Simulate(fom = fom_name)
                
                if(printVal):
                    print(params, fom)
                
                self._nb_call_fev += 1
                fom_best_so_far = self.best.get('fom')
                
                if(not(flagTesting)):
                    if (fom_best_so_far is None) or (fom_best_so_far > fom):
                        self.best['fom'] = fom
                        self.best['params'] = params
                        self.best['omegas'] = args.get('omegas')
                        self.best['fun'] = repr(simulator.controlFun)
                        self._time_elapsed = self._timer.time() - self._time_init
                        self._bestfom_nbev.append([self._nb_call_fev, fom])
                        self._bestfom_time.append([self._time_elapsed, fom])
                        self._bestfom_func.append([repr(simulator.controlFun), fom])
    
                return fom
            
            
        else:
            def arg2costFun(params):
                simulator.update_control_parameters(params, **argsUpdateParams)
                self._randomize_simulator(simulator)
                
                fom = simulator.Simulate(fom = fom_name)
                if(printVal):
                    print(params, fom)
                
                fom_best_so_far = self.best.get('fom')
                if(not(flagTesting)):
                    if (fom_best_so_far is None) or (fom_best_so_far > fom):
                        self.best['fom'] = fom
                        self.best['params'] = params
                        self.best['omegas'] = args.get('omegas')
                
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
        pass
        


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
    seed = 1255656112
    name_fom_main = ['f2t2:neg_fluenceNorm:0.0001_smooth:0.01']
    name_fom_testing = ['f2t2:neg_fluenceNorm:0.0001_smooth:0.01', 'f2t2', 'fluenceNorm', 'smooth', 'varN']
    name_fom_re = ['last:f2t2:neg_fluenceNorm:0.0001_smooth:0.01', 'last:f2t2', 'fluenceNorm', 'smooth', 'last:varN']

    paramsSim = {'L':L, 'N':N, 'mu':0, 'T':T, 'dt':0.1, 'flagInterTime':False,
                     'setup':'1', 'state_init':'GS_i', 'state_tgt':'11111',
                     'fom_name':name_fom_main, 'noise':None}

        

    if(flag_CRAB_CL):
        paramsSim['overall'] = {'bounds':[0,1], 'ow_X':[0,T], 'ow_Y':[0,1]}
        paramsSim['guess'] = {}
        paramsSim['control'] = {'func':'step'}        
        
        
        
        
    if(flag_PW):
        controlFun_dico = {'T':T}
        controlFun_dico['overall'] = {'bounds':[0,1], 'ow_X':[0,T], 'ow_Y':[0,1]}
        controlFun_dico['guess'] = None
        controlFun_dico['control'] = {'func':'step', 'nbP':15} 
    
        paramsSim['controlFun'] = controlFun_dico 
    
    
        paramsOptim_DE =  {'algo':'DE','name_res':None, 'nb_params':15, 'seed':seed,
                           'params_init':'uniform_0_1', 'params_bound':'range_0_1',
                           'de_maxiter':100,'_FLAG_NAME':'DE', 'de_popsize':2} 
    
        paramsOptim_GP = {'algo':'GP', 'name_res':None, 'nb_params':15,
                          'params_init':'uniform_0_1', 'params_bound':'range_0_1', 
                          'gp_init’:40,’gp_acq':'ei','gp_kernel':None,'gp_wnoise':0.01,
                          'gp_maxiter':300,'_FLAG_NAME':'GP','flag_MP':7, 'seed':seed} 
        
        paramsOptim_NM = {'algo':'NM','name_res':None, 'nb_params':15,'flag_MP':7,'seed':seed,
                          'params_init':'uniform_-2_2', 'params_bound':'range_0_1',
                          'nm_maxiter':5000,'nm_maxfev':15000,'nm_ftol':1e-6, '_FLAG_NAME':'NM'}
    
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
    
        
        
    if(False):       
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