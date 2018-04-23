#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the Simultaion.lightSimulation class)
#  
#
#
#
#============================================================================== 
from . import LightSimulation as lsim
from ...Utility.Optim import Optim as opt
from ...Utility.Misc import Params as pms
from ...Utility.Quantum import Hamiltonian as ham

import numpy.random as rdm
import numpy as np
import pdb as pdb #debugger 
import copy as cp

class ToyModelOptim(opt.AbstractOptim):
    """ 
    Implementation of the abstract optimization for the toyModel (simulated via
    the lightSimulation class)
    
    BUGS:
        Testing works as long as some params are not changed between simul and testing
    
    """
    # parameters MANDATORY to initialise the simulator (should be maintained)
    LIST_PARAMS_SIM_NEEDED = {}
    LIST_PARAMS_SIM_NEEDED['setup'] =  "setup of the quantum systems dimensions (cf. Utility.Params)"
    LIST_PARAMS_SIM_NEEDED['eta'] =  "coupling between motional internal (lamb dicke param)"
    LIST_PARAMS_SIM_NEEDED['tgt'] =  "amplitude of the target dynamics"
    LIST_PARAMS_SIM_NEEDED['tgt_type'] =  "type of target dynamics (e.g. 'sigmaX')"
    LIST_PARAMS_SIM_NEEDED['tgt_picture'] =  "In which frame the target is realized (None, 'initial', 'strong', 'interaction')"
    LIST_PARAMS_SIM_NEEDED['nPPP'] =  "Number of points to be simulated per period (e.g. 1 for strobo simulations)"
    LIST_PARAMS_SIM_NEEDED['nPeriod'] = "(or nPeriodEff) nb of (effective) periods to be simulated"
    LIST_PARAMS_SIM_NEEDED['nPeriodEff'] = "cf. nPeriod"
    LIST_PARAMS_SIM_NEEDED['type_simul'] = "how the system is simulated, by default 'FNS' (could be 'FME', 'HFE', 'VV') "

    # parameters OPTIONAL to initialise the simulator (should be maintained)  
    LIST_PARAMS_SIM_OPT = {}
    LIST_PARAMS_SIM_OPT['_skipMotionalError'] = "Numerical params - if True accept simulation even if it doesn't comply with the test"
    LIST_PARAMS_SIM_OPT['decimalRounding'] = "Numerical params - used to assess degeneracy of the eigenvalues"
    LIST_PARAMS_SIM_OPT['maxPopLastMotional'] = "Numerical params - maximum tolerated pop in the maximal fock state"
    LIST_PARAMS_SIM_OPT['maxDevSumProba'] = "Numerical params - maximum tolerated deviation of the sum of the pop compared to 1"
    LIST_PARAMS_SIM_OPT['IterMax'] = "nb of run per simulation (if they are rejected due to numerical reason)"    
    LIST_PARAMS_SIM_OPT['init_state'] = "init state"
    LIST_PARAMS_SIM_OPT['weights_FOM'] = "How to generate weights used to compute the FOM ('expDecay')"
    LIST_PARAMS_SIM_OPT['encoding_method'] = "How to encode the parameters"
    

    def __init__(self, paramsSimulation = None, paramsOptim = None, paramsSimulationTest = None):
        # call of self.initSimulator from the parent class
        opt.AbstractOptim.__init__(self, paramsOptim, paramsSimulation, paramsSimulationTest)
        

        
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
        Comments:
            Extra infos are generated
        """
        sim, pSim, nPeriods, nPeriodEff, n_pppEff, type_simul = self.getSimulator(p)

        #weights for computing the FOM
        tgt, tgtType, tgtPicture = self.addTargetToSimul(p, sim)

        args={'tgt': tgt}
        args['tgt_type'] = tgtType
        args['tgt_picture'] = tgtPicture
        args['methodEncoding'] = p.get('encoding_method')
        args['type_simul'] = type_simul
        args['weights_FOM'] = self.getWeightsToSimul(p, sim)
        args['fom_type'] = self.name_FOM
        args['setup'] = p.get('setup', None)
        
        args['scaling_fluence'] = 2 * np.square(tgt * pSim.n_int * pSim.omega)

        return sim, args


    def getSimulator(self, p):
        """
        Purpose:
            Custom initialization of the simulator based on the paramsSimulation
            provided. 
        Comments:
            
        """
        # Extract params for the simulation
        pSim, nPeriods, nPeriodEff, n_pppEff, type_simul = self.getParamsForSim(p)
        init_state_type = p.get('init_state')
        init = self.getInitialState(pSim, init_state_type)           
        
        # Generate the simulator 
        nbIterMax = p.get('IterMax', 1) #By def 1 iter
        sim = lsim.LightSimulation(pSim, init, iterMax = nbIterMax)
        
        #OverWrite some params if provided 
        for k in self.LIST_PARAMS_SIM_OPT:
            if((k in p) and hasattr(sim,k)):
                setattr(sim, k, self.params_simulation[k])

        return sim, pSim, nPeriods, nPeriodEff, n_pppEff, type_simul


    def generateNameSimul(self, name_method = None, args_simul = None):
        """
        What it should do:
            generate a name for the simulation (used to tag/store the results) 

        """
        setup = str(args_simul['setup'])
        fom = str(args_simul['fom_type'])
        tgt = str(args_simul['tgt'])
        tgtType = str(args_simul['tgt_type'])
        tgtPicture = str(args_simul['tgt_picture'])
        typeSimul = str(args_simul['type_simul'])        
        
        if (name_method is None):
            if (isinstance(fom, list)):
                fom = fom[0] # use first one as that is the one used for optim
            name = ut.concat2String(self.name_algo, 'Setup', setup, 'tgt', tgt)
            name = ut.concat2String(name, tgtType, tgtPicture, 'FOM', fom)
            name = ut.concat2String(name, 'typeSimul', typeSimul)
            
        else:
            raise NotImplementedError()
        return name


    
    def optimWrapper(self, simulator, args):
        """
        Purpose:
            Wrap the simulator to take arguments to be optim and return the figure of merit

        Output:
            funCost: Function <argsToOptim> -> FOM

        """
        tgt = args['tgt']
        method_encoding = args['methodEncoding']
        type_simul = args['methodEncoding']
        weights = args['weights_FOM']
        fomType = args['fom_type']
        scalingFluence = args['scaling_fluence']
        
        if isinstance(fomType, str):
            fomType = [fomType]
    
        def arg2costFun(params):
            simulator.setupCfc(params = params, controlType = None, scaling = tgt, methodEncoding = method_encoding)
            fom = simulator.runFOM(listRes = fomType, typeRun = type_simul, scaling_fluence = scalingFluence, weights_fom = weights)
            return fom
        return arg2costFun
                    



    
    def getInitParams(self, method = None):
        """
        TODO: Clean it up / Put in the optim class????
        What it should do:
                generate initial parameters for the optimization 
        Methods:
            + 'uniform_A_B' for each params value drawn from Unif(A, B)
            + 'normal_A_B' for each params value drawn from Normal(A, B)
            + 'zero' for each params 0
            + 'force_LIST' LIST is directly used as the init parameters
            +
        """        
        
        if method is None:
            init_args = None
        else:
            args = ut.splitString(method)
            if(args[0] == 'uniform'):
                init_args = [rdm.uniform(low = args[1], high = args[2]) for _ in range(self.nb_params)]
            elif(args[0] == 'normal'):
                init_args = [rdm.normal(loc = args[1], scale = args[2]) for _ in range(self.nb_params)]
            elif(args[0] == 'force'):
                init_args = ut.listFloatFromString(args[1])
            elif(args[0] == 'zero'):
                init_args = np.zeros_like(self.nb_params)
                
            ### For the initialization of NM    
            elif(args[0] == 'nmguess'):
                # for the NM algo we need to come up with N+1 vertices arround a guess
                # Based on Matlab routine
                # e.g. nmguess_[3,0,1]_0.5_1
                assert (len(args)>=2), 'nmguess, not the right format'
                if(len(args) == 2):
                    step0 = args[1]
                    stepNon0 = args[1]
                elif(len(args) == 3):
                    step0 = args[1]
                    stepNon0 = args[2]
                    
                init_args = np.zeros_like([self.nb_params + 1, self.nb_params])
                guess = ut.recastString(args[1])
                if(isinstance(guess, str)):
                    guess_vector = self.getInitParams(guess)
                elif(isinstance(guess, list)):
                    guess_vector = np.array(guess)
                    
                init_args[0,:] = guess_vector
                for i in range(self.nb_params):
                    perturb = np.zeros(self.nb_params)
                    if(guess[i] == 0):
                        perturb[i] = step0
                    else:
                        perturb[i] = stepNon0 
                    init_args[(i+1), :] = init_args[0,:] + perturb
                
                return init_args
                 
            elif(args[0] == 'nmuniform'):
                # Drawn from an uniform distrib
                # e.g. nmnormal_-1_1 or nmnormal_-5_5_0_3_-2_2 (for3 params)
                if(len(args) == 3):
                    init_args = rdm.uniform(args[1], args[2], [(self.nb_params+1, self.nb_params)])
                elif(len(args) == (2*self.nb_params + 1)):
                    init_args = [rdm.uniform(args[1+(2*i)], args[2 + 2*i], (self.nb_params+1)) for i in range(self.nb_params)]
                    init_args = np.transpose(init_args)
                else:
                    raise NotImplementedError()

            elif(args[0] == 'nmnormal'):
                # Drawn from an uniform distrib
                # e.g. nmnormal_-1_1 or nmnormal_-5_5_0_3_-2_2 (for3 params)
                if(len(args) == 3):
                    init_args = rdm.uniform(args[1],args[2], [(self.nb_params+1, self.nb_params)])
                elif(len(args) == (2*self.nb_params + 1)):
                    init_args = [rdm.normal(args[1+(2*i)], args[2 + 2*i], (self.nb_params+1)) for i in range(self.nb_params)]
                    init_args = np.transpose(init_args)
                else:
                    raise NotImplementedError()
                
            elif(args[0] == 'nmguesses'):
                # Directly make guesses of the N+1 vertexes
                # e.g. nmguesses_[[1,2],[2,3],[5,5]]
                init_args = np.array(ut.matrixFloatFromString(args[1]))
                assert (init_args.shape == (self.nb_params+1, self.nb_params)), 'nmguesses: not the right dim'
                
            else:
                #workaround TO BE CHECKED: 
                init_args = np.zeros(self.nb_params)
                cfc_tmp = self.simulator.getCfc(params = None, controlType = method, scaling = 1)
                vals_tmp = cfc_tmp[1]
                nbHarmo = len(cfc_tmp[1])
                if nbHarmo == 1:
                    init_args[0] = vals_tmp[0]
                else:
                    #pdb.set_trace()
                    nb_to_keep = min(int(nbHarmo / 2), self.nb_params)
                    init_args[:nb_to_keep] = vals_tmp[int(nbHarmo / 2):int(nbHarmo / 2)+nb_to_keep]
                    
        return init_args


# ---------------------------
# Implementation of new specific methods:
#   + addTargetToSimul : generate Target and append it to the simul objects
#   + addWeightsToSimul: generate weights for FOM and append it to the simul objects
# ---------------------------

    
    def addTargetToSimul(self, params, sim):
        """
        Purpose:
            add to a simulation (sim) a target built from the params dico 
            (paramSimulation)
        Output:
            STORE target value (tgt) 
            ADD the target to the simulator
        """    
        tgt = params['tgt']
        tgtType = params['tgt_type']
        tgtPicture = params['tgt_picture']
        
        if(tgtType == 'sigmaX'):
            om = sim.model.om
            n0 = sim.model.n0
            f0 = tgt * n0 * om
            htgt = ham.TIHamiltonian(sim.p, f0 * sim.fss.sigmaP + np.conjugate(f0) * sim.fss.sigmaM)
            sim.addTarget(htgt, tgtPicture)
            
        elif(tgtType in ['VV', 'HFE']):
            pCopy =  cp.copy(params)
            pCopy['type_simul'] = tgtType
            simTgt, _, _, _, _, _, _ = self.getSimulator(pCopy)
            #TODO: change
            simTgt.controlPicture = 'interaction'
            _, a, _, b = simTgt.runHFESimul()
            sim.htarget = simTgt.hFME

            if (tgtPicture == 'initial'):
                sim.stateTGT = a
            else:
                sim.stateTGT = b
                
        else:
            raise NotImplementedError()
        #Setup target
        
        return tgt, tgtType, tgtPicture


    def getWeightsToSimul(self, params, sim):
        """
        Purpose:
            generate (normalized) weightings for the computation of the FOM such 
            that we can put emphasis on some part of the dynamics (e.g could be 
            used to put more emphasis at the beginning) 
            e.g. weighted fidelity is computed as sum w(t)*F(t)
        OPtion:
            expDecay_float
            expDecayEff_float
            force_[vals]
        Output:
            STORE weights in the object
        """    
        method = params.get('weights_FOM')
        if (method is None):
            Wnorm = None
        else:
            bits = ut.splitString(method)
            nbPoints = len(sim.p.array_time)
            if(bits[0] == 'expDecay'):
                
                if (len(bits) == 2):
                    timeRatio = 0.1
                    weightAlloc = float(bits[1])
                    lambdaval = -np.log(1-weightAlloc) / (timeRatio * nbPoints)
                    
                else:
                    # want 30% of the weights allocated over the first 10% of the simul
                    timeRatio = 0.1
                    weightAlloc = 0.3
                    lambdaval = -np.log(1-weightAlloc) / (timeRatio * nbPoints)
                    #pdb.set_trace()
                W = np.exp(-lambdaval * np.arange(nbPoints))
    
            elif(bits[0] == 'expDecayEff'):
                raise NotImplementedError()
    
            elif(bits[0] == 'force'):
                W = np.array(ut.listFloatFromString(bits[1]))
                
            elif(bits[0] == 'steps'):
                if(len(bits) == 1):
                    n_discr = 5
                    
                elif(len(bits) == 2):
                    n_discr = bits[1]
                
                else:
                    raise NotImplementedError()                
                
                W = np.array([(n_discr - np.floor(i * n_discr/nbPoints)) for i in range(nbPoints)])
                    
            else:
                raise NotImplementedError()
    
            Wnorm = W / np.sum(W)
        sim.Wnorm = Wnorm
        return Wnorm


# ---------------------------
# Some extra methods
# + prepareParamsForSim
# + initInitialState
# ---------------------------
    def getParamsForSim(self, paramsSimulation):
        """
        Purpose:
            translate the parameters contained in the paramsSimulation dico into 
            the right structure for the initialisation of the simulator
        """
        # Properties of the QSystem
        etaGlobal = paramsSimulation['eta']
        setup = paramsSimulation['setup']
        type_simul = paramsSimulation.get('type_simul', None)
                
        # Time
        pDummy = pms.genSetupInter(setup = setup, gauge = type_simul) #Just to recup n_int    
        tgt = paramsSimulation['tgt']
        nPPPGlobal = paramsSimulation['nPPP']
        if ('nPeriod' in paramsSimulation):
            nPeriods = paramsSimulation['nPeriod']         
            nPeriodEff = max(1, nPeriods * tgt * pDummy.n_int)
        else:
            assert ('nPeriodEff' in paramsSimulation), 'Need either nPeriod or nPeriodEff'
            nPeriodEff = paramsSimulation['nPeriodEff'] # define length of the simul
            nPeriods = max(1, nPeriodEff / (tgt * pDummy.n_int))
            

        # Generate the parameters for the simul and Overwrite some parameters if provided
        pSim = pms.genSetupInter(setup, gauge = type_simul, n_period = nPeriods, n_ppp = nPPPGlobal, eta = etaGlobal)
        for k in self.LIST_PARAMS_SIM_OPT:
            if((k in paramsSimulation) and hasattr(pSim, k)):
                setattr(pSim, k, paramsSimulation[k])

        #store some extra info
        n_pppEff = int(nPPPGlobal  / (tgt * pDummy.n_int))
        #TODO: for it to be used to build the weighting of the FOM
        #self.array_index_Eff = 

        return pSim, nPeriods, nPeriodEff, n_pppEff, type_simul

    def getInitialState(self, params, initState = None):
        """
        Purpose:
            either pull out from the params an initial state or use by default 
            (1,0,0,...) - g0 state with the rights diemnsion
        """
        if(initState is not None):
            init = ut.listFloatFromString(initState)
        else:
            init = np.zeros(params.n_fockmax * params.n_intState, dtype = 'complex128')
            init[0] = 1
        return init



#==============================================================================
#                   DEPRECIATED
#============================================================================== 


#    def testResults(self, res, writeLogs= False):
#        """
#        What it should do:
#            test optimal parameters found and add the new results (namely FOM 
#            requested to the res object)  
#
#        """
#        if(writeLogs):
#            raise NotImplementedError()
#
#        #Use the testing simulator
#        params = self.params_simulation_testing
#        simulator = self.simulator_testing
#
#        # If a specific FOM has been specified for the testing use it, if not 
#        # use the one used for the optim
#        FOM = params.get('test_FOM', self.name_FOM)
#
#        arg2costFun = self.optimWrapper(simulator, FOM)
#        
#        #Use optimal_params found during the optimizations
#        init = res['optimal_params']
#        if(init is None):
#            print('No Optimal Parameters found, testing is done based on params_testing')
#            init_method = params['params_testing']
#            init = self.getInitParams(init_method)
#        res['test_params'] = init
#
#        #run the simulator     
#        resNew = arg2costFun(init)
#        if(isinstance(resNew, list)):
#            for i in range(len(resNew)):
#                name_tmp = 'test_' + FOM[i]
#                res[name_tmp] = resNew[i]
#        else:
#            name_tmp = 'test_' + str(FOM)
#            res[name_tmp] = resNew





