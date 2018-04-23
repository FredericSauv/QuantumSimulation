# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:51:45 2017

@author: fs
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:36:57 2017

@author: fs
"""
from ...Utility import Helper as ut
from ...ToyModels import ToyModel as toy
from ...Utility.Misc import Errors as error

import numpy as np
import matplotlib.pyplot as plt
import pdb

#==============================================================================
#                   LightSimulation
#    PURPOSE:
#        Simulate the toy model 
#        Generate FOM (figure of merit) based on results
#
#    MAIN METHODS:
#        - init : 
#         - faddT
#        - add target :
#        - getSomeResults :
#        - getExtraResults : (should be spin off)
#        -
#        - compare 
#==============================================================================
class LightSimulation:
    """ 
    """
    
    LIST_TYPE = list(['FNS', 'FME_NF', 'FME_K_NF', 'FME', 'FME_K', 'TGT_NF', 'TGT'])
    #LIST_RES_TYPE = list(['logFidelity', 'Fidelity', 'Heating'])
    LIST_RESULTS_ERROR = {'logFidelity': -6, 'Fidelity':0.0, 'NeglogFidelity': 1.0, 'NegFidelity':1.0, 'Heating': 1.0}
    
# ---------------------------
# INIT // SETUP FUNCTIONS
#----------------------------    
    def __init__(self, params, initState, iterMax = 5, sizeMax = 10000, cfc = None): 
        #        
        self.p = params
        self.init = initState

        # 
        self._nIterMax = iterMax
        self._sizeKmax = sizeMax
        self.d_rounding = params.decimalRounding
        self._ran = False
        self._resgen = False
        self._skipMotionalError = False

        #
        self.genModel(self.p, self.init)
        if (hasattr(self.p, 'gauge')):
            self.gauge = self.p.gauge
        else:
            self.gauge = None

        # 
        self.cfc = cfc
        
        #
        self.htarget = None
        self.stateTGT = None
     
    def setupCfc(self, params = None, controlType = None, scaling= None, methodEncoding = None):
        """
        Purpose:
            generate and setup the periodic control function (cfc is a list of
             {list_harmo, list_amplitudes}):
            
        """

        self.cfc = self.getCfc(params, controlType, scaling, methodEncoding)


    def getCfc(self, params = None, controlType = None, scaling = 1, methodEncoding = None):
        """
        Purpose:
            generate the periodic control function (cfc is a list of
             {list_harmo, list_amplitudes})either by either:
            + Encoding an array of paramteres (params) into a cfc 
            + GENERATE custom cfc based on the controlType chosen
            
        Remark:
            if params has alreay the shape of a cfc (i.e. a list of list then 
            it is returned without any change)

        """
        if(params is not None):
            if(isinstance(params[0], list)):
                cfc = params
            else:
                cfc = self.encodeParams2Cfc(params, scaling, methodEncoding)
        else:
            assert (controlType is not None), "getCfc: should provide  either params or controlType"
            cfc = self.getCustomCfc(controlType, scaling)
            
        return cfc


    def encodeParams2Cfc(self, params, scaling = 1, methodEncoding = None):
        cfc = self.model.encodeCustomControl(params, scaling, method = methodEncoding)
        return cfc

    def getCustomCfc(self, controlType, scaling = 1):
        _, _, cfc, _ = self.model.getControlFunction(controlType, scaling)
        return cfc

# ---------------------------
# MAIN FUNCTIONS
#   -runHFESimul: Use of the HFE/MAGNUS approximation to simulate the system
#   -runFNS: Use the FloquetNumerical Simulator to simulate the system
#   -computeFOM: run FNS and compute custom figure of merit 
# ---------------------------
    def runHFESimul(self):
        """
        Purpose:
            evolve the initial state according to the dynamics induced by the 
            HFE/FME approximation
        Steps: 
            (1) Build the effective Hamiltonian in the New Frame (NF)
            (2-3) Compute HEff Keff (FME here) in NF and store them
            (2-3) Evolve the state in this NF according to the effective Ham
            (4) Get back to initial picture
        Output:
            STORE effective Hamiltonian / micromotion operator
            RETURN states over time (taking into account the micromotion or not 
                                       x in NF or lab Frame)
        """
        # generate effecive hamiltonian in the New Frame (NF)
        self.hEffNF = self.model.genHfc(self.cfc, self.controlPicture)

        # generate Effective dynamic
        state_t_FME_NF, state_t_FME_withK_NF  = self.hEffNF.magnusEvol(self.init, gauge = self.gauge)
        self.hFME = self.hEffNF.Heff
        self.kFME = self.hEffNF.Keff

        # Back to Init Picture
        state_t_FME = self.model.changeOfPicture(state_t_FME_NF, self.controlPicture, 'initial')
        state_t_FME_withK = self.model.changeOfPicture(state_t_FME_withK_NF, self.controlPicture, 'initial')

        return state_t_FME, state_t_FME_withK, state_t_FME_NF, state_t_FME_withK_NF


    def runFNS(self):
        """
        Purpose:
            evolve the initial state based on Floquet Numerical Simulator 
            (Exact simulation - up to decimal rounding accuracy)
        Output: 
            STORE hfcInit : that's the Hamiltonian Fourier Coefficients in the 
                            initial frame
            RETURN success 
        TODO: Need to think more carefully about errors and managaement of them
        NotUnitary / TruncMotional
        BNZError       
        """
        self._ran = 0
        self._Iter = 0
        success = False
        #keep track of the changes of the dim of the problem to resize initState and TargetState Accordingly
        initial_nmot, initial_ntrunc = self.p.n_fockmax, self.p.n_trunc
        initial_init = np.array(self.init)
        
        temp_nmot = self.p.n_fockmax
        new_nmot, new_ntrunc = self.p.n_fockmax, self.p.n_trunc
        
        
        while(not(success) and self._Iter < self._nIterMax): 
            if ((new_nmot * (2 * new_ntrunc + 1) ) > self._sizeKmax):
                print('Diago of a matrix too big')
                raise error.TooBigError()
            self._Iter+=1
            if(self._Iter>1):
                print('iter: ' + str(self._Iter) + ' Regen model with nmot: ' + str(new_nmot) +' ,nTrunc: ' + str(new_ntrunc) )
                diff_nmot = int(new_nmot - temp_nmot)
                #self.fss.changeNMot(new_nmot)
                self.init = np.concatenate((self.init[:temp_nmot], np.zeros(diff_nmot, dtype = 'complex128'), self.init[temp_nmot:], np.zeros(diff_nmot, dtype = 'complex128')))
                self.p.n_trunc = new_ntrunc
                self.p.n_fockmax = new_nmot
                self.genModel(self.p, self.init)
                temp_nmot = new_nmot
                # if not(self.stateTGT is None):
                #     self.stateTGT = np.concatenate((self.stateTGT, np.zeros(int(diff_nmot * self.fss.n_int), dtype = 'complex128')))
                    #Don't need to regenerate control right??

            # Main: gen harmonics of the Hamiltonian in the init picture
            # Use floquet evolution
            # TODO: thinkagain at rules exception/etc.
            try:
                hfcInit = self.model.genHfc(self.cfc, picture='initial')
                state_t, coeffs = hfcInit.floquetEvol(self.init, self.d_rounding)
                proba_temp = self.model.fss.probaFromState(state_t)
                self.hfcInit = hfcInit
                self.fnscoeffs = coeffs
                self.testFNS(proba_temp, self.hfcInit.fm)
                success = True
            
            #except error.TruncFloquetlError as e:
            except error.BZNbError as e:
                new_ntrunc = int(new_ntrunc * 1.25)
                state_t = np.zeros([self.p.n_dt, self.fss.n_hilbert])
                
            except (error.TruncMotionalError, error.NotUnitaryError) as e:
                if(self._skipMotionalError):
                    print('Motional Error Skipped')
                    success = True
                else:
                    new_nmot = int(e.nb_mot * 1.35)

            
            except:
                print('Error not recognized: VERIF in LightSimul')
                new_nmot = int(temp_nmot * 1.35)
                new_ntrunc = int(new_ntrunc * 1.25) 
                
            finally:
                # reinitialize 
                self.p.n_trunc = initial_ntrunc
                self.p.n_fockmax = initial_nmot
                self.init = initial_init
                self.genModel(self.p, self.init)
                
        return success, state_t

    def runFOM(self, listRes = ['logFidelity'], typeRun = None, **args):
        """
        Purpose:
            evolve the initial state based on Floquet Numerical Simulator 
            (Exact simulation - up to decimal rounding accuracy) and compute a 
            figure of merit (FOM). If FNS failed return a pre-defined value for the FOM
        Output: 
            STORE hfcInit (from runFNS behavior)
            RETURN FOM 
               
        """
        # FNS Evolution 
        if(typeRun in [None, 'FNS']):
            self._ran, state_tmp = self.runFNS()
        elif(typeRun in ['HFE', 'FME', 'EFF', 'EFF']):
             _, state_tmp, _, _ = self.runHFESimul()
             self._ran = True
        else:
            raise NotImplementedError()
        
        self.state_store = state_tmp
        
        res=list([])

        if(self._ran):
            #To deal with cases where size of the Hilbert Space has beeen increased
            #when too much population in the last truncated motional state
            if (self.stateTGT.shape[1] != state_tmp.shape[1]):
                dim1 = self.stateTGT.shape[1]
                dim2 = state_tmp.shape[1]
                diff = int((dim2 - dim1)/2)
                st = np.concatenate((state_tmp[:,:int(dim1/2)], state_tmp[:,int(dim2/2):int(dim2-diff)]), axis=1)
                print(st.shape)
            else:
                st = state_tmp
            res = self.computeCustomRes(listRes, st, fail=False, **args)
                
        else:
            res = self.computeCustomRes(listRes, st=None, fail=True, **args)

        if(args.get('printRes', False)):
            print(res)
        return res


    
# ---------------------------
# AUXILLIARY FUNCTIONS
#   -genModel
#   -addTarget: Use of the HFE/MAGNUS approximation to simulate the system
#   -genTargetState: Use the FloquetNumerical Simulator to simulate the system
#   -computeFOM: run FNS and compute custom figure of merit 
# ---------------------------           
    def genModel(self, params, initState):
        """
        Purpose:
            gen the model and store params and state space associated
        """
        self.model = toy.ToyModel(params, initState)
        # update p and fss
        self.p = params
        self.fss = self.model.fss


    def addTarget(self, htgt, tgtPicture = None):
        """
        Purpose:
            store the target hamiltonian (htgt) and evolve the initial state 
            following htgt // the state returned is in the lab frame while htgt
            can be specified in another frame (tgtPicture) 
        Output:
            STORE htgt, state_TGT
        """
        self.htarget = htgt
        if (htgt is None):
            self.stateTGT = None
        else: 
            self.stateTGT = self.genTargetState(htgt, tgtPicture)
            

    def genTargetState(self, htgt, tgtPicture = None):
        """
        
            store the target hamiltonian (htgt) and evolve the initial state 
            following htgt // the state returned is in the lab frame while htgt
            can be specified in another frame (tgtPicture) 
        """
        st_tmp = htgt.evolution(self.p.array_time, self.init)
        if not((tgtPicture is None) or (tgtPicture == 'initial')):
            st_tmp = self.model.changeOfPicture(st_tmp, tgtPicture, 'initial')
        return st_tmp    
    

    def genCustomControl(self, controlType, controlTarget, store = True):
        """
        Purpose:
            Generate some custom control functions (they are generated in the 
            toymodel class)
        Output:
            either store it / or return it
        Comments:
            is it necessary to define a controlPicture??
        """
        cfc, htarget = self.model.genControlFunction(controlType, controlTarget)  


        if(store):
            self.cType = controlType
            self.cTarg = controlTarget
            if(self.cType == 'strong'):
                self.controlPicture = 'strong'
            else: 
                self.controlPicture = 'interaction'
            self.cfc = cfc 
            self.htarget = htarget
        else:
            return cfc, htarget
            
    def testFNS(self, proba_t, fm):
        """
        Purpose:
            Verify FloquetNumericalSimulations has worked:
            Test1: Verif deviation sum of probability over time cpared to 1
            Test2: population of the last fock state
            Test3: Weights of the Floquet modes  
            Raise relevant errors (should it be the case)
        """
        
        maxPop = self.p.maxPopLastMotional
        maxDev = self.p.maxDevSumProba
        decRnd = self.p.decimalRounding
        
        nmot = self.model.fss.n_mot
        ntrunc = self.model.fss.n_trunc
        
        sumProba = np.max(np.abs(np.sum(proba_t, 1) - 1))

        # Testing 1: Pop of the last Fock state is small enough
        popLast = np.max(np.sum(proba_t[:, self.fss.slice_mot[nmot-1]],1))           
        if(popLast > maxPop):
            print(str(popLast) + ' in last motional level')
            raise error.TruncMotionalError(nmot)

        # Testing 2: Truncature of the K matrix
        weight_agg = [np.log(np.average(np.abs(fm[sl])))/np.log(10) for sl in self.fss.slice_floquet_harmo]
        test_trunc = np.abs(np.max([np.max(weight_agg[0:5]), np.max(weight_agg[self.fss.n_blocks-6:self.fss.n_blocks-1])]))
        if(test_trunc < decRnd):           
            print(str(test_trunc)+' : decimal precision not good enough')
            # raise error.TruncFloquetlError(ntrunc)

        # Testing 3: Proba sum to 1 and 
        if (sumProba > maxDev):
            print('blemepro sumproba deviates: ' + str(sumProba))
            raise error.NotUnitaryError(nmot, ntrunc)
    
        self.weight_agg = weight_agg    
        return
    
# ---------------------------
# Computation of ad-hoc Fig Of Merit
# ---------------------------   
    def computeCustomRes(self, listRes, st, fail = False, **args):
        """
        Purpose:
            Compute custom (possibly complex) Figure of Merits 
            e.g. logFidelity_Heating0.1 = 
            based on computeSingleIndicator
        Arguments:
            listRes: list with the required FOM (e.f. ['heat', 'logF_heat0.1'])
                    fail: {T/F} {return the default value(from LIST_RESULTS_ERROR), 
                    compute it based on the state)
    
        Args (extra arguments)
            weights: weights used for computing the FOM (should have the same length as 
                    time dimension of the states)

        Output:
            return a list with the FOM requested (same len as the arg listRes)        
        FOM:
        
        """        

        res=list([])
        for el in listRes:
            components = ut.splitString(el)
            tmp = np.sum([self.computeSingleIndicator(c, st, fail = fail, **args) for c in components])
            res.append(tmp)
        return res
 
    def computeSingleIndicator(self, indic, st, fail = False, **args):
        """
        Purpose:
            compute single indicator
        Behavior implemented:
            First part
            * 'Fidelity' or 'logFidelity' or 'NegFidelity' or 'NeglogFidelity' 
              (need a target state to exist)
            * 'Heating': population not in ground motional 
            * 'Fluence': intensity (based on the control function) integral of 1/T * control(t)^2 
                   over one period
            
            second part (just after)
            + W (and weights not None) weighted average over time
            + m min value
            + M maxvalue (if none of W, m, M returns the average over time)
            + Nothing
            last part
            - if int returns (int * val) : allow to take into account of 
        """   
        #First part
        if('NeglogFidelity' in indic):
            assert hasattr(self, 'stateTGT'), 'To compute Fidelity need a stateTGT'
            if(fail):
                res_t = self.LIST_RESULTS_ERROR['NeglogFidelity']
            else:
                # between c. [0,1]
                res_t = np.log10(11 - 10 * self.fss.fidelity_distance_t(st, self.stateTGT)) 
            indic = ut.removeFromString(indic, 'NeglogFidelity')
 
        elif('NegFidelity' in indic):
            assert hasattr(self, 'stateTGT'), 'To compute Fidelity need a stateTGT'
            if(fail):
                res_t = self.LIST_RESULTS_ERROR['NegFidelity']
            else:            
                res_t = 1.000001 - self.fss.fidelity_distance_t(st, self.stateTGT)
            indic = ut.removeFromString(indic, 'NegFidelity')
        
        elif('logFidelity' in indic):
            assert hasattr(self, 'stateTGT'), 'To compute Fidelity need a stateTGT'
            if(fail):
                res_t = self.LIST_RESULTS_ERROR['logFidelity']
            else:
                # between c. [0,1]
                res_t = np.log10(self.fss.fidelity_distance_t(st, self.stateTGT) + 0.000001) 
            indic = ut.removeFromString(indic, 'logFidelity')
 
        elif('Fidelity' in indic):
            assert hasattr(self, 'stateTGT'), 'To compute Fidelity need a stateTGT'
            if(fail):
                res_t = self.LIST_RESULTS_ERROR['Fidelity']
            else:            
                res_t = self.fss.fidelity_distance_t(st, self.stateTGT)
            indic = ut.removeFromString(indic, 'Fidelity')
            
        elif('Heating' in indic):
            if(fail):
                res_t = self.LIST_RESULTS_ERROR['Heating']   
            else:
                res_t = self.fss.heat_t(st)
            indic = ut.removeFromString(indic, 'Heating')

        elif('Fluence' in indic):
            # Can be computed even if it has failed
            amplitudes = self.cfc[1]
            res_t = np.sum([np.square(np.abs(a)) for a in amplitudes])
            
            #Normalization of the Fluence
            scalingFluence = args.get('scaling_fluence', 1)
            res_t = res_t / scalingFluence
            
            indic = ut.removeFromString(indic, 'Fluence')
            
        else:
            print(indic)
            raise NotImplementedError()
        #pdb.set_trace()
        #SecondPart
        if len(indic) == 0:
            res = np.average(res_t)
            
        elif('W' == indic[0]):
            if(fail):
                res=res_t
            else:
                weights = args.get('weights_fom', 1)
                if(len(weights) == 1):
                    print ('weights used equal 1')
                res = np.sum(weights * res_t) / np.sum(weights)
            indic = ut.removeFromString(indic, 'W')
     
        elif('M' == indic[0]):
            if(fail):
                res=res_t
            else:
                res = np.max(res_t)
            indic = ut.removeFromString(indic, 'M')

        elif('m' == indic[0]):
            if(fail):
                res=res_t
            else:
                res = np.min(res_t)
            indic = ut.removeFromString(indic, 'm')
        else:
            res = np.average(res_t)

        #Lastpart (is there an int left)
        if(len(indic) > 0):
            res = float(indic) * res
        
        return res



        

        
        
# ---------------------------
# DEPRECIATED AT THE MOMENT
#   -runHFESimul: Use of the HFE/MAGNUS approximation to simulate the system
#   -runFNS: Use the FloquetNumerical Simulator to simulate the system
#   -computeFOM: run FNS and compute custom figure of merit 
# ---------------------------  
    def runFull(self, dbg = False):
        """
        Input: controlType (simple, strong, trap, etc.. cf. toymodel) 
        and params associated controlTarget
        
        DEFAULT RUN: just FNS
        
        """    
        if(dbg):
            pdb.set_trace()
        dicoState = {}
        
        # Gen and store cfc (control fourier components and htarget)
        # self.genControl(controlType, controlTarget)

        # FNS Evolution
        self._ran, state_tmp = self.runFNS()        
        assert self._ran, 'Floquet Simulation failed' 
        dicoState['FNS'] = state_tmp
        
        # Target dynamics
        dicoState['TGT_NF'] = self.htarget.evolution(self.p.array_time, self.init)
        dicoState['TGT'] = self.model.changeOfPicture(dicoState['TGT_NF'], self.controlPicture, 'initial')

        # Effective Dynamics
        dicoState['FME'], dicoState['FME_K'], dicoState['FME_NF'], dicoState['FME_K_NF'] = self.runHFESimul()

        
        self.stateT = dicoState
        self._ran = True
        self.fqe = self.hfcInit.fqe        
        
    def infos(self):
        print('LIST_TYPE:'); print('\n')
        print('FNS : Floquet Numerical Simulations'); print('\n')
        print('FME: Floquet Magnus Expansion //_NF: In the new frame // _K taking into account micromotion'); print('\n')
        print('TGT_NF', 'TGT'); print('\n')
        

        print('\n'); print('LIST_DATA:'); print('\n')
        
        print('RESULTS:')
        print('Heating: heatT (over time), heatTStrobo (only stroboscopic time)')
        
        print('EXTRA ANNOTATION:')
        print('_stro stroboscopic times i.e. t = nT')


    
    
def getPopulation(sim, population = 'g', channel = 'FNS'):
    """ 
    Patch // should be remove at one point and incorporated to the class
    get proba over time on a particular basis
    population: g: ground internal state // e: excited internal state 
    // h: heating // 'n' pop in the n motional state // '+': e + g /sqrt(2) 
    
    Channel :Which states to use (FNS FME TARGET)
    """
    ss = sim.fss
    if(population == 'g'):
        sl = ss.slice_int[0]
        proba = sim.probaT[channel]
    elif(population == 'e'):
        sl = ss.slice_int[1]
        proba = sim.probaT[channel]
    elif(population.isdigit()):
        sl = ss.slice_mot[int(population)]
        proba = sim.probaT[channel]
    elif(population == '+'):
        if(~hasattr(sim, 'popIntPlusMinus')):
            sim.genStatePlusMinus()
            sim.genResultsProba(computBasis = 1)
        proba = sim.probaTPlusMinus[channel]
        sl = ss.slice_int[0]
    elif(population == '-'):
        if(~hasattr(sim, 'popIntPlusMinus')):
            sim.genStatePlusMinus()
            sim.genResultsProba(computBasis = 1)
        proba = sim.probaTPlusMinus[channel]
        sl = ss.slice_int[1]
    elif(population == 'h'):
        proba = sim.heatT[channel][: ,np.newaxis]
        sl = slice(0,1,1)            
    else:
        assert True, 'not implemented'
    
    return np.sum(proba[:, sl], 1)
            

def plotLoadings(sims, figNb = 1):
    fig, ax = plt.subplots(figNb)
    ax.set_xlabel('BZ')
    ax.set_ylabel('log10 avg weights')
    ax.grid()

    if (isinstance(sims), list):
        for i in range(len(sims)):
           ax.plot(sims[i].weight_agg) 
    else:
        ax.set_title(sims.simulName)
        ax.plot(sims.weight_agg)


def plotPopDynamic(sims, tStart = None, tStop= None):
    pass


def plotPopEvol(sims, tStart = None, tStop= None):
    
    sim = sims

    if(tStart is None):
        tStart = 0
    if(tStop is None):
        tStop = sim.p.tmax

    # Time parameters 
    p = sim.p
    iStart = int(tStart/p.dt)
    iStop = int(tStop/p.dt)
    iSlice = slice(iStart, iStop)
    iSliceStrobo = slice(iStart, iStop, int(max(p.n_pointsPerPeriod,1)))
    time = p.array_time[iSlice]
    timeStrobo = p.array_time[iSliceStrobo]

    # Figure1 Evol of the spin down FNS vs FME
    fig, ax = plt.subplots(1)
    ax.plot(time, sim.popInt['FNS'][iSlice, 0], label='FNS', color='blue')
    ax.plot(time, sim.popInt['FME'][iSlice, 0], label='FME', color='red')
    ax.plot(time, sim.popInt['FME_K'][iSlice, 0],'--', label='FME with K', color='green')
    ax.scatter(timeStrobo, sim.popInt['FME'][iSliceStrobo, 0], edgecolors='r', facecolors='none')
    #ax.scatter(timeStrobo, sim.popInt['FME_K'][iSliceStrobo, 0], edgecolors='b', facecolors='none')
    ax.set_title(sim.simulName + 'FNS vs. FME')
    ax.legend(loc='best')
    ax.set_xlabel('time')
    ax.set_ylabel('spindown (pct pop.)')
    ax.set_ylim([-0.01,1.01])
    ax.grid()      

    # Figure2 Evol of the spin down
    fig, ax = plt.subplots(1)
    ax.plot(time, sim.popInt['FNS'][iSlice, 0], label='FNS', color='blue')
    ax.plot(time, sim.popInt['TGT'][iSlice, 0], label='Target', color='red')
    ax.scatter(timeStrobo, sim.popInt['FNS'][iSliceStrobo, 0], edgecolors='r', facecolors='none')
    #ax.scatter(timeStrobo, sim.popInt['FME_K'][iSliceStrobo, 0], edgecolors='b', facecolors='none')
    ax.set_title(sim.simulName + 'FNS vs. Target')
    ax.legend(loc='best')
    ax.set_xlabel('time')
    ax.set_ylabel('spindown (pct pop.)')
    ax.set_ylim([-0.01,1.01])
    ax.grid()      

    #Figure2: Heat
    ymax = 100*np.max(np.array([sim.heatT['FNS'][iSlice], sim.heatT['FME'][iSlice], sim.heatT['FME_K'][iSlice]]))
    fig, ax = plt.subplots(1)
    ax.plot(time, 100 * sim.heatT['FNS'][iSlice], label='FNS', color='blue')
    ax.plot(time, 100 * sim.heatT['FME'][iSlice], label='FME', color='red')
    ax.plot(time, 100 * sim.heatT['FME_K'][iSlice], '--', label='FME with K', color='green')
    ax.scatter(timeStrobo, 100 * sim.heatT['FME'][iSliceStrobo], edgecolors='r', facecolors='none')
    #ax.scatter(timeStrobo, 100 * sim.heatT['FME_K'][iSliceStrobo], edgecolors='b', facecolors='none')
    ax.set_title(sim.simulName+ 'FNS vs. FME')
    ax.legend(loc='best')
    ax.set_xlabel('time')
    ax.set_ylabel('heating (pct pop.)')
    ax.set_ylim([-0.001,ymax+0.001])
    ax.grid()      

    #Figure2: Heat
    ymax = 100*np.max(np.array([sim.heatT['FNS'][iSlice], sim.heatT['FME'][iSlice], sim.heatT['FME_K'][iSlice]]))
    fig, ax = plt.subplots(1)
    ax.plot(time, 100 * sim.heatT['FNS'][iSlice], label='FNS', color='blue')
    ax.plot(time, 100 * sim.heatT['TGT'][iSlice], label='Target', color='red')
    ax.scatter(timeStrobo, 100 * sim.heatT['FME'][iSliceStrobo], edgecolors='r', facecolors='none')
    #ax.scatter(timeStrobo, 100 * sim.heatT['FME_K'][iSliceStrobo], edgecolors='b', facecolors='none')
    ax.set_title(sim.simulName + 'FNS vs. Target')
    ax.legend(loc='best')
    ax.set_xlabel('time')
    ax.set_ylabel('heating (pct pop.)')
    ax.set_ylim([-0.001,ymax+0.001])
    ax.grid()  

        
       

    
        
if __name__ == '__main__':
    pass
