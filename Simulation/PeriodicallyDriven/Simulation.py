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

from ...ToyModels import ToyModel as toy

import numpy as np
import matplotlib.pyplot as plt
import pdb

#==============================================================================
#                   INIT
# 
#==============================================================================
#
#
#
#

class Simulation:
    """ Simulation
    METHODS:
        - init : Init
        - run :
        - getSomeResults :
        - getExtraResults : (should be spin off)
        -
        - compare 


    TODO:
        - in Strong picture
        - Add Change of Pictures
    """
    
    LIST_TYPE = list(['FNS', 'FME_NF', 'FME_K_NF', 'FME', 'FME_K', 'TGT_NF', 'TGT'])
    
    
    def __init__(self, params, initState, simulName='', iterMax = 5, sizeMax = 10000): 
        #        
        self.simulName = simulName
        self.p = params
        self.init = initState

        # 
        self._nIterMax = iterMax
        self._sizeKmax = sizeMax
        self.d_rounding = params.decimalRounding
        self._ran = False
        self._resgen = False

        #
        self.genModel(self.p, self.init)
        if (hasattr(self.p, 'gauge')):
            self.gauge = self.p.gauge
        else:
            self.gauge = None


    def run(self, controlType = 'simple', controlTarget = 1, dbg = False, htgt = None, targetPicture = None):
        """
        Input: controlType (simple, strong, trap, etc.. cf. toymodel) 
        and params associated controlTarget
        """    
        if(dbg):
            pdb.set_trace()
        dicoState = {}
        
        # Gen and store cfc (control fourier components and htarget)
        self.genControl(controlType, controlTarget)

        # FNS Evolution
        self._ran = False
        self._Iter = 0
        self._ran, state_tmp = self.runFNS()        
        assert self._ran, 'Floquet Simulation failed' 
        dicoState['FNS'] = state_tmp
        
        # Target dynamics
        if(htgt is None):
            dicoState['TGT_NF'] = self.htarget.evolution(self.p.array_time, self.init)
            dicoState['TGT'] = self.model.changeOfPicture(dicoState['TGT_NF'], self.controlPicture, 'initial')
        else:
            print('htarget forced')
            if (targetPicture is None):
                targetPicture = 'initial'    
            
            self.htarget = htgt
            dicoState['TGT_NF'] = self.htarget.evolution(self.p.array_time, self.init)
            dicoState['TGT'] = self.model.changeOfPicture(dicoState['TGT_NF'], targetPicture, 'initial')
            
            
        # Effective Dynamics
        dicoState['FME'], dicoState['FME_K'], dicoState['FME_NF'], dicoState['FME_K_NF'] = self.runEffectiveSimul()

        
        self.stateT = dicoState
        self._ran = True
        self.fqe = self.hfcInit.fqe
    
#    def addTarget(self, htgt, tgtPicture = None):    
#        self.htarget = htgt
#        if (htgt is None):
#            self.stateTGT = None
#        else: 
#            self.stateTGT = self.genTargetState(htgt, tgtPicture)
#            
#    
#    def genTargetState(self, htgt, tgtPicture = None):
#        st_tmp = htgt.evolution(self.p.array_time, self.init)
#        if not((tgtPicture is None) | (tgtPicture == 'initial')):
#            st_tmp = self.model.changeOfPicture(st_tmp, tgtPicture, 'initial')
#        return st_tmp    
    
    def getSomeResults(self, controlType = 'simple', controlTarget = 1):
        """ 
        If not already done run simul 
        Gen Results (a lot.. maybe too much.. add option verbose or not )
        """
        if(self._ran == 0): #TODO: Verif it was run with the same parameters
            self.run(controlType, controlTarget)
               
        self.genResultsProba()
        self.genResultsPopulation()
        self.genResultsHeating()
        self.genResultsFidelity()
        self.genResultsMiscellaneous()
        #self.genEnergy()
        self.genResultsFidelityProba()

        # Results in the {+,-} basis (instead of e/g)
        self.genStatePlusMinus()
        self.genResultsProba(computBasis = 1)
        self.genResultsPopulation(computBasis = 1)

        #
        self._resgen = True


    def genModel(self, params, initState):
        """
        gen the model and store params and state space associated for it
        """    
        self.model = toy.ToyModel(params, initState)
        # update p and fss
        self.p = params
        self.fss = self.model.fss
    

    def genControl(self, controlType, controlTarget):
        """
        Generate and save the control functions
        associated to the parameters (Fourrier Rep)
        """
        self.cType = controlType
        self.cTarg = controlTarget
        if(self.cType == 'strong'):
            self.controlPicture = 'strong'
        else: 
            self.controlPicture = 'interaction'
        self.cfc, self.htarget = self.model.genControlFunction(self.cType, self.cTarg)  


    def runEffectiveSimul(self):
        """
        Steps: 
            (1) Build the Periodic Hamiltonian in the New Frame (NF)
            (2-3) Compute HEff Keff (FME here) in NF and store them
            (2-3) Evolve the state in this NF according to the effective Ham
            (4) Get back to initial picture
        Output:
            Save
        """
        # generate effecive hamiltonian in the New Frame (NF)
        self.hEffNF = self.model.genHfc(self.cfc, self.controlPicture)
        #self.hNF = hams.PeriodicHamiltonian(self.p, self.hfcNF, self.p.omega, self.fss)

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
        Floquet Numerical Simulator (Exact simulation - up to decimal rounding accuracy)
        RETURN: success 
        STORE: hfcInit 
        """
        self._Iter = 0
        success = False
        new_nmot = self.p.n_fockmax
        new_ntrunc = self.p.n_trunc
        
        
        while(not(success) and self._Iter < self._nIterMax): 
            if ((new_nmot * (2 * new_ntrunc + 1) ) > self._sizeKmax):
                print('Diago of a matrix too big')
                break
            self._Iter+=1
            if(self._Iter>1):
                print('iter: ' + str(self._Iter) + ' Regen model with nmot: ' + str(new_nmot) +' ,nTrunc: ' + str(new_ntrunc) )
                diff_nmot = new_nmot - self.p.n_fockmax
                #self.fss.changeNMot(new_nmot)
                self.init = np.concatenate((self.init, np.zeros(int(diff_nmot* self.fss.n_int), dtype = 'complex128')))
                self.p.n_trunc = new_ntrunc
                self.p.n_fockmax = new_nmot
                self.genModel(self.p, self.init)
                # Don't need to regenerate control right??

            # Main: gen harmonics of the Hamiltonian in the init picture
            # Use floquet evolution
            hfcInit = self.model.genHfc(self.cfc, picture='initial')
            state_t, coeffs = hfcInit.floquetEvol(self.init, self.d_rounding)
            proba_temp = self.model.fss.probaFromState(state_t)
            self.hfcInit = hfcInit
            self.fnscoeffs = coeffs
 
            success, new_nmot, new_ntrunc = self.testFNS(proba_temp, self.hfcInit.fm)
            

        return success, state_t

            
    def testFNS(self, proba_t, fm):
        """
        Verify FloquetNumericalSimulations has worked:
            Test1: Verif deviation sum of probability over time cpared to 1
            Test2: population of the last fock state
            Test3: Weights of the Floquet modes  
        """
        
        maxPop = self.p.maxPopLastMotional
        maxDev = self.p.maxDevSumProba
        decRnd = self.p.decimalRounding
        
        is_ok = True
        new_nmot = self.model.fss.n_mot
        new_ntrunc = self.model.fss.n_trunc

        # Testing 1: Proba sum to 1 and 
        sumProba = np.max(np.abs(np.sum(proba_t, 1) - 1))
        if (sumProba > maxDev):
            print('blemepro sumproba deviates: ' + str(sumProba))
            new_nmot =  int(new_nmot * 1.35)
            is_ok = False

        # Testing 2: Pop of the last Fock state is small enough
        popLast = np.max(np.sum(proba_t[:, self.fss.slice_mot[self.fss.n_mot-1]],1))           
        if(popLast > maxPop):
            print(str(popLast) + ' in last motional level')
            new_nmot =  int(new_nmot * 1.35)    
            is_ok = False            
    
        # Testing 2: Truncature of the K matrix
        weight_agg = [np.log(np.average(np.abs(fm[sl])))/np.log(10) for sl in self.fss.slice_floquet_harmo]
        test_trunc = np.abs(np.max([np.max(weight_agg[0:5]), np.max(weight_agg[self.fss.n_blocks-6:self.fss.n_blocks-1])]))
        if(test_trunc < decRnd):           
            print(str(test_trunc)+' : decimal precision not good enough') 
            new_ntrunc = int(new_ntrunc * 1.25)      
            is_ok = False

        self.weight_agg = weight_agg    
            
        return is_ok, new_nmot, new_ntrunc


    def genNprint(self, controlType = 'simple', controlTarget = 1):
        """ If not already done run simul + gen Results
        Print Some infos about the simulation
        """
        self.getSomeResults(controlType, controlTarget)
        self.printSomeResults()
        

    def printSomeResults(self):
        """ If not already done run simul + gen Results
        Print Some infos about the simulation
        """
        #self.getSomeResults(controlType, controlTarget)
        print('#########   BEGIN RESULTS  #########')
        print(self.simulName + ': ' + str(self.cType))
        print('\n')
        print('some indicators (heating5first / heatingLast): ')
        print(self.pop5fock, self.popLast)
        
        print('Range QuasiEn: ')
        print(np.min(np.abs(self.fqe)), np.max(np.abs(self.fqe)))
        print('\n')

        # if((controlType == 'simple') or (controlType == 'noHeat')):
        print('### heating = max proba(%) not in ground ###')
        print('Simul / Target / Magnus /Magnus withK')
        h = self.heatMax
        print(pctNdec(h['FNS']), pctNdec(h['TGT']), pctNdec(h['FME']), pctNdec(h['FME_K']))
        print('STROBO')
        h = self.heatMaxStro
        print(pctNdec(h['FNS']), pctNdec(h['TGT']), pctNdec(h['FME']), pctNdec(h['FME_K']))        
        print('\n')

        print('### Fidelity to FNS (%): ### ')
        print('Magnus /Magnus withK/ MagnusNF /MagnusKNF')
        f = self.fid
        print(pctNdec(f['FME_FNS']), pctNdec(f['FME_K_FNS']), pctNdec(f['FME_NF_FNS']), pctNdec(f['FME_K_NF_FNS']))
        print('STROBO')
        f = self.fidStro
        print(pctNdec(f['FME_FNS']), pctNdec(f['FME_K_FNS']), pctNdec(f['FME_NF_FNS']), pctNdec(f['FME_K_NF_FNS']))


        print('### Fidelity to FNS (%) - WITH PROBA: ### ')
        print('Magnus /Magnus withK/ MagnusInit /Magnus withKInit')
        f = self.fidProba
        print(pctNdec(f['FME_FNS']), pctNdec(f['FME_K_FNS']), pctNdec(f['FME_NF_FNS']), pctNdec(f['FME_K_NF_FNS']))
        print('Strobo')
        f = self.fidProbaStro
        print(pctNdec(f['FME_FNS']), pctNdec(f['FME_K_FNS']), pctNdec(f['FME_NF_FNS']), pctNdec(f['FME_K_NF_FNS']))
        print('#########   END RESULTS  #########')
        print('\n')

     
    def genResultsMiscellaneous(self):
        """
        """
        pTmp = self.probaT['FNS']
        self.popLast = np.max(pTmp[: ,self.fss.n_mot-1] + pTmp[:,self.fss.n_hilbert-1])
        self.pop5fock = [np.max(pTmp[:, n]) for n in np.arange(5)]




    def genResultsFidelity(self):
        """Fidelity measures
        """
        slice_strobo = self.p.slice_time_strobo
        
        state = self.stateT
        dico1, dico2, dico3 = {}, {}, {}
        
        dico1['FME_TGT'], dico2['FME_TGT'], dico3['FME_TGT'] = self.genSevFidelities(state['FME'], state['TGT'], slice_strobo)
        dico1['FME_FNS'], dico2['FME_FNS'], dico3['FME_FNS'] = self.genSevFidelities(state['FME'], state['FNS'], slice_strobo)
        dico1['FME_K_FNS'], dico2['FME_K_FNS'], dico3['FME_K_FNS'] = self.genSevFidelities(state['FME_K'], state['FNS'], slice_strobo)
        dico1['FME_NF_FNS'], dico2['FME_NF_FNS'], dico3['FME_NF_FNS'] = self.genSevFidelities(state['FME_NF'], state['FNS'], slice_strobo)
        dico1['FME_K_NF_FNS'], dico2['FME_K_NF_FNS'], dico3['FME_K_NF_FNS'] = self.genSevFidelities(state['FME_K_NF'], state['FNS'], slice_strobo)        
        
        #Main measures
        self.quality = dico2['FME_K_FNS']; self.quality_strobo = dico3['FME_K_FNS']
        self.fidT = dico1; self.fid = dico2; self.fidStro = dico3 

    def genResultsFidelityProba(self):
        """Fidelity measures based on proba (rather than state) - i.e. don't depend on relative phase
        """
        slice_strobo = self.p.slice_time_strobo
        
        proba = self.probaT
        dico1, dico2, dico3 = {}, {}, {}
        
        dico1['FME_TGT'], dico2['FME_TGT'], dico3['FME_TGT'] = self.genSevFidProba(proba['FME'], proba['TGT'], slice_strobo)
        dico1['FME_FNS'], dico2['FME_FNS'], dico3['FME_FNS'] = self.genSevFidProba(proba['FME'], proba['FNS'], slice_strobo)
        dico1['FME_K_FNS'], dico2['FME_K_FNS'], dico3['FME_K_FNS'] = self.genSevFidProba(proba['FME_K'], proba['FNS'], slice_strobo)
        dico1['FME_NF_FNS'], dico2['FME_NF_FNS'], dico3['FME_NF_FNS'] = self.genSevFidProba(proba['FME_NF'], proba['FNS'], slice_strobo)
        dico1['FME_K_NF_FNS'], dico2['FME_K_NF_FNS'], dico3['FME_K_NF_FNS'] = self.genSevFidProba(proba['FME_K_NF'], proba['FNS'], slice_strobo)        

        #Store
        self.qualityP = dico2['FME_K_FNS']; self.qualityP_strobo = dico3['FME_K_FNS']
        self.fidProbaT = dico1; self.fidProba = dico2; self.fidProbaStro = dico3  

    def genResultsProba(self, computBasis = 0):
        """ 
        gen proba over time based on the state
        """
        dico = {}
        if(computBasis == 0):
            state = self.stateT
        elif(computBasis == 1):
            state = self.stateTPlusMinus
        else:
            assert True, 'genResultsProba, selected computBasis not implemented'
        
        for typ in self.LIST_TYPE:
            dico[typ] = self.fss.probaFromState(state[typ])

        if(computBasis == 0):
            self.probaT = dico
        elif(computBasis == 1):
            self.probaTPlusMinus = dico
            
            
    #TODO: Think again     
    
    def genStatePlusMinus(self):
        dico = {}
        state = self.stateT
        
        for typ in self.LIST_TYPE:
            dico[typ] = self.chgeComputBasis(state[typ])
        self.stateTPlusMinus = dico

    def chgeComputBasis(self, state_t, newbasis=1):
        """ newbasis = 1 from {g, e} to {+, -} with +/- = 1/sqrt(2) (g +/- e)
        """
        state_ncb = np.zeros_like(state_t)
        if(newbasis == 1):
            for m in np.arange(self.p.n_fockmax):
                indices = self.fss.slice_mot[m]
                toreplace = state_t[:, indices]
                r1 = 1/np.sqrt(2) * (toreplace[:,0] + toreplace[:,1])
                r2 = 1/np.sqrt(2) * (toreplace[:,0] - toreplace[:,1])
                replacement = np.c_[r1, r2]
                state_ncb[:, indices] = replacement 
        else:
            assert True, 'Simulation.chgeComputBasis change of Basis not yet implemented'
        return state_ncb

        
    def getPopulation(self, population = 'g', channel = 'FNS'):
        """ 
        get proba over time on a particular basis
        population: g: ground internal state // e: excited internal state 
        // h: heating // 'n' pop in the n motional state // '+': e + g /sqrt(2) 
        """
        ss = self.fss
        if(population == 'g'):
            sl = ss.slice_int[0]
            proba = self.probaT[channel]
        elif(population == 'e'):
            sl = ss.slice_int[0]
            proba = self.probaT[channel]
        elif(population.isdigit()):
            sl = ss.slice_mot[int(population)]
            proba = self.probaT[channel]
        elif(population == '+'):
            if(~hasattr(self, 'popIntPlusMinus')):
                self.genStatePlusMinus()
                self.genResultsProba(computBasis = 1)
            proba = self.probaTPlusMinus[channel]
            sl = ss.slice_int[0]
        elif(population == '+'):
            if(~hasattr(self, 'popIntPlusMinus')):
                self.genStatePlusMinus()
                self.genResultsProba(computBasis = 1)
            proba = self.probaTPlusMinus[channel]
            sl = ss.slice_int[0]
        elif(population == 'h'):
            proba = self.heatT[channel]
            sl = slice(0,1,1)            
        else:
            assert True, 'not implemented'
        
        return np.sum(proba[:, sl], 1)
        
        
        
        
        
    def genResultsPopulation(self, computBasis = 0):
        """ 
        gen proba over time based on the state
        """
        dicoInt = {}; dicoMot = {};
        sl_int = self.fss.slice_int
        sl_mot = self.fss.slice_mot
        if(computBasis == 0):
            proba = self.probaT
        elif(computBasis == 1):
            proba = self.probaTPlusMinus
        else:
            assert True, 'genResultsProba, selected computBasis not implemented'
        
        for typ in self.LIST_TYPE:
            probaTmp = proba[typ]
            dicoInt[typ] = np.transpose([np.sum(probaTmp[:,sl],1) for sl in sl_int])
            dicoMot[typ] = np.transpose([np.sum(probaTmp[:,sl],1) for sl in sl_mot])

        if(computBasis == 0):
            self.popInt = dicoInt
            self.popMot = dicoMot
        elif(computBasis == 1):
            self.popIntPlusMinus = dicoInt
            self.popMotPlusMinus = dicoMot
        
    def genResultsHeating(self):
        """ 
        Generate heating indicators
        Heating is defined as everything which is not on the ground motional state
        XXX_ht : heating over time
        """
        slice_mot0 = self.fss.slice_mot[0]
        slice_strobo = self.p.slice_time_strobo
        proba = self.probaT
        dico1 = {}; dico1str = {}; dico2 = {}; dico2str = {}
        
        for typ in self.LIST_TYPE:
            dico1[typ], dico1str[typ], dico2[typ], dico2str[typ] = self.genSevHeating(proba[typ], slice_strobo, slice_mot0)
        
        self.heatT = dico1; self.heatTStro = dico1str
        self.heatMax= dico2; self.heatMaxStro = dico2str
    
        
# --------------------
# Results: auxilliary
# --------------------
    def genSevFidelities(self, state1, state2, sliceStrobo):
        """
        """
        ft = self.fss.fidelity_distance_t(state1, state2)
        f = np.average(ft)
        f_strobo = np.average(ft[sliceStrobo])

        return ft, f, f_strobo     

    def genSevFidProba(self, p1, p2, sliceStrobo):
        """
        """
        fpt = 1 - self.fss.proba_distance_t(p1, p2)
        fp = np.average(fpt)
        fp_strobo = np.average(fpt[sliceStrobo])

        return fpt, fp, fp_strobo     

    def genSevHeating(self, proba_t, slice_strobo, indexMot0):
        """retrun hT: heat over time // hT_str: heat over strobotime // h maximum heat
        """
        hT = self.proba2heat(proba_t, indexMot0)
        hT_stro = self.proba2heat(proba_t[slice_strobo, :], indexMot0)
        h = self.proba2heatMax(proba_t, indexMot0)
        h_stro = self.proba2heatMax(proba_t, indexMot0)

        return hT, hT_stro, h, h_stro

    def proba2heat(self, prob_t, indexMot0):
        """ 
        Heating is defined as everything which is not on the ground motional state
        """
        heat_t = 1 - (np.sum(prob_t[: ,indexMot0], 1))
        return heat_t

    def proba2heatMax(self, prob_t, indexMot0):
        """ 
        max Heating over a the whole 
        """
        heatMax = np.max(self.proba2heat(prob_t, indexMot0))
        return heatMax

        
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
        
        
        
        
        
        
# ======================================== #
#    Misc
# ======================================== #
def pctNdec(val, dec = 4):
    return round(100 * val, dec) 

    
    
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
            
    

# ======================================== #
#    PLOTTING  // PLOTTING MULTIPLE
# ======================================== #

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






# ===================
#   TO FINISH
# ===================


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


# LIST_COLORS = []
# LIST_COLORS_STRO = []

# def compareFidelities(sims, listTeff = None, typeFid = 'normal'):
#     nbSim = len(sims)
#     amplNorm = np.array([s.amplNorm for s in sims])
#     omEff = np.array([s.omeff for s in sims])
#     dt = np.array([s.p.dt for s in sims])
#     index_Teff = int(2 * np.pi * omEff / dt)
#     index_strobo = np.array([s.p.n_pointsPerPeriod for s in sims])

#     if (listTeff is None):
#         if (typeFid == 'normal'):
#             f = np.array([s.fid['FNS'] for s in sims])
#             fStrobo = np.array([s.fidStro['FNS'] for s in sims])
#         elif (typeFid == 'normal'):
#             f = np.array([s.fidProba['FNS'] for s in sims])
#             fStrobo = np.array([s.fidProbaStro['FNS'] for s in sims])

#     else:
#         nbTeff = len(Teff)
#         f = np.zeros(nbSim, nbTeff)
#         sliceTmp = [[slice(0, t * index_Teff[s]) for s in np.arange(nbSim)] for t in np.arange(Teff)]
#         sliceTmp_Stro = [[slice(0, t * index_Teff[s], index_strobo[s]) for s in np.arange(nbSim)] for t in np.arange(Teff)]
             
        
        


    
        
if __name__ == '__main__':
    pass
