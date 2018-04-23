#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
from ..Misc import Errors as error
from .. import Helper as ut
from .StateSpace import FiniteHilbertSpace as hsp

import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as spa
import pdb

#==============================================================================
#                   Hamiltonian 
# PURPOSE:
#    Abstract Class to enforce some structure
#        
# MAIN (Abstract) METHODS:
#        - init: 
#        - Evolution: 
#
# MAIN Attributes
#        - 
#
#==============================================================================
class Hamiltonian:
    """
        Class: Generic Hamiltonian
    """

    def __init__(self, ss = None, hbar = 1): 
        self._existU = False
        self.ss = ss
        self.hbar = 1
        self.U = None
        self.Hmatrix = None


# --------------------------------------------------------------------------- #
#   Evolve a state 
# --------------------------------------------------------------------------- #
    def Evolution(self, time, initState, Uoptional = None):
        pass
    
# --------------------------------------------------------------------------- #
#   Set the evolution(operator) 
# --------------------------------------------------------------------------- #
    def genU(self):
        pass

    def setU(self, U, time_array):
        """
            Purpose:
                setup the evolution operator (Should be a 3D array with the 
                following dimensions: t x d x d)
                Should be provided with the corresponding time array
        """
        self.U = U # 
        self.time_U = time_array   
        self._existU = True


# --------------------------------------------------------------------------- #
#   Utility
# --------------------------------------------------------------------------- #
    def isHermitian(self, H = None):
        if(H is None):
            H = self.Hmatrix
        return hsp.isHermitian(H)

    def isUnitary(self, U = None):
        if (U is None):
            U = self.U
        return hsp.isUnitary(U)

    def dagger(self, H = None):
        if(H is None):
            H = self.Hmatrix
        return hsp.dagger(H)   





#==============================================================================
#                   Time Independent Hamiltonian
# PURPOSE:
#    Time Independent Hamiltonian
#        
# MAIN (Abstract) METHODS:
#        - init: 
#        - Evolution: 
#
# MAIN Attributes
#        - Hmatrix: representation 
        
#
#==============================================================================
class TIHamiltonian(Hamiltonian):
    """
        Time Independent Hamiltonians
    """
    
    def __init__(self, Hmatrix, ss = None, hbar = 1):
        super().__init__(ss, hbar = hbar) # Python 2: super(TIHamiltonian, self).__init__()
        self.Hmatrix = Hmatrix
        

    def Evolution(self, timeArray, initState, Hoptional = None):
        """
        Purpose:
            get the state over time according the static dynamics induced by Hmatrix
        Output 
            state over time as a t (number of time points) x d (HilbertSpace dim)
            array
                
        """    

        if (Hoptional is None):
            Hoptional = self.Hmatrix 
        hvals, hvects = ut.degenEV(Hoptional)
        coeffs = hsp.projection(initState, hvects) 
        coeffs_t = np.exp(-1.0j / self.hbar * hvals * timeArray[:, np.newaxis])
        state_t = np.dot((coeffs * coeffs_t), hvects.transpose())


        return state_t

    def Evolution1t(self, time, initState, Hoptional = None):
        """
            get the state for only one point in time 
            (just recasting the time scalar value in teh right format)
        """    
        timeArray = np.array([time])
        state_t = self.Evolution(timeArray, initState, Hoptional)
        if(np.abs(1-np.sum(np.square(np.abs(state_t))))>1e-2):
            pdb.set_trace()
        return state_t


    def getGroundState(self, H_opt):
        """
        Purpose:
            get the state associated to the smallest eigen value
        """
        if (H_opt is None):
            H_opt = self.Hmatrix 
        evals, evecs = ut.sortedEigenSystem(H_opt)
        
        return evals[0], evecs[0]

    def __call__(self, t):
        raise NotImplementedError()



#==============================================================================
#                   Time Dependent Hamiltonian
# PURPOSE:
#    Time Dependent Hamiltonian
#        
# MAIN (Abstract) METHODS:
#        - init: 
#        - Evolution: either by means of Piecewise constant Hamiltonian approxiamtion
#                      or Adiabatic
#
# MAIN Attributes
#        - HList: []
#
#==============================================================================
class TDHamiltonian(Hamiltonian):
    """
        Time dependent Hamiltonian encoded as H = [[fun1, operator1], ...] 
        H(t) = SUM fun1(t) * operator1
        another way to define it is by setting U directly using setU
    """ 

    def __init__(self, HList, ss = None, hbar = 1):
        Hamiltonian.__init__(self, ss, hbar)
        self.HList = HList

    
    def Evolution(self, init_state, time = 1.0, time_step=0.01, method = 'PWCH', intermediate = True):
        """
           Evolve an initial state (init_state) up to a time T
           if store is True save the intermediate state (i.e at each time step) 
        """
        # pdb.set_trace()
        index_time = np.arange(np.ceil(time/time_step) + 1)
        time_array = index_time * time_step
        time_array[-1] = time
        self.time_array = time_array

        if(method == 'PWCH'):
            state_t = self.EvolutionPWCH(init_state, time_array, intermediate)
            res = state_t

        elif(method == 'Adiabatic'):
            state_t, energies = self.EvolutionAdiabatic(init_state, time_array, intermediate)
            res = [state_t, energies]
        elif(method == 'U'):
            raise NotImplementedError()

        else:
            raise NotImplementedError()
        
        return res


    def EvolutionPWCH(self, init_state, time, intermediate = True):
        """
        Purpose:
            Evolve an initial state (init_state) either up to a time T (if time is a float)
            or for a list of times (if time is an array) 
        """ 
        if(isinstance(time, float)):
            time = [0, time]
        #Ht_iterable = (self.evalH(t) for t in time[:-1])
        
        if(intermediate): 
            nb_time = len(time)
            phi = np.zeros([nb_time, len(init_state)], dtype = 'complex128')       
            phi[0] = init_state
            for i in range(nb_time - 1):
                Ht = self.evalH(time[i])
                if (not(Ht.isHermitian())):
                    pdb.set_trace()
                    print('not Hermitian')
                time_step = time[i+1] - time[i]
                phi[i+1] = Ht.Evolution1t(time_step, phi[i])
        else:
            phi = init_state
            for i in range(nb_time - 1):
                Ht = self.evalH(time[i])
                time_step = time[i+1] - time[i]
                phi = Ht.Evoution1t(time_step, phi)
                
          
        return phi   
        
    def EvolutionAdiabatic(self, init_state, time, intermediate = True):
        """
        Purpose:
            Evolve an initial state (init_state) adiabatically.
            Doesn't work for degenerate eigenspace 
        """ 
        #pdb.set_trace()
        if(isinstance(time, float)):
            time = [0, time]
        H0 = self.evalH(time[0], matrix = True)
        evals, evecs = ut.sortedEigenSystem(H0) #just an ascending sorting here        
        coeffs0 = [self.ss.ip(ev, init_state) for ev in evecs]
        
        if(intermediate): 
            nb_time = len(time)
            phi = np.zeros([nb_time, len(init_state)], dtype = 'complex128')       
            energies = np.zeros([nb_time, len(init_state)], dtype = 'complex128')            
            phi[0] = init_state
            energies[0] = evals 
            for i in range(nb_time - 1):
                Ht = self.evalH(time[i+1], matrix = True) 
                evals, evecs = ut.sortedEigenSystem(Ht)
                phi[i+1] = np.dot(evecs, coeffs0)
                energies[i+1] = evals
        else:
            HT = self.evalH(time[-1], matrix = True)
            energies, evecs = ut.sortedEigenSystem(HT)
            phi = np.dot(evecs, coeffs0)

        return phi, energies



    def evalH(self, time, matrix = False):
        """
           H evaluated at time t based on self.HList
           return 
        """ 
        Ht = np.zeros_like(self.HList[0][1], dtype = 'complex128')
        for c in self.HList:
            Ht += c[0](time) * c[1] 
        
        
        if(not(matrix)):        
            Ht = TIHamiltonian(Ht)

        return Ht

    def __call__(self, time):
        return self.evalH(time, matrix = True)

#==============================================================================
#                   Time Periodic Hamiltonian
# PURPOSE:
#    Time Dependent Hamiltonian
#        
# MAIN (Abstract) METHODS:
#        - init: 
#        - Evolution: either by means of Piecewise constant Hamiltonian approxiamtion
#                      or Adiabatic
#
# MAIN Attributes
#        - HList: []
#
#==============================================================================
class PeriodicHamiltonian(Hamiltonian):
    """
    Periodic Hamiltonians:
    H = SUM_l exp^{1.0j l omega t} H_l
    Built from a list of the different harmonics/amplitude of the Hamiltonian 
    INPUT:class TIHamiltonian(Hamiltonian):

        HMatrixlist = list (list_l, list_Hl, list_Hlbigger)
        with list_Hlbigger Hl represented in a bigger space (only necessary when Hilbert space has been truncated)
        e.g. for one harmonics ()

    METHODS (main):
        - MagnusEvol
        - FloquetEvol

    METHODS (annexe)

    """
    def __init__(self, params, HFCList, fss = None):
        super().__init__(fss) # Python 2: super(TIHamiltonian, self).__init__()
        self.H = HFCList
        self.index_hfc = HFCList[0]    
        self.val_hfc = HFCList[1]
        if(len(HFCList) == 3):
            self.val_hfc_bigger = HFCList[2]
        else:
            self.val_hfc_bigger = HFCList[1]
        self.om = params.omega
        self.p = params
        self.fss = fss

        if hasattr(params, 'decimalRounding'):
            self.dec_rnd = params.decimalRounding
        else:
            print('init Periodic Hamiltonian: missing decimal rounding')
            self.dec_rnd = 10

    def Evolution(self, timeArray, initState):
        """
        Can only compute evolution using FloquetEvolution or Magnus Evolution
        """
        raise NotImplementedError

    #@profile
    def EvolutionMagnus(self, initState, gauge = None, useK = True):
        """
        Gen Truncated Heff, Keff then evolve the init state
        Output:
        self.Heff, self.Keff are updated
        return1: state (evolution under the effective Hamiltonian only)
        return2: state_with_K (taking into account the micromotion too)
        """
        # Effective Hamiltonians: build them / store them
        params = self.p
        self.Heff, self.Keff, populatedK = self.genHEff(gauge=gauge)

        # Add impact of Kick operator i.e. e^{iK(t=0)}
        if (useK & populatedK):
            #UF = np.zeros([ss.n_hilbert, ss.n_hilbert, params.n_pointsPerPeriod], dtype='complex128')
            if(gauge in ['VV', 'A', 'HFE']):
                UFtmp = ut.expOp(self.Keff[:,:,0], 1.0j)
                initState = np.dot(UFtmp, initState)


        # Get the evolution under Heff First
        state_t_target = self.Heff.evolution(params.array_time, initState)
        state_t_target_withK = np.copy(state_t_target)
        
        # Add impact of Micromotion
        if (useK & populatedK):
            #UF = np.zeros([ss.n_hilbert, ss.n_hilbert, params.n_pointsPerPeriod], dtype='complex128')
            for i in range(params.n_pointsPerPeriod):
                slicetmp = slice(i, params.n_dt, params.n_pointsPerPeriod)
                UFtmp = ut.expOp(self.Keff[:,:,i], -1.0j)
                state_t_target_withK[slicetmp,:] = np.dot(state_t_target[slicetmp,:], np.transpose(UFtmp))
                #np.allclose(np.dot(np.transpose(np.conj(UFtmp)), UFtmp), np.eye(30))
        return state_t_target, state_t_target_withK

    #@profile
    def EvolutionFloquet(self, initState, dec_rnd = 10):
        """
        Gen Truncated Heff, Keff then evolve the init state
        Output:
        self.fqe, self.fm are updated
        return1: state (evolution under the effective Hamiltonian only)
        return2: coeffs
        """
        n_h = self.fss.n_hilbert
        n_b = self.fss.n_blocks

        dec_rnd = self.dec_rnd
        fss = self.fss
        omega =self.om
        n_trunc = fss.n_trunc
        n_hilbert = fss.n_hilbert
        t = self.p.array_time

        # get Floquet quasi-energies and modes
        _, fqe, fm = self.genFloquetSystem(n_trunc, rounding = dec_rnd)
        self.fqe = fqe
        self.fm = fm

        # Get coeffs at time 0
        new_shape = [n_h, n_h * n_b]
        fm_rs = np.reshape(fm, new_shape, 'F')        
        tmp = np.dot(initState, fm_rs.conj()) #### DONT FORGET TO CONJUGATE
        coeffs = [np.sum(tmp[i*n_b:(i+1)*n_b]) for i in range(n_h)]
        
        # Now compute evol
        blocks_rs = np.tile(np.arange(-n_trunc, n_trunc+1) * omega, n_hilbert)[:, np.newaxis]
        qen_rs = np.repeat(fqe, fss.n_blocks)[:, np.newaxis]
        coeffs_rs = np.repeat(coeffs, fss.n_blocks)[:, np.newaxis]
            
        coeffs_t = (blocks_rs - qen_rs) * t
        coeffs_t = np.exp(1j * coeffs_t)
        coeffs_t= coeffs_rs * coeffs_t
        array_evol = np.dot(fm_rs, coeffs_t).transpose()
                
        return array_evol, coeffs 


    #@profile
    def genFloqK(self, n_trunc):
        """Generate K the Floquet quasienergy operator = 
        Input:
        - Hamiltonian Fourier Coefficients 
                hfc = [[harmonics number list], [Fourier components (opearors)]]
        - Params (n_trunc, )

        Output:
        - K (dim = [n_hilbert * n_blocks, n_hilbert * n_blocks])
        """
        
        # params         
        index_hfc = self.index_hfc
        val_hfc = self.val_hfc
        omega = self.om
        n_hfc = len(index_hfc)
        d_hilbert = val_hfc[0].shape
        n_hilbert = d_hilbert[0]
        n_blocks = 2 * n_trunc + 1
        
        Kresult = np.zeros([n_hilbert * n_blocks, n_hilbert * n_blocks], dtype='complex128')
        
        # Fill the  matrix
        for i in range(n_hfc):
            f = index_hfc[i]
            v = val_hfc[i]
            if (f == 0):  
                for j in range(-n_trunc, n_trunc + 1):
                    slice_tmp = slice((n_trunc + j) * n_hilbert, (n_trunc + j + 1) * n_hilbert,1)
                    Kresult[slice_tmp, slice_tmp] = v + j * omega * np.eye(n_hilbert)
            else:
                for j in range(max(f, 0), min(n_blocks-1, n_blocks + f - 1)):
                    row_slice_tmp = slice(j * n_hilbert,(j + 1) * n_hilbert,1)
                    col_slice_tmp = slice((j-f) * n_hilbert,(j - f + 1) * n_hilbert,1)                
                    Kresult[row_slice_tmp, col_slice_tmp] = v
        
        if(not(ut.isHermitian(Kresult))):
            raise error.NotHError('KResult in genFloqK')
                    
        return Kresult


    def genHEff(self, gauge = None):
        """
        Assumption: hfc is sorted
        
        INPUT:

            - gauge = None --> U(t) = e^{iK(t)} e^{-i H_F t}
                Heff = H_0 + sum_{l>0} \( [H_0, H_l - H_{-l}] + [H_{-l}, H_{l}] \) / (l omega) + O(omega^{-2})
                Keff(t) = 0 + sum_{l != 0} H_l (e^{i l omega t} - 1) / (i l omega)

            - gauge = 'A' --> ToImplement (cf. AlbertVerdeny) 
                Heff = H_0 + sum_{l>0} [H_{-l}, H_{l}] 
                Keff = 

            - gauge = 'HFE' --> ToImplement (cf. N Goldman/Bukov)
            
            - gauge = 'VV' --> VanVleck Perturbation Th. (cf. Eckardt 2015)
        
        OUTPUT:
            - Truncated version of Heff (resH a TIHamiltonian)  and Keff (resK a 3D array as it is not time independent dxdxt)

        TODO: 
            - Go Up to order 2 (H and K)
            - Implement other gauges ('A', HFE) (not sure A is interesting)
        """

        # Init      
        index_hfc = self.index_hfc
        val_hfc = self.val_hfc
        val_hfc_bigger = self.val_hfc_bigger
        ss = self.fss
        om = self.om
        params = self.p
        
        resH = np.zeros(ss.d_operator, dtype='complex128')
        #identity = np.eye(ss.n_hilbert, dtype='complex128')
        #resK = np.tile(identity[:,:,np.newaxis], (1,1, params.n_pointsPerPeriod))
        resK = np.zeros([ss.n_hilbert, ss.n_hilbert, params.n_pointsPerPeriod], dtype='complex128')
        hfc0 = np.zeros(ss.d_operator, dtype='complex128')
        hfc0_bigger = np.zeros(ss.d_operator_bigger, dtype='complex128')

        n_harmo = np.size(index_hfc)
        idx_harmo0 = index_hfc.searchsorted(0)
        val_harmoPos = index_hfc[index_hfc > 0]
        val_harmoNonNull = index_hfc[index_hfc != 0]

        orderK = params.orderK
        orderH = params.orderH
        populatedK = False

        if(gauge in [None, 'FME']):
        # Building the Magnus approx of the Floquet Hamiltonian     
            if ((orderH >=0) & (idx_harmo0 < n_harmo)):
                resH += val_hfc[idx_harmo0]
                hfc0 += val_hfc[idx_harmo0]
                hfc0_bigger += val_hfc_bigger[idx_harmo0]
                        
            if (orderH >=1):
                for harmo in val_harmoPos:
                    idx_pos = index_hfc.searchsorted(harmo)
                    idx_neg = index_hfc.searchsorted(-harmo)         
                    if((idx_pos < n_harmo) & (idx_neg < n_harmo)):
                        tmp = 1/(harmo * om) * ss.commut(val_hfc_bigger[idx_pos], val_hfc_bigger[idx_neg])
                        resH += ss.trunc_m(tmp, ss.n_mot)
                        if(idx_harmo0 < n_harmo): # i.e. H0 is not null
                            tmp = 1/(harmo * om) * ss.commut(hfc0_bigger, val_hfc_bigger[idx_pos] - val_hfc_bigger[idx_neg])
                            resH += ss.trunc_m(tmp, ss.n_mot)
            #if (params.orderH>=2):
                # TODO: Complete

        # Micromotion operator Vectorization: use of 3D arrays d x d x t
        # resK[:, :, t1] = K(t1) represented in H i.e. dim = d x d
            if (orderK >=1):
                time_oneT = params.array_time[np.arange(params.n_pointsPerPeriod)]
                resK = np.zeros([ss.n_hilbert, ss.n_hilbert, params.n_pointsPerPeriod], dtype='complex128')
                for harmo in val_harmoNonNull:
                    idx_pos = index_hfc.searchsorted(harmo)
                    tmp1D = -1.0j/(harmo * om) * (np.exp(1.0j * harmo * om * time_oneT)-1)
                    resK += val_hfc[idx_pos][:,:,np.newaxis] * tmp1D[np.newaxis, np.newaxis,:]
                populatedK = True    

        elif(gauge == 'A'):
            print('Gauge A not yet implemented')

        elif(gauge == 'HFE'):
            print('Gauge HFE: not yet implemented, implement it!')
            
        elif(gauge == 'VV'):
            if ((orderH >=0) & (idx_harmo0 < n_harmo)):
                resH += val_hfc[idx_harmo0]
                hfc0 += val_hfc[idx_harmo0]
                hfc0_bigger += val_hfc_bigger[idx_harmo0]
            if (orderH >=1):
                for harmo in val_harmoPos:
                    idx_pos = index_hfc.searchsorted(harmo)
                    idx_neg = index_hfc.searchsorted(-harmo)         
                    if((idx_pos < n_harmo) & (idx_neg < n_harmo)):
                        tmp = 1/(harmo * om) * ss.commut(val_hfc_bigger[idx_pos], val_hfc_bigger[idx_neg])
                        resH += ss.trunc_m(tmp, ss.n_mot)
            if (orderH >=2):
                for harmo1 in val_harmoNonNull:
                    idx_pos = index_hfc.searchsorted(harmo1)
                    idx_neg = index_hfc.searchsorted(-harmo1)         
                    if((idx_pos < n_harmo) & (idx_neg < n_harmo)):
                        #[H_{-l}, [H0, Hl]]
                        tmp = 1/(2*np.square(harmo1 * om)) * ss.commut(val_hfc_bigger[idx_neg], ss.commut(hfc0_bigger, val_hfc_bigger[idx_pos]))
                        resH += ss.trunc_m(tmp, ss.n_mot)
                        for harmo2 in val_harmoNonNull:
                            harmo3 = harmo2 - harmo1
                            if(harmo3 in val_harmoNonNull):
                                index1 = idx_pos
                                index2 = index_hfc.searchsorted(-harmo2)
                                index3 = index_hfc.searchsorted(harmo3)
                                tmp = 1/(3 * harmo1 * harmo2 * np.square(om)) * ss.commut(val_hfc_bigger[index2], ss.commut(val_hfc_bigger[index3], val_hfc_bigger[index1]))
                                resH += ss.trunc_m(tmp, ss.n_mot)
            if (orderK >=1):
                time_oneT = params.array_time[np.arange(params.n_pointsPerPeriod)]
                resK = np.zeros([ss.n_hilbert, ss.n_hilbert, params.n_pointsPerPeriod], dtype='complex128')
                for harmo in val_harmoNonNull:
                    idx_pos = index_hfc.searchsorted(harmo)
                    #tmp1D = -1.0j/(harmo * om) * (np.exp(1.0j * harmo * om * time_oneT)-1)
                    tmp1D = -1.0j/(harmo * om) * (np.exp(1.0j * harmo * om * time_oneT))
                    resK += val_hfc[idx_pos][:,:,np.newaxis] * tmp1D[np.newaxis, np.newaxis,:]
                populatedK = True
            
            if (orderK >=2):
                pass
                                
        return TIHamiltonian(params, resH, ss), resK, populatedK

    
    def genFloquetSystem(self, n_trunc, rounding = 10):
        """Generate K the Floquet quasienergy operator = 
            Input:
                - Hamiltonian Fourier Coefficients 
                    hfc = [[harmonics number list], [Fourier components (opearors)]]
                - Params
                - Floquet State Space

            Output:
                - fqe (Floquet quasi energies) fm (Floquet modes)
        """

        fss = self.fss
        #hfc = self.H
        omega = self.om

        # Gen floquet quasi energy operator K
        floquetK = self.genFloqK(n_trunc)

        #Diago it and get floquet modes and Floquet quasi energies
        compressedK = spa.csc_matrix(floquetK)
        number_of_eigs = 2 * fss.n_hilbert # Just need to take enough
        evals, evecs = la.eigs(compressedK, k = number_of_eigs, sigma=0.0)
        if (np.sum(np.abs(np.imag(evals)))>10e-9):
            print('imaginary part of quasi energies neglected --> LOOK AT IT')
        evals = evals.real.astype(np.float64, copy=False)
        fqe, fm = filterBZ(evals, evecs, omega, 0) #FBZ
        
        # Check degeneracy of the quasienergies and take action if necessary
        deg_indices, is_uniquepb = ut.find_duplicates(fqe, rounding)
        if is_uniquepb:
            print('GenFloquetSystem: degenerates quasi energies')
            for index in deg_indices:
                if(len(index)>1):
                    to_orthogonalize = fm[:, index]
                    orthogonalized = ut.gram_schmidt(to_orthogonalize)
                    fm[:,index] = orthogonalized

        if(fqe.size != fss.n_hilbert):
            print('BNZError')
            raise error.BZNbError(fqe.size, fss.n_hilbert)
        #assert (fqe.size == fss.n_hilbert), 'blemepro nb floquet modes found is: ' + str(fqe.size)

        return floquetK, fqe, fm
        

#  ========================================================================   #
#                           AUXILLARY FUNCTIONS
#  ========================================================================   #

def filterBZ(evals, evecs, omega, zone = 0):
    """Return sorted evecs, evals belonging to one BZ
    i.e. abs(quasiE) < omega /2
    """
    if(zone == 0):
        bz_vals, bz_vecs = ut.trimEigenSystem(evals, evecs, -omega/2, +omega/2)
    else:
        # Maybe not the exact def
        bz_vals, bz_vecs = ut.trimEigenSystem(evals, evecs, (zone * omega) -omega/2, (zone * omega) + omega/2)

    return bz_vals, bz_vecs
 
 

