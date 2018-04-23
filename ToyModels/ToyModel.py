#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""

from ..Utility.Quantum import StateSpace as ss
from ..Utility.Quantum import Hamiltonian as ham

import scipy.optimize as optim
import numpy as np
import scipy.integrate as integrate


#==============================================================================
#                   ToyModel
#    PURPOSE:
#        Toy model: Spin in an harmonic Oscillator described in different pictures
#        H(t) = Omega_trap n + Omega_int/2 sigmaZ + f(t) sigmaX (1 + eta (a^T + a))
#        Different pictures are implemented Labframe ('initial') / Interaction 
#        ('interaction') / StrongDriving ('strong')

#    MAIN METHODS:
#        - init : 
#	     - genHfc: generate Harmonics of the Hamiltonian (based on a control function)
#                  list([l], [H_l]) -> f(t) = sum_l H_l * e^{i l omega t} 
#        - changeOfPicture : 
#        - getSomeResults :
#        - getExtraResults : (should be spin off)
#        -
#        - compare 
#   MAIN VARIABLES
#        - state_t: np array with dim (t (number of time steps) x d (Hilbert dim))
#
#==============================================================================
class ToyModel:
    def __init__(self, params, initState):
        """  
        Purpose:
            Store params of the toymodel, generate a StateSpace associated
        """
        self.p = params
        self.n0 = params.n_int
        self.nTr = params.n_trap
        self.n1 = self.n0 - self.nTr
        self.n2 = self.n0 + self.nTr
        self.eta = params.eta
        self.om = params.omega
        self.period = params.T
        self.tArray = params.array_time
        self.fss = ss.FloquetStateSpace(params.n_fockmax, params.n_intState, self.p)
        self.init = initState

    def changeOfPicture(self, state_t, initPicture = 'initial', finalPicture = 'initial'):
        """
        Purpose:
            Take a state_t And change it from frameInit to frameFinal
        """
        if(initPicture == finalPicture):
            state_out = state_t
        elif((initPicture == "initial") & (finalPicture == "interaction")):
            state_out = self.InitToInterPicture(state_t, self.tArray)
        elif((initPicture == "interaction") & (finalPicture == "initial")):
            state_out = self.InterToInitPicture(state_t, self.tArray)
        elif((initPicture == "initial") & (finalPicture == "strong")):
            state_out = self.InitToStrongPicture(state_t, self.tArray, self.alphaCoeffs)
        elif((initPicture == "strong") & (finalPicture == "initial")):
            state_out = self.StrongToInitPicture(state_t, self.tArray, self.alphaCoeffs)
        else:
            assert True, 'changeOfPicture - invalid: ' + str(initPicture) + ' to ' + str(finalPicture)

        return state_out
    

# ---------------------------
# Custom functions
# genControlFunctions
# encode
#----------------------------      
    def genControlFunction(self, typeC, target): 
        """"
            call getControlFunction // save some results // return the rest
        """
        omeff, amplNorm, cfc, htarget = self.getControlFunction(typeC, target)
        self.omeff = omeff
        self.amplNorm = amplNorm
        return cfc, htarget
    
    def getControlFunction(self, typeC, target): 
        """		
         Different control schemes
         INTERACTION PICTURE: H_{target} = target[0] * sigmaX
        	* 'simple': Resonant (with internal transition) MonoChromatic 
    		* 'noHeat': Resonant (with internal transition) + correction Bichromatic
    		* 'noHeat2': Not done yet 
    		* 'noHeat3': Not done yet
    
          TRAP: Force an harmonic at n_tr and try to remove heating
    		* 'trap': Resonant + H_tr = tgt[1] Bichromartic
    		* 'trapDummy': Resonant + H_{tr+X} = tgt2
    		* 'trapDummyCorrect': Resonant + H_{tr+X} = tgt2 + extra harmonics to correct heating
    		* 'trapCorrect1': Resonant + H_{tr} = tgt2 + extra harmonics to correct heating
    		* 'trapCorrect2': Resonant + H_{tr} = tgt2 + 2 extra harmonics to correct heating
          STRONG DRIVING PICTURE
    
          Output: return cfc (Fourrier representation of the control function)
    				, htarget (TIHAmiltonian)
            TODO: Verif strong / New Strong
            TODO: FINISH simpleRed, simpleBlue
        """
        om = self.om
        n0 = self.n0
        nTr = self.nTr
        n1 = self.n1
        n2 = self.n2
        fss = self.fss
        eta = self.eta
        
        omeff = 0
        amplNorm = 0
        
        if(typeC == 'simple'):
            f0 = np.conj(target)
            index_tmp = np.array([-n0,n0])
            val_tmp = np.array([ np.conj(f0), f0])
            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose() 
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))
        
        elif(typeC == 'simpleRed'):
            f0 = np.conj(target)
            index_tmp = np.array([-n1,n1])
            val_tmp = np.array([ np.conj(f0), f0])
            htarget = f0 * fss.anihil_sigmaP
            htarget += htarget.conj().transpose() 
            omeff = eta * np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))
            
        elif(typeC == 'simpleBlue'):
            f0 = np.conj(target)
            index_tmp = np.array([-n1,n1])
            val_tmp = np.array([ np.conj(f0), f0])
            htarget = f0 * fss.anihil_sigmaP
            htarget += htarget.conj().transpose() 
            omeff = eta * np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))

        elif(typeC == 'noHeat'):
            f0 = np.conj(target)
            n_a = n0 + 2 * nTr + 1 #Arbritrarly chosen
            fa = - (gFun(n1, n2, n0) * f0)/gFun(n1, n2, n_a)
            index_tmp = np.array([-n_a, -n0, n0, n_a])
            val_tmp = np.array([np.conj(fa), np.conj(f0), f0, fa])
            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose()
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))
        
        elif(typeC == 'trap'):
            # 2 harmonics one at n_tr (constraint) one at n_int (to engineer sigma_x)
            f0 = np.conj(target[0])
            ftr = target[1]
            index_tmp = np.array([-n0, -nTr, nTr, n0])
            val_tmp = np.array([np.conj(f0), np.conj(ftr), ftr, f0])
            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose() 
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))
        
        elif(typeC == 'trapDummy'):
            # 2 harmonics one at n_tr (constraint) one at n_int (to engineer sigma_x)
            f0 = np.conj(target[0])
            ftr = target[1]
            index_tmp = np.array([-n0, -(nTr+2), (nTr+2), n0])
            val_tmp = np.array([np.conj(f0), np.conj(ftr), ftr, f0])
            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose()
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))

        elif(typeC == 'trapDummyCorrect'):
            # 2 harmonics one at n_tr (constraint) one at n_int (to engineer sigma_x)
            # TODO: verif it can be np arrays then make sure that arrays are used everywhere
            f0 = np.conj(target[0])
            ftr = target[1]
            n_a = n0 + 2 * nTr + 1 #Arbritrarly chosen
            fa = - (gFun(n1, n2, n0) * f0 + gFun(n1, n2, nTr+2) * ftr) /gFun(n1, n2, n_a)
            index_tmp = np.array([-n_a,-n0, -(nTr+2),  (nTr+2), n0, n_a])
            val_tmp = np.array([np.conj(fa), np.conj(f0), np.conj(ftr), ftr, f0, fa])

            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose() 
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))

        elif(typeC == 'trapCorrect1'):
            # 3 harmonics one at n_tr (constraint) one at n_int (to engineer sigma_x) one to correct heating a 
            n_a = n0 + 2 * nTr + 1 #Arbritrarly chosen
            f0 = np.conj(target[0])
            ftr = target[1]
            fa = - (gFun(n1, n2, n0) * f0 + gFun(n1, n2, nTr) * ftr) /gFun(n1, n2, n_a)
            index_tmp = np.array([-n_a, -n0, -nTr, nTr, n0, n_a])
            val_tmp = np.array([np.conj(fa), np.conj(f0), np.conj(ftr), ftr, f0, fa])

            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose() 
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))

        elif(typeC == 'trapCorrect2'):
            #  4 harmonics 
            n_a = n0 + 2 * nTr + 1 #Arbritrarly chosen
            n_aa = n0 + 2 * nTr 
            f0 = np.conj(target[0])
            ftr = target[1]
            faa = np.square(np.abs(ftr))/(n0 * gFun2(n0, n2) * f0)
            fa = -1/(gFun(n1, n2, n_a)) * (gFun(n1, n2, n0) * f0 + gFun(n1, n2, nTr) * ftr + gFun(n1, n2, n_aa) * faa)
            index_tmp = np.array([-n_a, -n_aa,-n0, -nTr, nTr, n0, n_aa, n_a])
            val_tmp = np.array([np.conj(fa), np.conj(faa), np.conj(f0), np.conj(ftr), ftr, f0, faa, fa])

            htarget = f0 * fss.sigmaP
            htarget += htarget.conj().transpose() 
            omeff = np.abs(f0)
            amplNorm = np.abs(omeff / (n0 * om))

        elif(typeC == 'displ'):
            harmo1 = n2 -1 #arbitrarly chosen
            harmo2 = harmo1 + 2 * nTr
            fExtra = om * np.sqrt(target*(harmo1 + n2)*(harmo1 - n1)/(np.square(eta)*(n1 + n2)))
            index_tmp = np.array([-harmo2, -harmo1, harmo1, harmo2])
            val_tmp = np.array([np.conj(fExtra), np.conj(fExtra), fExtra, fExtra])
            htarget = np.conj(target) * fss.anihil
            #amplNorm = 

        elif(typeC == 'squeeze'):
            harmo1 = n2 -1 #arbitrarly chosen
            harmo2 = harmo1 + nTr
            fExtra = om * np.sqrt(target/(2 * eta * n0 * (1/(np.square(harmo1) - np.square(n0)) + 1/((harmo1 + n2)*(harmo1 - n1)))))
            index_tmp = np.array([-harmo2, -harmo1, harmo1, harmo2])
            val_tmp = np.array([np.conj(fExtra), np.conj(fExtra), fExtra, fExtra])
            htarget = np.conj(target) * np.dot(fss.anihil, fss.anihil)
            htarget += htarget.conj().transpose() 
            #amplNorm = 

        elif(typeC == 'strong'):
            #Special case: target is a list of coefficients of alpha(t) = d{f(t)}  / dt
            #resultsOptim = np.array([-0.8673243, -1.14802598,  0.8673243 , -0.55356358])
            self.alphaCoeffs = target
            self.alphaFun = lambda t: (fourrierInverse(target, om, t))
            index_tmp, val_tmp = alphaCoeffsToCfc(target, om)
            theta1, theta2, theta3, theta4, theta5, theta6, theta7 = genCoeffsHStrong0(self.alphaFun, self.period, self.om, self.n0, self.nTr, self.eta)
            htarget = theta6 * fss.sigmaY + theta4 * fss.sigmaZ
            omeff = np.max(np.abs([theta6, theta4]))
            amplNorm = np.abs(omeff / (n0 * om))
            #htarget += theta1 * fss.number + theta3 * fss.identity + theta4 * fss.sigmaZ + theta6 * fss.sigmaY
            #htarget += theta2 * np.dot((fss.creat - fss.anihil), fss.sigmaX) + np.dot((fss.creat + fss.anihil), theta5* fss.sigmaZ + theta7* fss.sigmaY)

        else:
            assert (0), 'bad choice of control type'

        index_sort = np.argsort(index_tmp)
        htarget = ham.TIHamiltonian(self.p, htarget)
        cfc = [index_tmp[index_sort], val_tmp[index_sort]]
        #Add primitive
        #cFc = 
        return omeff, amplNorm, cfc, htarget
    
    def genHfc(self, cfc, picture = 'interaction'):
        """
            Purpose: generate the Hamiltonian in a particular frame (picture)
                    and for a specific control function  
        """
        index_cfc = cfc[0]
        val_cfc = cfc[1]
        n_cfc = np.size(index_cfc)
        om = self.om
        n0 = self.n0
        nTr = self.nTr
        n1 = self.n1
        n2 = self.n2
        fss = self.fss
        eta = self.eta
        period = self.period

        if(picture=='interaction'):
            temp = [item for n in index_cfc for item in list([n - n2, n - n0, n - n1, n + n1, n + n0, n + n2])]
            index_hfc = np.unique(temp) #Sorted by unique
            n_hfc = np.size(index_hfc)
            val_hfc = [np.zeros(fss.d_operator, dtype= 'complex128') for _ in range(n_hfc)]
            val_hfc_bigger = [np.zeros(fss.d_operator_bigger, dtype= 'complex128') for _ in range(n_hfc)]        
            # Loop through fourrier coeff of the control                         
            for i in range(n_cfc):
                i_tmp = index_cfc[i]
                val_tmp = val_cfc[i]
                j_tmp = index_hfc.searchsorted(i_tmp - n2)
                val_hfc[j_tmp] += val_tmp * eta * fss.anihil_sigmaM
                val_hfc_bigger[j_tmp] += val_tmp * eta * fss.anihil_sigmaM_bigger
                j_tmp = index_hfc.searchsorted(i_tmp - n1)
                val_hfc[j_tmp] += val_tmp * eta * fss.creat_sigmaM 
                val_hfc_bigger[j_tmp] += val_tmp * eta * fss.creat_sigmaM_bigger        
                j_tmp = index_hfc.searchsorted(i_tmp - n0)
                val_hfc[j_tmp] += val_tmp * fss.sigmaM    
                val_hfc_bigger[j_tmp] += val_tmp * fss.sigmaM_bigger
                j_tmp = index_hfc.searchsorted(i_tmp + n2)
                val_hfc[j_tmp] += val_tmp * eta * fss.creat_sigmaP
                val_hfc_bigger[j_tmp] += val_tmp * eta * fss.creat_sigmaP_bigger
                j_tmp = index_hfc.searchsorted(i_tmp + n1)
                val_hfc[j_tmp] += val_tmp * eta * fss.anihil_sigmaP
                val_hfc_bigger[j_tmp] += val_tmp * eta * fss.anihil_sigmaP_bigger        
                j_tmp = index_hfc.searchsorted(i_tmp + n0)
                val_hfc[j_tmp] += val_tmp * fss.sigmaP
                val_hfc_bigger[j_tmp] += val_tmp * fss.sigmaP_bigger    

                # Initial picture H(t) = Omega_trap a^T*a + Omega_int/2 sigmaZ + f(t) sigmaX (1 + eta (a^T + a))
        elif(picture == 'initial'):
            n_cfc = np.size(index_cfc)
            # If constant term (i.e. harmonic 0) is not part of cfc add it
            # If not do nothing
            constantTermInCfc = ((index_cfc == 0).any())
            if(constantTermInCfc):
                index_hfc = np.array(index_cfc)
            else:
                index_hfc = np.sort(np.append(index_cfc, 0))
            n_hfc = np.size(index_hfc)
            
            val_hfc = [np.zeros(fss.d_operator, dtype= 'complex128') for _ in range(n_hfc)]
            val_hfc_bigger = [np.zeros(fss.d_operator_bigger, dtype= 'complex128') for _ in range(n_hfc)]
            operator_tmp = fss.sigmaX + eta * np.dot((fss.anihil + fss.creat), fss.sigmaX)
            operator_bigger_tmp = fss.sigmaX_bigger + eta * np.dot((fss.anihil_bigger + fss.creat_bigger), fss.sigmaX_bigger)  

            # Constant term
            i_tmp = index_hfc.searchsorted(0)
            val_hfc[i_tmp] += (n0 * om / 2) * fss.sigmaZ + (nTr * om) * fss.number
            val_hfc_bigger[i_tmp] += (n0 * om / 2) * fss.sigmaZ_bigger + (nTr * om) * fss.number_bigger

            # Loop through fourrier coeff of the control                         
            for i in index_cfc:
                i_tmp = index_cfc.searchsorted(i)
                j_tmp = index_hfc.searchsorted(i)                        
                val_tmp = val_cfc[i_tmp]
                val_hfc[j_tmp] += val_tmp * operator_tmp
                val_hfc_bigger[j_tmp] += val_tmp * operator_bigger_tmp
	    
        elif(picture == 'strong'):
            alphaTmp = lambda t: (fourrierInverse(cfc, om, t))
	        
            theta1 = nTr * om
            theta2 = 1j * eta * (integrate.quad(lambda t: alphaTmp(t), 0, period)[0]) / period
            theta3 = np.square(eta) * (integrate.quad(lambda t: np.square(alphaTmp(t)), 0, period)[0])/ period
            theta4 = n0 * om/2 * (integrate.quad(lambda t: np.cos(2*alphaTmp(t)), 0, period)[0])/ period
            theta5 = - n0 * om * eta * (integrate.quad(lambda t: alphaTmp(t) * np.sin(2*alphaTmp(t)), 0, period)[0])/(period)
            theta6 = n0 * om/2 * (integrate.quad(lambda t: np.sin(2*alphaTmp(t)), 0, period)[0])/period
            theta7 = n0 * om * eta * (integrate.quad(lambda t: alphaTmp(t) * np.cos(2*alphaTmp(t)), 0, period)[0])/ period

            h0 = theta1 * fss.number + theta3 * self.fss.identity + theta4 * self.fss.sigmaZ + theta6 * self.fss.sigmaY
            h0 += theta2 * np.dot((self.fss.creat - self.fss.anihil), self.fss.sigmaX) + np.dot((self.fss.creat + self.fss.anihil), theta5* self.fss.sigmaZ + theta7* self.fss.sigmaY)

            h0_bigger = theta1 * self.fss.number_bigger + theta3 * self.fss.identity_bigger + theta4 * self.fss.sigmaZ_bigger + theta6 * self.fss.sigmaY_bigger
            h0_bigger += theta2 * np.dot((self.fss.creat_bigger - self.fss.anihil_bigger), self.fss.sigmaX_bigger) + np.dot((self.fss.creat_bigger + self.fss.anihil_bigger), theta5* self.fss.sigmaZ_bigger + theta7* self.fss.sigmaY_bigger)

            # Only able to compute the average Hamiltonian
            index_hfc = np.array([0])
            val_hfc = list([h0])
            val_hfc_bigger = list([h0_bigger])

        # Generate the periodic Hamiltonian (is it always periodic)
        Hfc_list = list([index_hfc, val_hfc, val_hfc_bigger])
        Hfc = ham.PeriodicHamiltonian(self.p, Hfc_list, self.fss)
        return Hfc

# ---------------------------
# Change of pictures
# ---------------------------
    def InterToInitPicture(self, state_t, tArray):
        """

        """
        transfo = self.expHrest(1.0, tArray)
        state_tranfo = state_t * transfo
        return state_tranfo

    def InitToInterPicture(self, state_t, tArray):
        """

        """
        transfo = self.expHrest(-1.0, tArray)
        state_tranfo = state_t * transfo
        return state_tranfo
    
    def expHrest(self, coeffs, tArray):
        """ Evol under H_rest
			e^{-i * coeffs * H_{rest} * t} // HAS BEEN FLATTENED AS DIAGO 
			with H_rest = om_trap n + sigma_Z * om_int/ 2 
        """
        n_mot = self.fss.n_mot
        om = self.om
        nTr = self.nTr
        n0 = self.n0
        
        t = tArray[:, np.newaxis]
        n = self.fss.arrayMotState
        s = np.concatenate((np.repeat(-1, n_mot), np.repeat(1, n_mot)))
        transfo_1 = np.exp(np.outer(t, -1.0j * coeffs * nTr * om * n)) # e^{-i n omega_trap t}
        transfo_2 = np.exp(np.outer(t, -1.0j * coeffs * n0 * om * s /2 )) # e^{-i sigma_z omega_int t}
        transfo = transfo_1 * transfo_2
        return transfo

    def StrongToInitPicture(self, state_t, tArray, coeffs):  
        """
		R = e^{-i alpha(t) sigmaX (1+eta  (a+a^T))}
		state_out = R state_in
        """  
        fss = self.fss
        eta = self.eta
        
        alphaTmp = lambda t: (fourrierInverse(coeffs, self.om, t))
        alpha = alphaTmp(self.tArray)[:, np.newaxis]        
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        tmp_sin = sinalpha * state_t
        tmp_cos = cosalpha * state_t
	    
        # taylor order0 in eta 
        # R1  = cos(alpha(t)) I - i sin(alpha(t)) sigmaX
        state_out = tmp_cos - 1j * fss.sigmaX(tmp_sin)
	    
        # taylor order 1 in eta 
        # R2 = R1 - alpha(t) * eta * sin(alpha(t)) * (a + a^T)
        #	      - 1.0j alpha(t) * eta * cos(alpha(t)) * (a + a^T) * sigmaX
        a_state_t = np.dot(state_t, np.transpose(fss.anihil + fss.creat))
        state_out -=  eta * alpha * tmp_sin * a_state_t
        state_out -=  1.0j * alpha * tmp_cos * eta * self.fss.sigmaX(a_state_t)

        #return state_t_init
        return state_out


    def InitToStrongPicture(self, state_t, tArray, coeffs):  
        """
        R = e^{i alpha(t) sigmaX (1+eta  (a+a^T))}
        phi_f = R phi_init
        
        """  
        fss = self.fss
        eta = self.eta
        
        alphaTmp = lambda t: (fourrierInverse(coeffs, self.om, t))
        alpha = alphaTmp(self.tArray)[:, np.newaxis]        
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        tmp_sin = sinalpha * state_t
        tmp_cos = cosalpha * state_t
	    
        # taylor order0 in eta 
        # R1  = cos(alpha(t)) I + i sin(alpha(t)) sigmaX
        state_out = tmp_cos + 1j * self.fss.sigmaX(tmp_sin)
	    
        # taylor order 1 in eta 
        # R2 = R1 - alpha(t) * eta * sin(alpha(t)) * (a + a^T)
        #	      + 1.0j alpha(t) * eta * cos(alpha(t)) * (a + a^T) * sigmaX
        a_state_t = np.dot(state_t, np.transpose(fss.anihil + fss.creat))
        state_out -=  eta * alpha * tmp_sin * a_state_t
        state_out +=  1.0j * alpha * tmp_cos * eta * self.fss.sigmaX(a_state_t)

        #return state_t_init
        return state_out

# ---------------------------
# Custom encoding of parameters into a controlfunction (cfc):
# 
# 
# ---------------------------
    def encodeCustomControl(self, params, scaling = 1, method = None):
        """
        Purpose:
            Transform a list of parameters into a cfc (control Fourier coefficients 
            which is a list[list of harmonics, list of values])
            + either provide the name of the encodingMethodToBeUsed or 
            + depending on the number of parameters provided it will pick the right
              encoding method (if not ow specified in method), e.g. if 2 parameters
              are  passed it will use encRes2h (put the first param as the amplitude
              on the internal resonance of the system and the second on some other harmonic)
            
            Use for optimization purposes: params are encoded into a cfc (which will 
            be subsequently simulated to generate a FOM)
        """

        if(method is None): #default 
            if(len(params) == 1):
                cfc = self.encRes1h(params)
            elif (len(params) == 2):
                cfc = self.encRes2h(params)
            elif (len(params) == 3):
                cfc = self.encRes2h(params)   
            elif(len(params) == 4):
                cfc = self.encRes4h(params)
            elif(len(params) == 5):
                cfc = self.encRes5h(params)
            else:
                cfc = self.encResNhNo0(params)
        else:
            fun  = getattr(self, method)
            cfc = fun(params)

        # All the amplitudes are scaled by  (n0 * om * tgt)
        om = self.om 
        n0 = self.n0
        cfc[1] = np.array(cfc[1]) * n0 * om * scaling

        return cfc

    def encRes1h(self, params):
        """
            harmonics put on the res freq (i.e. params[0] is put as the amplitude 
            on the (internal) resonance of the system
            f(t) = params0 x e{i om_int t} + params1 xe{i om_arb t} + c.c.
        """
        n0, f0 = self.n0, params[0]

        # control function
        list_harmo = np.array([-n0, n0])
        list_value = np.array([np.conjugate(f0), f0])
        cfc = list([list_harmo, list_value])
        return cfc


    def encRes2h(self, params):
        """
            params[0] is put as the amplitude on the (internal) resonance of 
            the system, params[1] on an 'arbitrary' harmonics
            f(t) = params0 x e{i om_int t} + params1 xe{i om_arb t} + c.c.
        """
        nTr = self.nTr
        n0, f0 = self.n0, params[0]
        n1, f1 = n0 + 2 * nTr + 1, params[1]

        # control function
        list_harmo = np.array([-n1, -n0, n0, n1])
        list_value = np.array([np.conjugate(f1), np.conjugate(f0), f0, f1])
        cfc = list([list_harmo, list_value])
        return cfc

    def encRes4h(self, params):
        """
            first harmonic on (internal) resonance // second spaced by n_tr 
            third by 2 n_tr // last arbitrary  
            f(t) = params0 x e{i om_int t} + params1 x e{i om_int + om_tr t} + 
                   params2 x e{i om_int + 2om_tr t} + params3 x e{i om_arb t} +
        """
        #First harmo (on resonance with the internal transition)
        nTr = self.nTr
        n0, f0 = self.n0, params[0]
        n1, f1 = n0 + nTr, params[1]
        n2, f2 = n0 + 2 * nTr, params[2]
        n3, f3 = n0 + 2 * nTr + 1, params[3]

        list_harmo = np.array([-n3, -n2, -n1, -n0, n0, n1, n2, n3])
        list_value = np.array([np.conjugate(f3), np.conjugate(f2), np.conjugate(f1), np.conjugate(f0), f0, f1, f2, f3])
        cfc = list([list_harmo, list_value])
        return cfc

    def encRes3h(self, params):
        """
            first harmonic on (internal) resonance // second spaced by n_tr 
            third by 2 n_tr // last arbitrary  
            f(t) = params0 x e{i om_int t} + params1 x e{i om_int + om_tr t} + 
                   params2 x e{i om_int + 2om_tr t} + params3 x e{i om_arb t} +
        """
        #First harmo (on resonance with the internal transition)
        nTr = self.nTr
        n0, f0 = self.n0, params[0]
        n1, f1 = n0 + nTr, params[1]
        n2, f2 = n0 + 2 * nTr + 1, params[2]

        list_harmo = np.array([-n2, -n1, -n0, n0, n1, n2])
        list_value = np.array([np.conjugate(f2), np.conjugate(f1), np.conjugate(f0), f0, f1, f2])
        cfc = list([list_harmo, list_value])
        return cfc

    def encRes5h(self, params):
        """
            first harmonic on (internal) resonance // second spaced by n_tr 
            third by 2 n_tr // 2 last arbitrary  
            f(t) = params0 x e{i om_int t} + params1 x e{i om_int + om_tr t} + 
                   params2 x e{i om_int + 2om_tr t} + params3 x e{i om_arb t} +
        """
        nTr = self.nTr
        n0, f0 = self.n0, params[0]
        n1, f1 = n0 + nTr, params[1]
        n2, f2 = n0 + 2 * nTr, params[2]
        n3, f3 = n0 + 2 * nTr + 1, params[3]
        n4, f4 = n0 + 4 * nTr - 1, params[4]

        list_harmo = np.array([-n4, -n3, -n2, -n1, -n0, n0, n1, n2, n3, n4])
        list_value = np.array([np.conjugate(f4), np.conjugate(f3), np.conjugate(f2), np.conjugate(f1), np.conjugate(f0), f0, f1, f2, f3, f4])
        cfc = list([list_harmo, list_value])
        return cfc


    def encResNhNo0(self, params):
        """
            zero constant value parameters are allocated as amplitudes on harmonics
            1,2,... and their complex conjugates on -1, -2, ...) 
        """
        nbharmonics = len(params)
        arrayparams = np.array(params)
        list_harmo = np.arange(-nbharmonics, nbharmonics + 1)
        list_value = np.concatenate((np.conj(arrayparams[::-1]), [0], arrayparams))
        cfc = list([list_harmo, list_value])
        return cfc

    def encResNh(self, params):
        """
            Parameters are allocated as amplitudes on harmonics 0, 1, 2,... (and 
            their complex conjugates on -1, -2, ...) 
              
        """
        nbharmonics = len(params) - 1
        arrayparams = np.array(params)
        list_harmo = np.arange(-nbharmonics, nbharmonics + 1)
        list_value = np.concatenate((np.conj(arrayparams[1:][::-1]), arrayparams))
        cfc = list([list_harmo, list_value])
        return cfc


#  ========================================================================   #
#                           AUXILLARY
#  ========================================================================   #



def gFun2(N0, n):
    return 2 * N0 /(np.square(n) - np.square(N0)) 

def gFun(N1, N2, n):
    return gFun2(N2, n) + gFun2(N1, n)


def derivateFR(funFR, omega = 1):
	"""
	Derivate function in Fourrier Representation list([Freq], [Ampl])
	i.e. F(t) = a * e(i * l * omega * t) -> f(t) = a * i * l * omega * e(i * l * omega * t) 
	([l], [a]) -> ([l], [l * omega * a])
	"""
	index_F = funFR[0]    
	val_F = funFR[1]    

	index_f = index_F
	val_f = [ index_F[harmo] * 1.0j * omega * val_F[harmo] for harmo in len(index_F)]

	return [index_f, val_f]


def integrateFR(funFR, omega = 1):
	"""
	Derivate function in Fourrier Representation list([Freq], [Ampl])
	i.e. f(t) = a * e(i * l * omega * t) -> F(t) = - i * a / (l * omega) * e(i * l * omega * t) 
	([l], [a]) -> ([l], [l * omega * a])
	"""
	index_f = funFR[0]    
	val_f = funFR[1]    
	
	index_F = index_f
	val_F = [(-1.0j * val_f[harmo]/ (index_f[harmo] * omega)) for harmo in len(index_f)]

	return [index_F, val_F]

def fourrierInverse(coeffs, omega, t):
    """
		Build a periodic function based on its fourrier coeffs
		[a, b, c, d, ...] -> a cos(w t) + b sin( w t) + c cos(2 w t) + d sin(2 w t) + ...
    """
    nbHarmonics = int(len(coeffs)/2)
    res = 0
    for i in range(nbHarmonics):
        res += coeffs[2*i]*np.cos((i+1)*omega*t) + coeffs[2*i+1]*np.sin((i+1)*omega*t)
    return res




















#  ========================================================================   #
#                           OPtimization for the strong regime(DEPRECIATED)
#   optimStrongCase
#     getTermsCostFunction
#     objectiveFun
#  ========================================================================   #

def optimStrongCase(nbHarmonics = 2 ,target =0, typeOptim = 'Y', algo = 'DE', weights = None, extremeBounds = 5):
    # Needed for the optim
    if (weights is None):
        if(typeOptim == 'DE'):
            weights = 20
        else:
            weights = 10
        
    nbArgs = nbHarmonics * 2
    objFun = lambda t: (objectiveFun(t, target = target, weighting = weights, typeOptim = typeOptim))
    

    if(algo == 'DE'):
        #Differential evolution
        bound1arg = (-extremeBounds, extremeBounds)          
        boundsDE = [bound1arg for _ in range(nbArgs)]
        res = optim.differential_evolution(objFun, boundsDE)
        optimCoeffs = res['x']
    else:
        # e.g. BFGS
        init = np.random.sample(nbArgs)
        res = optim.minimize(objFun, x0 = init, method=algo)
        optimCoeffs = res.x
        
    print(optimCoeffs)
    print(getTermsCostFunction(optimCoeffs))
    print('\n')

    return optimCoeffs



def getTermsCostFunction(coeffs):
    """ 
	Compute the different terms appearing in FME(H_{strong}) (using the TE on O(Eta))
    H(0) = i * eta * term1 sigY - eta * om_0/2 * term2 sigZ (a+aT) 
    """
    alphaTmp = lambda t: (fourrierInverse(coeffs, 1, t))

    # The one we want to maximize / or get close to a target
    term0 = 1/2 * np.abs(integrate.quad(lambda t: np.sin(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_Y
 
    # The ones we want as close as 0 as possible (TE up to order 1)
    term1 = np.abs(integrate.quad(lambda t: alphaTmp(t) * np.sin(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_z (a + aT)
    term2 = np.abs(integrate.quad(lambda t: alphaTmp(t) * np.cos(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_y (a + aT)
    term3 = np.abs(integrate.quad(lambda t: alphaTmp(t), 0, 2 * np.pi)[0]) # sigma_x (a + aT)
    term4 = 1/2 * np.abs(integrate.quad(lambda t: np.cos(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_z
    term5 = np.abs(alphaTmp(0)) # To force strobo collapsing to Identity

	# The ones we want as close as 0 as possible (TE up to order 2)
    term6 = 0.1 * np.abs(integrate.quad(lambda t: (alphaTmp(t) ** 2) * np.sin(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_y (a + aT)
    term7 = 0.1 * np.abs(integrate.quad(lambda t: (alphaTmp(t) ** 2) * np.cos(2*alphaTmp(t)), 0, 2 * np.pi)[0]) # sigma_z (a + aT)

    return np.array([term0, term1, term2, term3, term4, term5, term6, term7])
	#       strength = n0 * om * 2.60277 / (4 * np.pi)


def objectiveFun(coeffs, target = 0, weighting = 40, typeOptim = 'Y'):
    """ 
	Goal term1 close to target while other terms close to 0
    type: 
        Y/Y2 get the sigY terms but minimize others (Y2 takes into account TE(eta) in O(eta^2))
        Z/Z2 get the sigZ terms but minimize others (Z2 takes into account TE(eta) in O(eta^2))
    target:
        0 maximize the Y/Z term
        <>0 minimize (term - target) Y/Z term


    """
    resTmp = getTermsCostFunction(coeffs)
    #print(resTmp)

    if(typeOptim == 'Y'):
        indicePlus = [0]
        indiceMinus = [1,2,3,4,5]
    elif(typeOptim == 'Z'):
        indicePlus = [4]
        indiceMinus = [1,2,3,0,5]
    elif(typeOptim == 'Y2'):
        indicePlus = [0]
        indiceMinus = [1,2,3,4,5,6,7]
    elif(typeOptim == 'Z2'):
        indicePlus = [4]
        indiceMinus = [1,2,3,0,5,6,7]
    else:
        assert True, 'objectiveFun: type not implemented'

    if (target == 0):
        res = - sum(resTmp[indicePlus]) + weighting * sum(resTmp[indiceMinus])
    else:
        res = np.abs(resTmp[indicePlus] - target) + weighting * sum(resTmp[indiceMinus])
    return res
        

def alphaCoeffsToCfc(alphaCoeffs, omega):
    """
    	alphaCopeffs : [a, b, c, d, ...] 
    	--> alpha(t) a cos(w t) + b sin( w t) + c cos(2 w t) + d sin(2 w t) + ...
    	--> f(t) = d(alpha(t)) / dt = ()
    	--> cfc: f(t) as a list of harmonics number and amplitudes
    """
    nbHarmonics = int(len(alphaCoeffs)/2)
    harmonics = list()
    amplitudes = list()

    for i in range(nbHarmonics):
    	amplTmp = i * omega / 2 * (alphaCoeffs[2*i+1] + 1j * alphaCoeffs[2*i])
    	harmonics.append(-i)
    	amplitudes.append(np.conj(amplTmp))
    	harmonics.append(i)
    	amplitudes.append(amplTmp)

    return np.array(harmonics), np.array(amplitudes)

def genCoeffsHStrong0 (alphaFun, period, om, n0, nTr, eta):
	theta1 = nTr * om
	theta2 = 1j * eta * (integrate.quad(lambda t: alphaFun(t), 0, period)[0]) / period
	theta3 = np.square(eta) * (integrate.quad(lambda t: np.square(alphaFun(t)), 0, period)[0])/ period
	theta4 = n0 * om/2 * (integrate.quad(lambda t: np.cos(2*alphaFun(t)), 0, period)[0])/ period
	theta5 = - n0 * om * eta * (integrate.quad(lambda t: alphaFun(t) * np.sin(2*alphaFun(t)), 0, period)[0])/(period)
	theta6 = n0 * om/2 * (integrate.quad(lambda t: np.sin(2*alphaFun(t)), 0, period)[0])/period
	theta7 = n0 * om * eta * (integrate.quad(lambda t: alphaFun(t) * np.cos(2*alphaFun(t)), 0, period)[0])/ period

	return theta1, theta2, theta3, theta4, theta5, theta6, theta7

