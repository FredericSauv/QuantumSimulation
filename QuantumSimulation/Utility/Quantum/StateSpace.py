#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:15:32 2017

@author: frederic
"""
import numpy as np
import pdb 

if __name__ == '__main__':
    import sys
    sys.path.append("../../")
    import Utility.Quantum.Hamiltonian as ham
    import Utility.Helper as ut    
else:
    from . import Hamiltonian as ham
    from .. import Helper as ut

#TODO: add energy_average in SS // Not Here but in Hamiltonians
#TODO: most of the methods should be static/cls method

#==============================================================================
#                   FiniteHilbertSpace
#TODO: Write applyOperator
# >> FiniteHilbertSpace >>   Qudits
#                       >>   Qubits  >> OneQbit
#                                   >> TwoQbits 
#============================================================================== 
class FiniteHilbertSpace:
    """ Class: Finite Hilbert Space (with dim(H) = d)
        Implementation of probabilities, inner product, fidelities measure, Expected Value of an operator
        Convention: quantum state over t time points are represented as t x d arrays
    """
    
    def __init__(self, list_dim = [2]):
        self.list_dim = list_dim #nb DoF
        self.dim = np.product(list_dim) # dim of the total Hspace
        self.ground = np.zeros(self.dim, dtype = 'complex128')
        self.ground[0] = 1.0
        

# ---------------------------
# Static methods acting on kets / operators
# 
# proba / ip / norm / distance / heating / fidelity / apply {X,Y,Z} 
# ---------------------------
    @staticmethod
    def probaFromState(ket):
        """
        Purpose:
            transform probability amplitudes in proba
        """
        return np.square(np.abs(ket))
    
    @staticmethod
    def ip(ket1, ket2):
        """
        Purpose:
            Compute (complex) inner product 
        Dimensions:
            + if 1d Compute <ket1 | ket2>
            + if 2d kets (i.e. t rows d columns)
            t should be the time index and d the dim of the Hilbert space 
            >> returns <ket1(t) | ket2(t)> a t entries array
            + Nd last dim is the Hilbert space 
        TODO: Still don't deal with all the dims
        """
        assert (ket1.shape == ket2.shape), "unconsistent dim of ket1, ket2"
        res = np.sum(np.conj(ket1) * ket2, len(ket1.shape)-1)
        return np.abs(res)
    
    @staticmethod    
    def norm(ket1):        
        """
        Purpose:
            Compute the usual norm ||ket1|| = sqrt(<ket1 | ket1>) (float)
        """
        return np.sqrt(FiniteHilbertSpace.ip(ket1,ket1))
    
    @staticmethod
    def is_unit_norm(ket1):
        """ verif ||ket1|| = 1
        """
        return np.allclose(FiniteHilbertSpace.norm(ket1), 1.0)

    @staticmethod    
    def measurement(ket1, nb = 1, measur_basis = None, num_precis = 1e-6):        
        """Projective measurement in the basis in which the ket is represented 
        (if ket_proj is None) or in any other basis
        Returns an array with each entry corresponding to the frequency of the 
        measurement based on nb measurements
        
        !!CONVENTION!! ket_proj = N x D (with D dim of the Hilbert space and N the
        number of elements provided)
        
        #TODO: only work for one ket (as opposed to array of kets over time)
        # Ortho measurement only
        #TODO: Could probably do better (concise + dealing with different shapes of
        data)
        """
        #pdb.set_trace()
        dim_ket = len(ket1)
        

        single_value = False
        if(measur_basis is None):
            measur_basis = np.eye(dim_ket) #Computational basis
        elif (len(measur_basis.shape) == 1):
            assert(len(measur_basis) == dim_ket), "problem with the dim"
            measur_basis = [measur_basis] # just reshape it
            single_value = True
        else:
            assert(measur_basis.shape[1] == dim_ket)
            
        index_measure = np.arange(len(measur_basis))
        proj = [FiniteHilbertSpace.ip(basis, ket1) for basis in measur_basis]
        proba = FiniteHilbertSpace.probaFromState(proj)
        assert(np.sum(proba)<=(1.0 + num_precis)), "not a valid proba distrib" #should it be relaxed 
        proba_cum = np.concatenate(([0],np.cumsum(proba)))
        samples = np.random.sample(nb)
        observations = [np.sum((samples > proba_cum[i]) * (samples <= proba_cum[i+1])) for i in index_measure]
        frequencies = np.array(observations)/nb
        if(single_value):
            assert(len(frequencies) == 1), "pb"
            frequencies = frequencies[0]
        return frequencies
    
# ---------------------------
# Fidelity measures (2 conventions with square or not)
# ---------------------------
    @staticmethod
    def fidelity(ket1, ket2):
        # Compute |<ket1 | ket2>|^2 
        return np.abs((FiniteHilbertSpace.ip(ket1, ket2)))
    
    @staticmethod
    def fidelity_avg(ket1_t, ket2_t):
        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
        # same as fidelity_distance but for states over time
        return np.average(FiniteHilbertSpace.fidelity(ket1_t, ket2_t))

    @staticmethod
    def fidelity2(ket1, ket2):
        # Compute |<ket1 | ket2>|^2 
        return np.square(np.abs(FiniteHilbertSpace.ip(ket1, ket2)))
    
    @staticmethod
    def fidelity2_avg(ket1_t, ket2_t):
        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
        # same as fidelity_distance but for states over time
        return np.average(FiniteHilbertSpace.fidelity2(ket1_t, ket2_t))    

# ---------------------------
# Methods for operators
#
# ---------------------------
    def ket(self, i, j):
        res = np.zeros([self.dim, self.dim])
        res[i, j] = 1
        return res
    
    @staticmethod
    def commut(op1, op2):
        """
        Purpose:
            Compute the commutator of two operators
            [A,B] = AB - BA 
        """
        return np.dot(op1, op2) - np.dot(op2,op1)

    @staticmethod
    def expectation_value(left, operator, right = None):
        """
        Purpose:
            Compute <left | operator | right> 
        """
        if(right is None):
            right = left
        leftconj = np.transpose(np.conj(left))
        return np.dot(leftconj, np.dot(operator, right))
    
    @staticmethod
    def projection(ket, basis):
        """ 
        coeff[i] = <basis[i], ket>
        where basis is a set of column vectors
        """
        coeffs = np.dot(ket, basis.conj()) 
        return coeffs

    @staticmethod
    def dagger(H):
        return np.conj(np.transpose(H))

    @staticmethod
    def isHermitian(H):
        """
        Purpose:
            check if an operator is Hermitian
        """
        return np.allclose(H, FiniteHilbertSpace.dagger(H))

    @staticmethod
    def isUnitary(H):
        """
        Purpose:
            
        """
        test = False
        shape = H.shape
        if shape[0] == shape[1]:
             test = np.allclose(np.dot(H, FiniteHilbertSpace.dagger(H)), np.eye(shape[0]))
        return test

    @staticmethod
    def Exp(op, coeff):
        """
        Purpose:
            
        """
        hvals, hvects = ut.degenEV(op)
        res = np.zeros_like(op, dtype = 'complex128')
        for i in range(len(hvals)):
            res += np.exp(coeff * hvals[i]) * np.outer(hvects[:,i], hvects[:,i])
        return res
# ---------------------------
# FUNCTIONS ACTING ON PROBA
# proba distance
# ---------------------------
    @staticmethod
    def proba_distance(p1, p2):
        # Compute |<ket1 | ket2>|^2
        assert (p1.shape == p2.shape), "unconsistent dim of ket1, ket2"
        tmp = np.abs(p1 - p2)/2
        res = np.sum(tmp, len(tmp.shape)-1)
        return res
    


    
    
    
#==============================================================================
#                   Qbits class
#  Representation of a finite number of qbits (N 2-levels systems):
#  By convention basis is ordered as follow:
#  |00..00>, |00..01>, |00..10>, ..., |10..00>, |10..01>..., |11..11>}
#
#==============================================================================
class QuDits(FiniteHilbertSpace):    
    def __init__(self, nbLevels = 2):
        FiniteHilbertSpace.__init__(self, list_dim = np.repeat(nbLevels, 1))
        self._GenUsualOperators()
        self._GenUsualStates()
        #self.name_state ={'0': self.phi0, 'plus': self.phiPlus}
        self.name_operator ={'X': self._X, 'F': self._F,'Z': self._Z, 'I': self._I}

                
    def _GenUsualOperators(self):
        """
        Purpose:
            Generate the Pauli matrix associated to a specified qbit
            It's a matrix with S*S entries (with S = 2^N) as it acts on the 
            whole space
            
            
            + X_q = |0><1|_q + |1><0|_q
            + Z = - |0><0| + |1><1|
            + Y = -1.0j |0><1| + 1.0j |1><0|
            + I = |0><0| + |1><1|
            + H = 1/ sqrt(2) |0><+| + |0><-|
        """
        N = self.dim
        
        # I
        self._I = np.diag(np.repeat(1, N))
        
        #Z
        if ut.is_even(N):
            lim = int(N/2)
            diag = [i for i in range(-lim, lim+1) if i != 0]
        else:
            lim = int((N-1)/2)
            diag = np.arange(-lim, lim+1)
        self._Z = np.diag(diag)
                
        #F
        row_gen = np.arange(0, N)
        col_gen = np.arange(0, N)
        gen = np.outer(row_gen, col_gen)
        self._F = 1/np.sqrt(N) * np.exp(2.0j*np.pi/N * gen)
        
        #X
        self._X = np.dot(self._F, np.dot(self._Z, FiniteHilbertSpace.dagger(self._F)))
   

    def _GenUsualStates(self):
        """
            _zBasis[j] is the jth (smaller) Evector of the Z operator
            _zBasis[j] is the jth (smaller) Evector of the X operator
        """
        N = self.dim
        self._zBasis = np.diag(np.repeat(1, N))
        self._xBasis = np.dot(self._F, self._zBasis).T

        #self.phiPlus = 1/np.sqrt(N) * np.ones(N, dtype = 'complex128') 
        #self.phi0 = self._computBasis[0]
        

    def GetStateByName(self, name):
        """
        Purpose:
           Provide a way to call a state by its name
           e.g. getStateByName("phi0") >> |00..00>
        """
        if (name[0] == "x"):
            nb = int(name[1:])
            state = self._xBasis[nb]
        elif(name[0] == "z"):
            nb = int(name[1:])
            state = self._zBasis[nb]
        elif(ut.is_int(name)):
            number = int(name)
            state = self._zBasis[number]    
        else:
            raise NotImplementedError()
            
        return state
    
#==============================================================================
#                   Qbits class
#  Representation of a finite number of qbits (N 2-levels systems):
#  By convention basis is ordered as follow:
#  |00..00>, |00..01>, |00..10>, ..., |10..00>, |10..01>..., |11..11>}
#
#==============================================================================
class QuBits(FiniteHilbertSpace):    
    def __init__(self, nbQbits = 2):
        FiniteHilbertSpace.__init__(self, list_dim = np.repeat(2, nbQbits))
        self._nbQbits = nbQbits
        self._GenIndicesN()
        self._GenUsualStates()
        self.name_state ={'zero': self.phi0, 'one': self.phi1,'minus': self.phiMinus, 'plus': self.phiPlus}
        #self.name_pauli ={'X': self.sigmaX, 'Y': self.sigmaY,'Z': self.sigmaZ, 'I': self.identity}

    def _GenIndicesN(self):
        """
        Purpose:
            For N qbits the basis is given by:
            |00..00>, |00..01>, |00..10>, ..., |10..00>, |10..01>..., |11..11>    
        
            self._indices[i][j] is a list of indices of the basis elements where 
            qbit j (0..N) is in state i (0 or 1)
            
            furthermore (for any k) basis elements with indices 
            self._indices[0][j][k]  and  self._indices[1][j][k] are related by 
            a flip (i.e. 0->1) on the j-th qbit
            
        """
        N = self._nbQbits
        S = self.dim
        index0 = [[i for i in np.arange(S) if (i % 2**(N-q) < 2**(N-q-1))] for q in np.arange(N)]
        index1 = [[i for i in np.arange(S) if (i % 2**(N-q) >= 2**(N-q-1))] for q in np.arange(N)]
        self._indices = [index0, index1]

                
    def _Gen1QbitOperator(self, typePauli = 'Z', qbit = 0):
        """
        Purpose:
            Generate the Pauli matrix associated to a specified qbit
            It's a matrix with S*S entries (with S = 2^N) as it acts on the 
            whole space
            
            
            + X_q = |0><1|_q + |1><0|_q
            + Z = - |0><0| + |1><1|
            + Y = -1.0j |0><1| + 1.0j |1><0|
            + I = |0><0| + |1><1|
            + H = 1/ sqrt(2) |0><+| + |0><-|
        """
        
        #assert (typePauli in self.name_pauli.keys()), 'wrong Pauli name'
        S = self.dim
        res = np.zeros([S, S], dtype = 'complex128')
        index0 = self._indices[0][qbit]
        index1 = self._indices[1][qbit]        
        
        if(typePauli == 'I'):
            np.fill_diagonal(res, 1)
        
        elif(typePauli == 'Z'):
            indices_to_update = np.column_stack((index0, index0))
            for i,j in indices_to_update:
                res[i,j] = - 1 # i.e. - |0><0|_qbit
            indices_to_update = np.column_stack((index1, index1))
            for i,j in indices_to_update:
                res[i,j] = 1 # i.e. |1><1_qbit|
        
        elif(typePauli == 'X'):
            indices_to_update = np.column_stack((index0, index1))
            for i,j in indices_to_update:
                res[i,j] = 1
            indices_to_update = np.column_stack((index1, index0))
            for i,j in indices_to_update:
                res[i,j] = 1

        elif(typePauli == 'Y'):
            indices_to_update = np.column_stack((index0, index1))
            for i,j in indices_to_update:
                res[i,j] = -1.0j # i.e. - j |0><1|_qbit
            indices_to_update = np.column_stack((index1, index0))
            for i,j in indices_to_update:
                res[i,j] = 1.0j # i.e. + j |0><0|_qbit
                
        elif(typePauli == 'H'):
            indices_to_update = np.column_stack((index0, index0))
            for i,j in indices_to_update:
                res[i,j] = 1/np.sqrt(2)
            indices_to_update = np.column_stack((index0, index1))
            for i,j in indices_to_update:
                res[i,j] = 1/np.sqrt(2)
            indices_to_update = np.column_stack((index1, index0))
            for i,j in indices_to_update:
                res[i,j] = 1/np.sqrt(2)
            indices_to_update = np.column_stack((index1, index1))
            for i,j in indices_to_update:
                res[i,j] = -1/np.sqrt(2)
        
        else:
            raise NotImplementedError()
            
        return res
    
    def applyOperators(self, state = 'phi0', listOperator = ['X1', 'Z3']):
        """
        Purpose:
            Apply simple operators without building the matrix representation
            (should be faster)
            
        e.g.
            applyOperators([1,0,0,0], ['X1', 'Z2']) >>  res = Z2 X1 state
        """
        raise NotImplementedError

    def _GenUsualStates(self):
        """
            |00..00>, |11..11>, |++..++>, |-- -->
        """
        S = self.dim
        N = self._nbQbits
        self.phi0 = np.zeros(S, dtype = 'complex128')
        self.phi0[0] = 1.0 + 0.0j
        self.phi1 = np.zeros(S, dtype = 'complex128')
        self.phi1[-1] = 1.0 + 0.0j
        self.phiPlus = 2**(-N/2) * np.ones(S, dtype = 'complex128') 
        self.phiMinus = self.phiPlus * np.tile([-1, 1], int(S/2))

    def GenSeparableStates(self, list_amp):
        """
            list_amp = [[phi1_0, phi1_1], ...,  [phiN_0, phiN_1]]
            
            |Phi> = |Phi1> (x) |Phi2> .... (x) |PhiN>
            with |phiI> = phiI_0 |0> + phiI_1 |1>
        """
        raise NotImplementedError()
        

    def GetStateByName(self, name):
        """
        Purpose:
           Provide a way to call a state by its name
           e.g. getStateByName("phi0") >> |00..00>
        """
        if name in self.name_state:
            state = self.name_state.get(name)
        else:
            raise NotImplementedError()
            
        return state

    def ChangeOfBasis(self, state, initBasis = "01", targetBasis = "+-"):
        """
        Purpose:
            Change of basis returns the coordinates in the new basis
        """
        raise NotImplementedError()
        if(initBasis == targetBasis):
            state = state
            
        elif(initBasis == "01" and targetBasis == "+-"):
            for n in self._nbQbits:
                Htmp = self._Gen1QbitOperator('H', n)
                state = np.dot(Htmp, state)
                
        elif(initBasis == "+-" and targetBasis == "01"):    
            for n in self._nbQbits:
                Htmp = self._Gen1QbitOperator('H', n)
                state = np.dot(Htmp, state)
                
        else:
            raise NotImplementedError()
        
        
        return state
#==============================================================================
#                   OneQbit class
#  Representation of a qbit (2-levels system):
#  Inherits from Qbits and implement a few other methods:
#
#==============================================================================
class OneQbit(QuBits):
    """
    Class defining the representaion of vectors/operators for a two levels system
    in the {|0>, |1>} basis

    """
    def __init__(self):
        
        QuBits.__init__(self, nbQbits = 1)
        self._Gen2QbitsOperators()
        self.name_pauli ={'X': self._X, 'Y': self._Y, 'Z': self._Z, 'I': self._I}

    def _Gen2QbitsOperators(self):
        """
            |00..00>, |11..11>, |++..++>, |-- -->
        """
        self._X = self._Gen1QbitOperator('X', 0) 
        self._Y = self._Gen1QbitOperator('Y', 0) 
        self._Z = self._Gen1QbitOperator('Z', 0) 
        self._I = self._Gen1QbitOperator('I', 0)
        

    def GetPauliByName(self, name):
        if name in self.name_pauli :
            pauli = self.name_pauli.get(name)
        else:
            raise NotImplementedError()
            
        return pauli

    @staticmethod
    @ut.vectorise_method_ndim(1)
    def _get_bs_coordinates(ket):
        """ get cartesian position on the bloch sphere representation
        """        
        theta, phi = OneQbit._get_bs_angles()
        x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        return x, y, z


    @staticmethod
    @ut.vectorise_method_ndim(1)
    def _get_bs_angles(ket1):
        """ get theta and phi involved in the Bloch Sphere representation
        """
        assert (len(ket1) == 2), "should be a 2 elements array"
        assert OneQbit.is_unit_norm(ket1), "should be a unit norm vector"
        theta = 2 * np.arcos(np.abs(ket1[0]))
        phi = np.angle(ket1[1]) - np.angle(ket1[0])
        return theta, phi
#==============================================================================
#                   TwoQbits class
#  Representation of 2 qbits (2-levels system):
#  Inherits from Qbits and implement a few other methods:
#
#==============================================================================        
class TwoQbits(QuBits):
    """
    Class defining the representaion of vectors/operators for a two levels system
    in the {|0>, |1>} basis

    """
    def __init__(self, nbQbits = 2):
        QuBits.__init__(self)
        self._Gen2QbitsStates()
        self._Gen2QbitsOperators()

    def _Gen2QbitsStates(self):
        """
            |00..00>, |11..11>, |++..++>, |-- -->
        """
        self.name_state['bell0'] = 1/np.sqrt(2) * np.array([1,0,0,1], dtype= 'complex128')
        self.name_state['bell1'] = 1/np.sqrt(2) * np.array([1,0,0,- 1], dtype= 'complex128')
        self.name_state['supp1'] = 1/2 * np.array([1, 1, 1, 1], dtype= 'complex128')
        
    def _Gen2QbitsOperators(self):
        """
            |00..00>, |11..11>, |++..++>, |-- -->
        """
        self._X = [self._Gen1QbitOperator('X', q) for q in range(self._nbQbits)]
        self._Y = [self._Gen1QbitOperator('Y', q) for q in range(self._nbQbits)]
        self._Z = [self._Gen1QbitOperator('Z', q) for q in range(self._nbQbits)]

        
    
        

#==============================================================================
#                   StateSpace
#TODO: Inherit from FiniteHilbertSpace
#TODO: Change name to InternalMotionalSpace // TwoDoFSpace ...
#TODO FloquetHilbertSpace AbstractClass
#============================================================================== 
class StateSpace:

#class SpinInHOscillator: 
    """Class defining the representaion of vectors/operators in a state space 
       with motional and internal DoF (internal x motional)

    METHODS:
        - 

        -

    CONVENTION:
        - 1 Basis = {g0, g1, ..., gN, e0, e1, ..., eN}
        - 2 Basis = {g0, e0, g1, e1,.., gN, eN}
#    def fidelity_distance_t(self, ket1_t, ket2_t):
#        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
#        # same as fidelity_distance but for states over time
#        return np.square(np.abs(np.sum(np.conj(ket1_t) * ket2_t,1)))
    


    """
# ---------------------------
# Init / changes
# ---------------------------         
    def __init__(self, n_mot, n_int, params): 
        self.n_mot = n_mot
        self.n_int = n_int
        self.convention = params.convention
        self.genRepresentationSS()


    def changeNMot(self, new):
        self.n_mot = new
        self.genRepresentationSS()
    

    def genRepresentationSS(self):
        self.n_hilbert = self.n_mot * self.n_int
        self.d_operator = [self.n_hilbert, self.n_hilbert]
        
        self.sigmaP = self.representSigmaPlusOp(self.n_mot, self.n_int,1, self.convention)
        self.sigmaM = self.representSigmaMinusOp(self.n_mot, self.n_int,1, self.convention)
        self.creat = self.representCreationOp(self.n_mot, self.n_int,1, self.convention)
        self.anihil = self.representAnihilOp(self.n_mot, self.n_int,1, self.convention)
        self.sigmaZ = self.representSigmaZOp(self.n_mot, self.n_int,1, self.convention)
        self.sigmaX = self.representSigmaXOp(self.n_mot, self.n_int,1, self.convention)
        self.sigmaY = self.representSigmaYOp(self.n_mot, self.n_int,1, self.convention)
        self.number = self.representNumberOp(self.n_mot, self.n_int,1, self.convention)
        self.identity = np.eye(self.n_hilbert)        
        
        self.creat_sigmaP = np.dot(self.sigmaP, self.creat)
        self.anihil_sigmaP = np.dot(self.sigmaP, self.anihil)
        self.creat_sigmaM = np.dot(self.sigmaM, self.creat)
        self.anihil_sigmaM = np.dot(self.sigmaM, self.anihil)
        
        self.creat_sigmaZ = np.dot(self.sigmaZ, self.creat)
        self.anihil_sigmaZ = np.dot(self.sigmaZ, self.anihil)
        self.number_sigmaZ = np.dot(self.sigmaZ, self.number)
        
               
        #workaround: represent operators in a slightly bigger space and do the truncation
        #            after multiplication of the operators
        # not very clean
        self.n_bigger = 2        
        self.n_mot_bigger = self.n_mot + self.n_bigger
        self.n_hilbert_bigger = self.n_mot_bigger * self.n_int
        self.d_operator_bigger = [self.n_hilbert_bigger, self.n_hilbert_bigger]
        self.sigmaP_bigger = self.representSigmaPlusOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.sigmaM_bigger = self.representSigmaMinusOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.sigmaX_bigger = self.representSigmaXOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.sigmaY_bigger = self.representSigmaYOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.sigmaZ_bigger = self.representSigmaZOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.creat_bigger = self.representCreationOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)
        self.anihil_bigger = self.representAnihilOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)

        self.creat_sigmaP_bigger = np.dot(self.sigmaP_bigger, self.creat_bigger)
        self.anihil_sigmaP_bigger = np.dot(self.sigmaP_bigger, self.anihil_bigger)
        self.creat_sigmaM_bigger = np.dot(self.sigmaM_bigger, self.creat_bigger)
        self.anihil_sigmaM_bigger = np.dot(self.sigmaM_bigger, self.anihil_bigger)  
        
        self.identity_bigger = np.eye(self.n_hilbert_bigger)  
        self.number_bigger = self.representNumberOp(self.n_mot + self.n_bigger, self.n_int,1, self.convention)

        
        if(self.convention == 1):
            self.arrayMotState = np.concatenate((np.arange(self.n_mot), np.arange(self.n_mot)))
            self.arrayIntState = np.concatenate((np.repeat(0, self.n_mot), np.repeat(1, self.n_mot)))
            self.slice_mot = [slice(i, self.n_hilbert, self.n_mot) for i in range(self.n_mot)]
            self.slice_int = [slice(i * self.n_mot, (i+1) * self.n_mot, 1)  for i in range(self.n_int)]

        elif(self.convention == 2):
            self.arrayMotState = np.reshape([[i,i] for i in range(self.n_mot)], self.n_hilbert)
            self.arrayIntState = np.repeat(np.arange(self.n_int), self.n_mot)
            self.slice_int = [slice(i, self.n_hilbert, self.n_int) for i in range(self.n_int)]
            self.slice_mot = [slice(i * self.n_int, (i+1) * self.n_int, 1)  for i in range(self.n_mot)]     

  

# ---------------------------
# Main opeartors represented in the State Space
# ---------------------------
    def representSigmaPlusOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = np.zeros([n_mot * n_int, n_mot * n_int], dtype = 'complex128')
        if(convention == 1):
            for i in range(n_mot):
                operator[i + n_mot, i] = coeffs
        elif(convention == 2):
            for i in range(n_mot):
                operator[2*i + n_mot, 2*i] = coeffs
        return operator
        
    def representSigmaMinusOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = self.representSigmaPlusOp(n_mot, n_int, coeffs, convention)
        return operator.transpose().conj()
        
    def representCreationOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = np.zeros([n_mot * n_int, n_mot * n_int], dtype = 'complex128')
        if(convention == 1):
            for i in range(n_mot-1):
                operator[i + 1, i] = coeffs * np.sqrt(i+1)
                operator[i + n_mot + 1, i + n_mot] = coeffs * np.sqrt(i+1)
        elif(convention == 2):
            for i in range(2 * n_mot - 2):
                operator[2*i + 2, 2*i] = coeffs * np.sqrt(i+1)
    
        return operator
        
    def representAnihilOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = self.representCreationOp(n_mot, n_int, coeffs, convention)
        return operator.transpose().conj()
    
    def representSigmaZOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = np.zeros([n_mot * n_int, n_mot * n_int], dtype = 'complex128')
        if(convention == 1):
            tmp = np.append(np.repeat(-1, n_mot), np.repeat(1, n_mot))
            operator = np.diag(tmp)
        elif(convention == 2):
            tmp = [j for i in range(n_mot) for j in np.array([-1,1])]
            operator = np.diag(tmp)
        return operator

    def representSigmaYOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = 1j* self.representSigmaPlusOp(n_mot, n_int, coeffs, convention) - 1j*self.representSigmaMinusOp(n_mot, n_int, coeffs, convention)
        return operator

    def representSigmaXOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator =  self.representSigmaMinusOp(n_mot, n_int, coeffs, convention)
        operator += self.representSigmaPlusOp(n_mot, n_int, coeffs, convention)
        return operator
            
    def representNumberOp(self, n_mot = 10, n_int = 2, coeffs=1, convention = 1):
        operator = np.zeros([n_mot * n_int, n_mot * n_int], dtype = 'complex128')
        if(convention == 1):
            tmp = np.append([i for i in range(n_mot)], [i for i in range(n_mot)])
            operator = np.diag(tmp)
        #TODO: Verif
        elif(convention == 2):
            tmp = [i for i in range(n_mot) for j in np.array([-1,1])]
            operator = np.diag(tmp)
        return operator

# ---------------------------
# Functions acting on operators
# Commutation
# ---------------------------
    def commut(self, op1, op2):
        return np.dot(op1, op2) - np.dot(op2,op1)
        

# ---------------------------
# FUNCTIONS ACTING ON STATE
# proba / Truncation / ip / norm / distance / heating / fidelity / apply {X,Y,Z} 
# ---------------------------
    def probaFromState(self, state):
        """
        Purpose:
            transform probability amplitudes in proba
        """
        return np.square(np.abs(state))
       

    def trunc_m(self, to_trunc, n_mot_trunc):
        """
        Purpose:
            Truncate the motional basis representation of a state
        """
        n_mot_init = to_trunc.shape[0]/self.n_int
        res = np.zeros([n_mot_trunc * self.n_int, self.n_mot * self.n_int], dtype = 'complex128')
        if(self.convention == 1):
            index_tmp = np.hstack((np.arange(0,n_mot_trunc), np.arange(n_mot_init, n_mot_init+n_mot_trunc, dtype = 'int64')))
        elif(self.convention == 2):
            index_tmp = np.arange(0, n_mot_trunc * self.n_int)
        res = to_trunc[np.ix_(index_tmp, index_tmp)]
        return res    
    

    def ip(self, ket1, ket2):
        """
        Purpose:
            Compute <ket1 | ket2>
            if dim(kets) = t * d t should be the number of times and d the 
            dim of the Hilbert space >> <ket1(t) | ket2(t)> a t entries array
        """
        assert (ket1.shape == ket2.shape), "unconsistent dim of ket1, ket2"

        if(len(ket1.shape) == 1):
            res = np.dot(np.conj(ket1), ket2)
            
        elif(len(ket1.shape) == 2):
            res = np.sum(np.conj(ket1) * ket2,1)
            
        else:
            raise NotImplementedError()
            
        return res
        

    def norm(self, ket1):
        """
        Purpose:
            Compute sqrt(<ket1 | ket1>)
        """
        return np.sqrt(self.ip(ket1,ket1))
    

    def fidelity(self, ket1, ket2):
        # Compute |<ket1 | ket2>|^2 
        return np.abs((self.ip(ket1, ket2)))
    
    
    def fidelity_avg(self, ket1_t, ket2_t):
        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
        # same as fidelity_distance but for states over time
        return np.average(self.fidelity(ket1_t, ket2_t))

    def fidelity2(self, ket1, ket2):
        # Compute |<ket1 | ket2>|^2 
        return np.square(self.abs(self.ip(ket1, ket2)))
    

    def fidelity2_avg(self, ket1_t, ket2_t):
        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
        # same as fidelity_distance but for states over time
        return np.average(self.fidelity2(ket1_t, ket2_t))    
    
    

    def heat_t(self, ket1_t):
        """
        Purpose:
            Heating is defined as the sum of population not in the ground 
            motional state (assume that the sum of pop = 1)
        """
        p1_t = self.probaFromState(ket1_t)
        heat = 1 - np.sum(p1_t[:,self.slice_mot[0]],1)
        return heat

    def heat_avg(self, ket1_t):
        heat = np.avg(self.heat_t(ket1_t))
        return heat

    def sigmaY(self, state_t):
        """ Apply sigmaX WITHOUT using matrix multiplication
        state_t : t x d with d the dimension of the HilbertSpace
        """ 
        return np.concatenate((-1.0j * state_t[:, self.slice_int[1]], 1.0j * state_t[:, self.slice_int[0]]), axis=1)

    def sigmaX(self, state_t):
        """ Apply sigmaX WITHOUT using matrix multiplication
        state_t : t x d with d the dimension of the HilbertSpace
        """ 
        return np.concatenate((state_t[:, self.slice_int[1]], state_t[:, self.slice_int[0]]), axis=1)


    def sigmaZ(self, state_t):
        """ Apply sigmaZ without using matrix multiplication
        state_t : t x d with d the dimension of the HilbertSpace
        """ 
        return np.concatenate((state_t[:, self.slice_int[0]], - state_t[:, self.slice_int[0]]), axis=1)

    def expectation_value(self, left, operator, right):
        # Compute <left | operator | right> 
        leftconj = np.transpose(np.conj(left))
        return np.dot(leftconj, np.dot(operator, right))   

# ---------------------------
# FUNCTIONS ACTING ON PROBA
# proba / Truncation / ip / norm / distance / heating / fidelity
# ---------------------------
    def proba_distance(self, p1, p2):

        return np.sum(np.abs(p1 - p2))/2 
    
    def proba_distance_t(self, p1_t, p2_t):
        # assume ket1_t ket2_t are 2d arrays - rows: time column: HSpace
        # same as fidelity_distance but for states over time
        return np.sum(np.abs(p1_t - p2_t),1)/2
    
 

# ---------------------------
# Change of basis (for internal dof)
# 
# ---------------------------
    def chgeComputBasis(self, state_t, newbasis= '+-'):
        """ 
        newbasis = '+-' from {g, e} to {+, -} with +/- = 1/sqrt(2) (g +/- e)
        """
        state_ncb = np.zeros_like(state_t)
        if(newbasis == '+-'):
            for m in np.arange(self.n_mot):
                indices = self.slice_mot[m]
                toreplace = state_t[:, indices]
                r1 = 1/np.sqrt(2) * (toreplace[:,0] + toreplace[:,1])
                r2 = 1/np.sqrt(2) * (toreplace[:,0] - toreplace[:,1])
                replacement = np.c_[r1, r2]
                state_ncb[:, indices] = replacement 
        else:
            assert True, 'StateSpace.chgeComputBasis change of Basis not yet implemented'
        return state_ncb




    

#==============================================================================
#                   FloquetStateSpace
#
#============================================================================== 
class FloquetStateSpace(StateSpace): 
    """
    Same thing but for Extended Floquet Space (inherits from the SS)
    (EXTRA) METHODS:
        - Representation of the operators
        -

    """
    def __init__(self, n_mot, n_int, params): 
        StateSpace.__init__(self, n_mot, n_int, params)
        self.omega = params.omega
        self.n_trunc = params.n_trunc
        self.genRepresentationFSS()
        
    def changeNTrunc(self,newN):
        self.n_trunc = newN
        self.genRepresentationFSS()
    
    def changeNMot(self, newN):
        #StateSpace.changeNMot(new)
        StateSpace.changeNMot(self, newN)
        #super(FloquetStateSpace, self).changeNMot(new)
        self.genRepresentationFSS()
        
        
    def genRepresentationFSS(self):        
        self.n_blocks = 2 * self.n_trunc +1
        self.n_floquet = self.n_blocks * self.n_hilbert
        self.slice_floquet_harmo = [slice(i*self.n_hilbert, (i+1)*self.n_hilbert,1) for i in range(self.n_blocks)]
        if(self.convention == 1):
            self.slice_floquet_mot = [slice(i, self.n_hilbert * self.n_blocks, self.n_mot) for i in range(self.n_mot)]
            #self.slice_floquet_int = [slice(i * self.n_mot, (i+1) * self.n_mot, 1)  for i in range(self.n_int)]
        elif(self.convention == 2):
            self.slice_floquet_int = [slice(i, self.n_hilbert*self.n_blocks, self.n_int) for i in range(self.n_int)]
            #self.slice_mot = [slice(i * self.n_int, (i+1) * self.n_int, 1)  for i in range(self.n_mot)]          
        #self.mask_harmo = [slice(i*self.n_hilbert, (i+1)*self.n_hilbert,1) for i in range(n_blocks)]   
    
       
    def ipFloquet(self, ket1, ket2):
        scalar_blocks = [np.dot(ket1[self.slice_harmo[i]].conj(), ket2[self.slice_harmo[i]])for i in range(self.n_blocks)]
        return np.sum(scalar_blocks)


    
#==============================================================================
# Some testing
#==============================================================================
if __name__ == '__main__':
    #import matplotlib.pylab as plt
    
# --------------------------------------------------------------------------- #
#   1 qbit object
# --------------------------------------------------------------------------- #        
    oneQ = OneQbit()            
    oneQ.GetPauliByName('X')
    oneQ.GetPauliByName('Y')
    phiPlus = oneQ.phiPlus
    phiMinus = oneQ.phiMinus
    FiniteHilbertSpace.measurement(phiPlus, 10, phiMinus)
    FiniteHilbertSpace.measurement(phiPlus, 10)


# --------------------------------------------------------------------------- #
#   2 qbit object
# --------------------------------------------------------------------------- #            
    twoQ = TwoQbits()            
    print(twoQ._X[0])
    print(twoQ._X[1])
    print(twoQ._Y[0])
    
    print(np.dot(twoQ._X[0], twoQ._X[1]))
    print(np.dot(twoQ._X[0], twoQ._X[0]))
    print(twoQ.commut(twoQ._X[0], twoQ._X[0]))
    

    ket1 = twoQ.phi0
    ket2 = twoQ.phi1
    ket3 = twoQ.phiMinus
    ket4 = twoQ.phiPlus

    twoQ.ip(ket1, ket1) 
    twoQ.norm(ket2)
    twoQ.ip(ket1, ket1 * np.exp(0.3j))    
    twoQ.fidelity(ket1, ket3)
# --------------------------------------------------------------------------- #
#   Linear algebra
# --------------------------------------------------------------------------- #      
    L3 = QuDits(3)
    L3._X
    L3.IsHermitian(L3._X)
    L3.IsUnitary(L3.Exp(L3._X, +1.0j))
    L3.ket(0,2)
    
    twoQ.ip(np.random.sample([5,2]), np.random.sample([5,2]))