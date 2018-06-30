#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:18 2018

@author: fred
"""
if(__name__ == '__main__'):
    import sys
    sys.path.insert(0, '../')
    import Helper as ut
else:
    from .. import Helper as ut

from numpy import array
import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pylab as plt
import pdb 

# TODO: Exponential
# TODO: Gradient
# TODO: Implement Bounded / SaS as class decorator
# TODO: Scaling sin(Pi T)
# TODO: n-D I/O 
# TODO: better management of effective params / update etc..
#TODO: repr / buildFromRep

#==============================================================================
#                   3 Behaviors (to be subclassed to inherit their methods)
# ShiftedAndScaled
# Bounded
# 
# TODO: re-think the way they are implemented.. maybe use composition .. wabout the gradients
#==============================================================================
class ShiftedAndScaled():
    """ Shift and scale a function according to certain 'constraints' and 'type'
    + 1 constraint (e.g. constarints = [[t0,y0]])
        Starting with a function f create g s.t. g(t0) = y0 with
        
        + g(t) = f(t) + (y0 - f(t0)) for typeSS = 'shift'
        + g(t) = f(t) * (y0 / f(t0)) for typeSS = 'scale'
        
    + 1 constraint (e.g. constarints = [[t0,y0]])
        - typeSS = 'scale'
        - typeSS = 'sin'
        
    + For 1 endpoint:
        - typeSS = 'shift'
        - typeSS = 'scale'
        
    Meant to be inherited in concrete ParametrizedFunctions or collections
    
    Example:
        ShiftedAndScaled([[0,0], [1,5]])
        ShiftedAndScaled([[0,0]], 'shift')
        ShiftedAndScaled([[2,3]]) 
    """
    def __init__(self, constraints = None, typeSS = 'scale'):
        self.__SetSaS(constraints, typeSS)

    def __SetSaS(self, constraints = None, typeSS = 'scale'):
        self._typeSS = typeSS
        self._constraints = constraints
        self.UpdateSaS(self._constraints)

    def UpdateSaS(self, ### = None, typeSS = None):
        if (constraints is None):
            constraints = self._constraints
        if (typeSS is None):
            typeSS = self._typeSS 
            
        self._scale = 1
        self._shift = 0
            
        if (constraints is not None):
            if not(ut.is_iter(constraints[0])):
                constraints = [constraints]
                
            if(len(constraints) == 2):
                x0, y0 = constraints[0]
                x1, y1 = constraints[1]
                f0, f1 = self.__call__(x0), self.__call__(x1)
                if(typeSS == 'sin'):
                    # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                    assert (y0 == y1), "invalid constraints"
                    self._shift = y0
                    self._scale = lambda t: np.sin(np.pi / (x1 - x0) * (t - x0))
                elif(typeSS == 'sin2'):
                    # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                    assert (y0 == y1), "invalid constraints"
                    self._shift = y0
                    self._scale = lambda t: np.square(np.sin(np.pi * (x1 - x0) * (t - x0)))
                
                else:
                    # f'(t) = shift + scale * f(t)
                    if ((f1-f0)!=0):
                        self._scale = (y1-y0) / (f1-f0)
                    else:
                        print('scale fixed to 1')
                        self._scale=1
                    self._shift = y0 - f0 * self._scale        
            elif(typeSS == 'scale'):
                x0, y0 = constraints[0]
                f0 = self.__call__(x0)

                if (f0 == 0):
                    print('scale fixed to 1')
                    self._scale = 1 
                else:
                    self._scale = y0/f0
            
            elif(typeSS == 'shift'):
                x0, y0 = constraints[0]
                f0 = self.__call__(x0)
                self._shift = y0 - f0
                
    @ut.vectorise_method
    def ShiftAndScale(self, value, t = None):
        if(self._scale is None):
            scale = 1
        elif(hasattr(self._scale, '__call__')):
            scale = self._scale(t)
            
        else:
            scale = self._scale
            
        if(self._shift is None):
            shift = 0
        else:
            shift = self._shift
        
        return value * scale + shift
   
         
class Bounded():
    """
    Purpose:
        impose boundaries on the output of a function
    Example
    
    """
    def __init__(self, boundaries = None):
        self.__SetupBoundaries(boundaries) #To ensure f(0) = self.norm

    def __SetupBoundaries(self, boundaries = None):
        self._boundaries = boundaries
    
    def UpdateBoundaries(self, boundaries = None):
        self.__SetupBoundaries(boundaries)
    
    @ut.vectorise_method
    def Bound(self, value):
        if ((hasattr(self, '_boundaries')) and (self._boundaries is not None)):
            if value < self._boundaries[0]:
                res = self._boundaries[0]
            elif value > self._boundaries[1]:
                res = self._boundaries[1]
            else:
                res = value
        else:
            res = value
            
        return res

class Overwrite():
    """
    Purpose:
        Overwrite 
    
    """
    def __init__(self, listX = None, listY=None):
        self._listX = None
        self._listY = None
        self._list_is_interval = None
        self._nbOWPoints = 0
        self.__SetupOW(listX, listY) #To ensure f(0) = self.norm

    def __SetupOW(self, listX, listY):
        if(listX is not None):
            assert len(listX) == len(listY), "listX and listY should have the same length"
            self._listX = listX
            self._listY = listY
            self._list_is_interval = [isinstance(x, tuple) for x in self._listX]
            self._nbPoints = len(listX)
        
    def UpdateOW(self, listX, listY):
        self.__SetupOW(listX, listY)
    
    @ut.vectorise_method
    def OW(self, x):
        res = [False, None]
        for i in self._nbOWPoints:
            if(self._list_is_interval[i]):
                check = (x>= self._listX[i][0]) * (x<= self._listX[i][1]) 
            else:
                check = (x == self._listX[i])

            if(check):
                res = [True, self._listY[i]]
                break
            
        return res

#==============================================================================
#                   ParametrizedFunctionFactory
# 
# Abstract class for parametrized functions
# Implement Bounded and ShiftAndScaled behaviors
#TODO: implement __add__ + - () ** etc.. based on collection of functions
#==============================================================================
class ParametrizedFunctionFactory(ShiftedAndScaled, Bounded):
    """
    Purpose:
        Abstract class for parametrized functions
        
    To be implemented in the child classes:
        + _ProcessInitParams >> process the parameters for initialization of the function
        + _ProcessUpdateParams >> process the parameters for the update of the parameters
        + _CustomUpdate >> extra specific procedure to run when updating the parameters
        + Gradient >> return the gradient of the function wrt to parameters
        + _Evaluate(t) >> Evaluate the functions for the input t
    Example
    
    """
    LIST_NAME_PARAMS = []
    def __init__(self, params = None, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        self._params = None
        self._nbParams = None
        self._nbTotalParams = None
        self._fixedParams = fixedParams
        self.SetUpParams(params)
        ShiftedAndScaled.__init__(self, constraints, typeSS)
        Bounded.__init__(self, boundaries)
        if(fixedParams):
            self._nbTotalParams = 0

    @ut.vectorise_method
    def __call__(self, variable):
        """ Call the evaluation of the function. Then Scale, shift and bound it 
        if needed
        """
        res = self._Evaluate(variable)
        res = self.ShiftAndScale(res, t = variable)
        res = self.Bound(res)
        return res

#-----------------------------------------------------------------------------#
# Parameters management
#-----------------------------------------------------------------------------#            
    def SetUpParams(self, params):
        self._params = self._ProcessInitParams(params)
        self._nbParams = len(self._params)
        
    def UpdateParams(self, params, **args):
        """
        Purpose:    
            Update paramaters dictionary (_params) with new parameters (params).
            params (its keys) should be a subset of _params if not it will fail
        Extra:
            Extra custom update procedures of the parameters can be implemented in 
            _ProcessUpdateParams (for preprocessing of the parameters, e.g. dealing
            with the case params is an array) and _CustomUpdate (to reflect updates
            to other part of the object)
        """
        if(self._nbTotalParams>0):
            params = self._ProcessUpdateParams(params)
            self._params = ut.merge_dico(self._params, params, update_type = -1)
            self._CustomUpdate(params)
            self.UpdateSaS()
            
    def GetParams(self, name):
        return self._params[name]
        
    def FixParams(self):
        """ To make the parameters un-updatable
        Not fully implemented
        """
        self._nbTotalParams = 0
        self._fixedParams = True
        
    def PrintParams(self, indent_str = ""):
        print(indent_str + str(self.__class__))
        print(self._params)
        print("Shift, scale, and bounds = " + str(self._shift) + ", " + str(self._scale) + ", " + str(self._boundaries)) 



#-----------------------------------------------------------------------------#
# Some extra capabilities
#-----------------------------------------------------------------------------#
    def PlotFunction(self, range_t):
        y = [self.__call__(t) for t in range_t]
        plt.plot(np.array(range_t), np.array(y))
    
    def Fluence(self, range_t, normalized = True):
        """ Fluence by default is defined as 
        fluence = sum f(t)**2 dt (/ T if normalized)
        """
        time_step = np.diff(range_t)
        val_square = np.square(self(range_t))
        res = np.sum(np.array(val_square[:-1] * time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res
    
    def Smoothness(self, range_t, normalized = True):
        """ Smoothness is defined as default as
        smoothness = avg (f(t) - f(t + dt)**2 / dt
        """
        time_step = np.array(range_t[1:]) - np.array(range_t[:-1])
        val_square = np.array([self(t)**2 for t in range_t[:-1]])
        res = np.sum(np.array(val_square * time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res

    def L2Norm(self, otherFunc, range_t, normalization = 1.0):
        """ Compute the distance to another function based on L2Norm
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        time_step = np.diff(range_t)
        diff_square = np.square(self(range_t) - otherFunc(range_t))
        T = (range_t[-2] - range_t[0])
        res = np.sum(np.array(diff_square * time_step[:-1])) / T
        return res




#-----------------------------------------------------------------------------#
# IO
#-----------------------------------------------------------------------------#
    def Fun2Dico(self):
        """ Output a dico which should allow to reconstruct the function
        """
        dico = {"name_func":self.__class__.__name__}
        dico['params'] = self._params
        dico['constraints'] = self._constraints
        dico['typeSS'] = self._typeSS
        dico['boundaries'] = self._boundaries
        dico['fixedParams'] =self._fixedParams
        
        return dico

    def __repr__(self):
        """ Output a text file which should allow to reconstruct the function
        """
        name_func = self.__class__.__name__
        str_params = str(self._params)
        str_final = name_func + "(params=" + str_params   
        
        if(self._constraints is not None):
            str_final += ",constraints="
            str_final += str(self._constraints)
            str_final += ",typeSS='"
            str_final += str(self._typeSS)
            str_final += "'"
            
        if(self._boundaries is not None):
            str_final += ",boundaries="
            str_final += str(self._boundaries)

        if(self._fixedParams):
            str_final += ", fixedParams=True"            
        str_final += ")"
        return str_final
        
    
    def CloneFunc(self):
        return eval(repr(self))

    @classmethod
    def Repr2Fun(self, representation):
        """ Build a function from a text file/ dico
        """
        if ut.is_str(representation):
            func = eval(representation)
            
        elif ut.is_dico(representation):
            name_func = representation.pop('name_func')
            func = eval(name_func)(**representation)
            
        else:
            raise NotImplementedError()
    
        return func

    @staticmethod
    def l2_norm_list(listFunc, range_t, funcRef = None):
        """ Compute the L2 distance of a list of functions to the first one
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        assert(ut.is_iter(listFunc)), "functions should be provided as a list"
        dt = np.diff(range_t)
        T = (range_t[-2] - range_t[0])
        t_eff = range_t[:-1]
        if(funcRef is None):
            funcRef = listFunc[0]
        ref = funcRef(t_eff)
        l2norm = [np.sum(np.square(f(t_eff) - ref) * dt) / T for f in listFunc] 
        return l2norm

# --------------------------------------------------------------------------- #
#   To be implemented in the subclasses
# --------------------------------------------------------------------------- #                        
    def Gradient(self, variable = None, **args):
        raise NotImplementedError()
    
    def _ProcessInitParams(self, params):
        raise NotImplementedError()

    def _CustomUpdate(self, params):
        raise NotImplementedError()

    def _Evaluate(self, variable):
        raise NotImplementedError()

    def _ProcessUpdateParams(self, params):
        raise NotImplementedError()
    

#==============================================================================
#                   ParametrizedFunctions
# 
# StepFunc         
# FourierFunc
# LinearFunc
# ConstantFunc
# ChebyshevFun
#TODO: bounded scale ... as functions sclae probably split up in different function sin
#============================================================================== 
# --------------------------------------------------------------------------- #
#   Step Function
# --------------------------------------------------------------------------- #
class StepFunc(ParametrizedFunctionFactory):
    """
        params = {'F' = [f_1, .., f_N], 'Tstep' = [T_1, .., T_N], t0 = t0}
        f(t) = f_n with t in [t_(n-1), t_n[
        f(t) = F_0
    """        
    NAME_FUNC = "StepFunc"
    LIST_NAME_PARAMS = ['F', 'F0']
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        ParametrizedFunctionFactory.__init__(self, params, constraints, typeSS, boundaries, fixedParams)

    def _ProcessInitParams(self, params):
        """
        Purpose:
            Create the parameters needed for the function based on the dictionary params. 
            2 behaviors implemented:
                + if 'F' has been omitted A = [0,0,0...,0]
        """
        if (isinstance(params, dict)):
            F = params.get('F')
            F0 = params.get('F0', 0)
            T = params.get('Tstep')
            
            if(F is not None):
                T, F = ut.sortAccordingToFirst(T, F)                
                self._nbSteps = len(F)
            else:
                T = ut.sortAccordingToFirst(T)
                self._nbSteps = len(T)
                F = np.zeros_like(T)
            params = {'Tstep':T, 'F':np.array(F), 'F0':F0}
        else:
            raise NotImplementedError()    
        self._nbTotalParams = len(F) + 1
        return params            

    def _Evaluate(self, t):
        """
        Purpose:
            if t < T_{N+1} but t >= t_{N} returns F_{N}
            if t < T_0 returns F0
        """
        idx_t = self.GetParams('Tstep').searchsorted(t - 10e-10) - 1
        
        if (idx_t < 0):
            res = self.GetParams('F0')
        else:
            res = self.GetParams('F')[idx_t]
        return res


    def Gradient(self, t = None, **args):
        """
        Purpose:
             Compute gradient for all/ subset of parameters             
        """            
        raise NotImplementedError()

    def _ProcessUpdateParams(self, params):
        """
        if the params to be updated is a dict filter it and pass it
        if not assume it's an iterable and that it should be used to update F
        """
        if(isinstance(params, dict)):
            dico_params = ut.filter_dico(params, self._eff_params_list)
        else:
            assert len(params) == self._nbSteps, " params don't have the right size"
            dico_params = {'F':np.array(params)}            
        return dico_params
        
    def _CustomUpdate(self, params):
        pass
    
    def Smoothness(self, array_t = None):
        # use of lag1 autocoorelation??
        raise NotImplementedError()


# --------------------------------------------------------------------------- #
#   1D Gaussian
# --------------------------------------------------------------------------- #        
class SquareExponential(ParametrizedFunctionFactory):
    """
        [sigma, mu, l] -> sigma^2 exp^{(x-mu)^2/l^2}
        TODO: Implement the multiDim + Kernel version // in eval and assert
    """
    LIST_NAME_PARAMS = ['sigma', 'mu', 'l'] 
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        ParametrizedFunctionFactory.__init__(self, params, constraints, typeSS, boundaries, fixedParams)

    def  _Evaluate(self, t):        
        res =  np.exp(-np.square((t-self._params['mu'])/self._params['l']))
        res *= np.square(self._params['sigma'])
        return res
        
    def _ProcessInitParams(self, params):
        if(isinstance(params, dict)):
            sigma = params.get('sigma', 1)
            mu = params.get('mu', 0)
            l =  params.get('l', 1)
            
        elif ut.is_iter(params):
            sigma = params[0]
            mu = params[1]
            l = params[2]
            
        else:
            raise NotImplementedError
        
        assert (l > 0), "l has to be > 0"
        params = {'sigma': sigma, 'mu': mu, 'l':l}
        self._nbTotalParams = 3
        return params


    def Gradient(self, t = None, **args):
        if(t is None):
            raise NotImplementedError()
        else:
            sigma = self.GetParams['sigma']
            mu = self.GetParams['mu']
            l = self.GetParams['l']
            eval_x = self._Evaluate(t)
            
            sigma_grad = eval_x * 2 / sigma
            mu_grad = eval_x * 2 * (x - mu) / np.square(l)
            l_grad = eval_x * (-2) * np.square(x - mu) /np.power(l, 3)
            res = {'sigma': sigma_grad, 'mu':mu_grad, 'l':l_grad}
        return res

    def _ProcessUpdateParams(self, params):
        """
        Deal with the case where update params are an array
        1st element assigned to 'a' (slope), 2nd (if provided) to 'b' (i.e. y(0))
        """
        if(not(isinstance(params, dict))):
            assert (len(params) <= 3), 'pb size params'
            if(len(params) == 1):
                params = {'sigma':params[0]}
            elif(len(params) == 2):
                params = {'sigma':params[0], 'mu': params[1]}
            else:
                params = {'sigma':params[0], 'mu': params[1], 'l': params[2]}
        return params
    
    def _CustomUpdate(self, params):
        pass


# --------------------------------------------------------------------------- #
#   Fourrier Function
# --------------------------------------------------------------------------- #        
class FourierFunc(ParametrizedFunctionFactory):
    """
    params = {'A' = [a_1, .., a_N], 'B' = [b_1, .., b_N], 'om' = [om_0, 
                      .., om_N], 'Phi' = [phi_0, phi_N], 'c0' = c0)
        
    >> f(t) = Norm * [c + Sum(0..N) {a_i*cos(om_i t + phi_i) + b_i*cos(om_i t + phi_i)}]
    
    """
    LIST_NAME_PARAMS = ['A', 'B', 'om', 'Phi', 'c0']
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        ParametrizedFunctionFactory.__init__(self, params, constraints, typeSS, boundaries, fixedParams)

    def _ProcessInitParams(self, params):
        """
        Purpose:
            Create the parameters needed for the function based on the dictionary params. 
            2 behaviors implemented:
                + if 'nbH' is an entry in params >> init with nbH harmonics set-up as zero 
                + else need to provide Harmonics value i.e. A, B        _PreProcessInitParams
        """
        #pdb.set_trace()
        if ut.is_dico(params):
            nbH = params.get('nbH')
            if(nbH is not None):
                self._nbHarmonics = nbH
                A = np.zeros(nbH)
                B = np.zeros(nbH)
            else:
                A = np.array(params.get('A'))
                B = np.array(params.get('B'))
                self._nbHarmonics = len(A)
                assert(len(A) == len(B)), 'Problem A/B'
            
            om = params.get('om', 1)
            Phi = params.get('Phi', 0)
            c0 = params.get('c0', 0)
            
            if not(ut.is_iter(om)):
                om = np.arange(1, self._nbHarmonics+1) * om            
            if not(ut.is_iter(Phi)):
                Phi = np.repeat(Phi, self._nbHarmonics)    
            params = {'A':A, 'B':B, 'om':om, 'Phi':Phi, 'c0':c0}

        else:
            raise NotImplementedError()
        self._nbTotalParams = len(A) + len(B) + len(Phi) + len(om) + 1   
        return params            


    def _Evaluate(self, t):
        """
        Purpose:
             compute f(t) = c0 + Sum(1..N) {a_i * cos(om_i t + phi_i) + 
                                 b_i*cos(om_i t + phi_i)}]
        """
        A, B, om, Phi, c0 = (self._params[n] for n in self.LIST_NAME_PARAMS)
        res = [A[i] * np.cos(om[i] * t + Phi[i]) + B[i] * np.sin(om[i] * t + Phi[i]) for i in range(self._nbHarmonics)]        
        res = (c0 + np.sum(res)) 
        return res


    def Gradient(self, t = None, **args):
        """
        Purpose:
             Compute gradient for all/ subset of parameters             
        """
        type_gradient = args.get['type']
        
        if(type_gradient is None):
            raise NotImplementedError()
        elif(type_gradient == 'om'):            
            raise NotImplementedError()
        elif(type_gradient == 'c0'):
            raise NotImplementedError()            
        else:
            raise NotImplementedError()
            
        raise NotImplementedError()
        

    def _ProcessUpdateParams(self, params):
        """
        if the params to be updated is a dict filter it and pass it
        if not assume it's an iterable and that it should be used to update F
        """
        if ut.is_dico(params):
            for k, v in params.items():
                assert (len(v) == len(self.GetParams(k))), " params don't have the right size"
        else:
            raise NotImplementedError()                     
        return params
    
    def _CustomUpdate(self, params):
        pass


# --------------------------------------------------------------------------- #
#   Linear Function
# --------------------------------------------------------------------------- #
class LinearFunc(ParametrizedFunctionFactory):
    """
        [a, b] -> f(t) = at + b
    """
    LIST_NAME_PARAMS = ['a', 'b'] 
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        ParametrizedFunctionFactory.__init__(self, params, constraints, typeSS, boundaries, fixedParams)

    def _Evaluate(self, t):        
        res = self._params['a'] * t + self._params['b']
        return res
        
    def _ProcessInitParams(self, params):
        if(isinstance(params, dict)):
            a = params.get('a', 0)
            b = params.get('b', 0)
        elif ut.is_iter(params):
            a = params[0]
            b = params[1]
        else:
            raise NotImplementedError
        params = {'a': a, 'b': b}
        self._nbTotalParams = 2
        return params


    def Gradient(self, t = None, **args):
        if(t is None):
            res = {'a':LinearFunc({'a':1, 'b':0}), 'b':ConstantFunc(1)}
        else:
            res = {'a': t, 'b':1}
        return res

    def _ProcessUpdateParams(self, params):
        """
        Deal with the case where update params are an array
        1st element assigned to 'a' (slope), 2nd (if provided) to 'b' (i.e. y(0))
        """
        if(not(ut.is_dico(params))):
            {self.LIST_NAME_PARAMS[i]:params[i] for i in range(len(params))}
        return params
    
    def _CustomUpdate(self, params):
        pass
    
    
class ConstantFunc(LinearFunc):
    """
        [a, b] -> f(t) = at + b
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        if(ut.is_dico(params)):
            b = params.get('b')
            LinearFunc.__init__(self, {'a': 0, 'b':b}, constraints, typeSS, boundaries, fixedParams)
        else:
            LinearFunc.__init__(self, {'a': 0, 'b':params}, constraints, typeSS, boundaries, fixedParams)

    
    
# --------------------------------------------------------------------------- #
#   Chebichev pa5rametrization
# --------------------------------------------------------------------------- #    
class ChebyshevFun(ParametrizedFunctionFactory):
    """
    Chebyshev functions
        params of the chebyshev representation:
            params = {'c0', 'C' = [c_1, .., c_N], domain, window)

        Resulting function
            f(t) = [c0 + Sum(1..N) {C_i* T_i(t)]
            where T_i is the Chebyshev polynomial of order i defined using domain and window
    
    """
    LIST_NAME_PARAMS = ['C', 'c0', 'domain', 'window']
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None, fixedParams = False):
        ParametrizedFunctionFactory.__init__(self, params, constraints, typeSS, boundaries, fixedParams)

    def _Evaluate(self, t):
        """
        Purpose:
             compute f(t) = norm * [ c0 + Sum(1..N) + {c_i * Cheb(i, t)} ]
        """
        res = self._chebFunc(t)
        return res

    def _ProcessInitParams(self, params):
        """
            Process the parameters dictionary provided for creating the object
            and build accordingly a chebichev function using numpy.polynomial.chebyshev
            
            Rules:
                if c0 not provided take 0 /if domain not provided take [-1,1] /
                if window not provided take [-1,1]
                
        """
        params = self._PreProcessInitParams(params)
        nbH = params.get('nbH')
        if (nbH is not None):
            C = np.zeros(nbH)
            self._nbHarmonics = nbH # Number of coefficients excluding c0
        else:
            C = np.array(params.get('C'))
            self._nbHarmonics = len(C)
            
        c0 = params.get('c0', 0)
        domain = params.get('domain', [-1, 1])
        window = params.get('window', [-1,1])
        params = {'C':C, 'c0':c0, 'domain':domain, 'window':window}
        coeffs = np.concatenate(([c0], C))
        self._chebFunc = cheb.Chebyshev(coeffs, domain, window)
        self._nbTotalParams = len(C) + 1
        return params            

    def _CustomUpdate(self, params):
        """
        Purpose:
            Called when parameters are updated (through UpdateParams)
        """
        self._UpdateChebyFunction(params)
    
    def _UpdateChebyFunction(self, params):
        """
        Purpose:
            Update underlying function coefficients
        """
        params_keys = params.keys()
        if("c0" in params_keys):
            self._chebFunc.coef[0] = params.get('c0')
        if("C" in params_keys):
            self._chebFunc.coef[1:] = params.get('C')
        if("domain" in params_keys):
            self._chebFunc.domain = params.get('domain')
        if("window" in params_keys):
            self._chebFunc.window = params.get('window')

    def Gradient(self, t = None, constant = False, **args):
        raise NotImplementedError()

    def _PreProcessInitParams(self, params):
        """
        Purpose:
            Can be implemented in the child classes to allow for more custom 
            rules to be implemented
        """
        return params


#==============================================================================
#                   CollectionParametrizedFunctionFactory
# 
# Collection of ParametrizedFunction i.e Plus(f1,f2,..) Times(f1, f2,...)
#==============================================================================            
# --------------------------------------------------------------------------- #
#   Abstract class
# --------------------------------------------------------------------------- #
#TODO: a lot
class CollectionParametrizedFunctionFactory(ShiftedAndScaled,Bounded):
    """
    Purpose:
        Just an abstract collection (list) of Parametrized function which can be
        used to represent any specific functions operations: plus/times/composition 
        Can create (and deal with) nested collections e.g. Plus(f1, Times(f2, f3))
        
    UpdateParams:
        +Two indices indexFunction (which function to update)
        + if indexFunction is None update all the functions are updated 
    
    AddNewFunctions / ReplaceFunction / RemoveFunction
        Manipulate the collection (or sub-collections)
    
    Calls:
         Implemented in the subclass
        
    """
    def __init__(self, func = None, constraints = None, typeSS = 'scale', boundaries = None):
        self.listFun = func
        self._nbFunc = len(func)
        self._listNbParams = [f._nbParams for f in self.listFun]
        self._listNbTotalParams = [f._nbTotalParams for f in self.listFun]
        self._nbParams = np.sum(self._listNbParams)
        self._nbTotalParams = np.sum(self._listNbTotalParams)
        ShiftedAndScaled.__init__(self, constraints, typeSS)
        Bounded.__init__(self, boundaries)

    def __call__(self, variable):
        raise NotImplementedError()
        
    def Gradient(self, variable = None):
        """ Return gradient of each functions in the nested collections
        Remarks: Not used/can(will) evolve
        """
        res = [f.Gradient(variable) for f in self.listFun]
        return res       
    
    # ----------------------------------------------------------------------- #
    #   Management of the parameters / functions
    # ----------------------------------------------------------------------- #
    def UpdateParams(self, params, indexFun = None, updateSaS = True, **args):
        """
        Purpose:
            Update the parameters of (one or several) underlying function(s):
        Input:
            + indexFun: which function to update - None: all functions are updated /
            int: jth function of the collection is updated / list: e.g. [i, j] 
            update the J-TH function of the I-TH function of the collection (to
            deal with nested collections)
            + updateSaS: if True recompute the scaling and shifting parameters
            
        """
        
        if(indexFun is None):
            if(ut.is_dico(params)):            
                for i in range(len(self.listFun)):    
                    self.listFun[i].UpdateParams(params, indexFun = indexFun, **args)
            else:
                i_params = 0
                for i in range(len(self.listFun)):
                    nb_to_update = self.listFun[i]._nbTotalParams
                    if(nb_to_update>0):
                        self.listFun[i].UpdateParams(params[i_params: i_params+nb_to_update], **args)
                        i_params += nb_to_update
                        
        elif(isinstance(indexFun, int)):
            self.listFun[indexFun].UpdateParams(params, **args)
        
        elif(ut.is_iter(indexFun)):
            # Waterfall behavior
            firstIndex = indexFun[0]
            if (len(indexFun) == 1):
                self.listFun[firstIndex].UpdateParams(params , indexFun = None,**args)                
            else:                
                restIndex = indexFun[1:]
                self.listFun[firstIndex].UpdateParams(params, indexFun = restIndex ,**args)
        else:
            raise NotImplementedError()
            
        if(updateSaS):
            self.UpdateSaS()
        
            
    def AddNewFunctions(self, listFun, indexFun = None, order = 0, updateSaS = True):
        """
        Purpose:
            Add new function(s) to the collection at 
        Parameters:
            + listFun a list of functions (can deal with a simple function)
            + indexFun where to add the new function(s) 
            + order if 0 (1) put it a the beginning (end) of the collection
            + updateSaS: if True recompute the scaling and shifting parameters
        """
        if not(ut.is_iter(listFun)):
            listFun = [listFun]

        if(indexFun is None):
            if(order== 0):
                self.listFun = listFun + self.listFun
            elif(order== 1):
                self.listFun = self.listFun + listFun
            
            self._nbFunc += len(self.listFun)
        
        elif (isinstance(indexFun, int)):
            self.listFun[indexFun].AddNewFunctions(listFun, None, order, updateSaS)
        
        elif (ut.is_iter(indexFun)):
            firstIndex = indexFun[0]
            if (len(indexFun) == 1):
                self.listFun[firstIndex].AddNewFunctions(listFun, None, order, updateSaS)
            else:                
                restIndex = indexFun[1:]
                self.listFun[firstIndex].AddNewFunctions(listFun, restIndex, order, updateSaS)
        else:
            raise NotImplementedError()
            
        if(updateSaS):
            self.UpdateSaS()

    def ReplaceFunction(self, func, indexFun = None, updateSaS = True):
        """
        Purpose:
            Replace a function  at a specific place in the (nested) collection
        Parameters:
            (cf. AddNewFunctions)
        """
        if(isinstance(indexFun, int)):
            self.listFun[indexFun] = func
        
        elif (ut.is_iter(indexFun)):
            firstIndex = indexFun[0]
            if (len(indexFun) == 1):
                self.listFun[firstIndex] = func
            else:                
                restIndex = indexFun[1:]
                self.listFun[firstIndex].ReplaceFunction(func, restIndex, updateSaS)
        else:
            raise NotImplementedError()
            
        if(updateSaS):
            self.UpdateSaS()

        
    def RemoveFunction(self, indexFun = 0, updateSaS = True):
        """
        Purpose:
            Remove functions from the (nested collections)
        """
        if(isinstance(indexFun, int)):
            self.listFun.pop(indexFun)
        
        elif (ut.is_iter(indexFun)):
            if (len(indexFun) == 1):
                self.listFun.pop(indexFun[0])
            else:
                firstIndex = indexFun[0]
                restIndex = indexFun[1:]
                self.listFun[firstIndex].RemoveFunction(restIndex, updateSaS)
        else:
            raise NotImplementedError()
            
        if(updateSaS):
            self.UpdateSaS()
            
    def GetFunction(self, indexFun = 0):
        """
        Purpose:
            Remove functions from the (nested collections)
        """
        if(isinstance(indexFun, int)):
            res = self.listFun[indexFun]
        
        elif(ut.is_iter(indexFun)):
            if (len(indexFun) == 1):
                res = self.listFun[indexFun[0]]
            else:
                firstIndex = indexFun[0]
                restIndex = indexFun[1:]
                res = self.listFun[firstIndex].GetFunction(restIndex)
        return res
    
    def PrintParams(self, indent_str = ""):
        print(indent_str+ str(self.__class__))
        print("Shift, scale, and bounds = " + str(self._shift) + ", " + str(self._scale) + ", " + str(self._boundaries)) 
        for f in self.listFun:
            f.PrintParams(indent_str = indent_str + "#")    

    # ----------------------------------------------------------------------- #
    #   Extra capabilities (exactly the same as in ParametrizedFunctionFactory)
    # ----------------------------------------------------------------------- #        
    def PlotFunction(self, range_t):
        y = [self.__call__(t) for t in range_t]
        plt.plot(np.array(range_t), np.array(y))
    
    def Fluence(self, range_t, normalized = True):
        """ Fluence by default is defined as 
        fluence = sum f(t)**2 dt (/ T if normalized)
        """
        time_step = np.diff(range_t)
        val_square = np.square(self(range_t))
        res = np.sum(np.array(val_square[:-1] * time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res
    
    def Smoothness(self, range_t, normalized = True):
        """ Smoothness is defined as default as
        smoothness = avg (f(t) - f(t + dt)**2 / dt
        """
        time_step = np.array(range_t[1:]) - np.array(range_t[:-1])
        val_square = np.array([self(t)**2 for t in range_t[:-1]])
        res = np.sum(np.array(val_square * time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res

    def L2Norm(self, otherFunc, range_t):
        """ Compute the distance to another function based on L2Norm
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        dt = np.diff(range_t)
        df_square = np.square(self(range_t[:-1]) - otherFunc(range_t[:-1]))
        T = (range_t[-2] - range_t[0])
        res = np.sum(np.array(df_square * dt)) / T
        
        return res
    
    @staticmethod
    def l2_norm_list(listFunc, range_t, funcRef = None):
        """ Compute the L2 distance of a list of functions to the first one
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        assert(ut.is_iter(listFunc)), "functions should be provided as a list"
        dt = np.diff(range_t)
        T = (range_t[-2] - range_t[0])
        t_eff = range_t[:-1]
        if(funcRef is None):
            funcRef = listFunc[0]
        ref = funcRef(t_eff)
        l2norm = [np.sum(np.square(f(t_eff) - ref) * dt) / T for f in listFunc] 
        return l2norm


    # ----------------------------------------------------------------------- #
    #   I/O
    # ----------------------------------------------------------------------- #
    def Fun2Dico(self):
        """ Output a dico which should allow to reconstruct the function
        """
        dico = {"name_func":self.__class__.__name__}
        dico['func'] = self.listFun
        dico['constraints'] = self._constraints
        dico['typeSS'] = self._typeSS
        dico['boundaries'] = self._boundaries
        
        return dico

    def __repr__(self):
        """ Output a text file which should allow to reconstruct the function
        """
        name_func = self.__class__.__name__
        str_list = "list(["
        for fun in self.listFun:
            str_list = str_list + repr(fun) + ","
        str_list = str_list[:-1]
        str_list += "])"
        str_final = name_func + "(func=" + str_list 
        
        if(self._constraints is not None):
            str_final += ",constraints="
            str_final += str(self._constraints)
            str_final += ",typeSS='"
            str_final += str(self._typeSS)
            str_final += "'"
            
        if(self._boundaries is not None):
            str_final += ",boundaries="
            str_final += str(self._boundaries)
        
        str_final += ")"
        return str_final
        
    
    def CloneFunc(self):
        """ Create a clone
        """
        return eval(repr(self))

    @classmethod
    def Repr2Fun(self, representation):
        """ Build a function from a text file/ dico
        """
        if ut.is_str(representation):
            func = eval(representation)
            
        elif ut.is_dico(representation):
            name_func = representation.pop('name_func')
            func = eval(name_func)(**representation)
            
        else:
            raise NotImplementedError()
    
        return func

    
# --------------------------------------------------------------------------- #
#   Implementations
# --------------------------------------------------------------------------- #
class Times(CollectionParametrizedFunctionFactory):
    """
    Implementation of collectionOfFunctions
    [f1, f2](t) >> f1(t) * f2(t)
    Gradient should be implemented
    """
    def __init__(self, func = None, constraints = None, typeSS = 'scale', boundaries = None):
        CollectionParametrizedFunctionFactory.__init__(self, func, constraints, typeSS, boundaries)

    @ut.vectorise_method
    def __call__(self, variable):
        resPerFun = [f(variable) for f in self.listFun]
        res = np.product(resPerFun)
        res = self.ShiftAndScale(res, t = variable)
        res = self.Bound(res)
        return res
        
class Plus(CollectionParametrizedFunctionFactory):        
    """
    Implementation of collectionOfFunctions
    [f1, f2](t) >> f1(t) + f2(t)
    Gradient should be implemented
    """
    def __init__(self, func = None, constraints = None, typeSS = 'scale', boundaries = None):
        CollectionParametrizedFunctionFactory.__init__(self, func, constraints, typeSS, boundaries)

    @ut.vectorise_method
    def __call__(self, variable):
        resPerFun = [f(variable) for f in self.listFun]
        res = np.sum(resPerFun)
        res = self.ShiftAndScale(res, t = variable)
        res = self.Bound(res)
        return res
 
class Composition(CollectionParametrizedFunctionFactory):
    """
    Implementation of collectionOfFunctions
    [f1, f2](t) >> f1(f2(t))
    Gradient should be implemented
    """
    def __init__(self, func = None, constraints = None, typeSS = 'scale', boundaries = None):
        CollectionParametrizedFunctionFactory.__init__(self, func, constraints, typeSS, boundaries)

    @ut.vectorise_method
    def __call__(self, variable):
        res = variable
        for f in self.listFun:
            res = f(res) 
            
        res = self.ShiftAndScale(res, t = variable)
        res = self.Bound(res)
        return res
       
        
#==============================================================================
#                     CUSTOM FUNCTIONS ZOOOO
# Custom implementation of some parametric functions. Updates of the parameters
# has been implemented accordingly, i.e. can be updated with an array of value, 
# only the relevant parameters are updated
#
# cFourierAmplitudesFixedCt   >>   Parameters are only A B
# cFourierAmplitudes          >>   parameters are only A, B and c0
# cSine                       >>   f(t) = sum B_l sin(om_l t) 
# cSineTrend                  >>   f(t) = at + b + sum B_l sin(om_l t) 
# cSineTrendFixedCt           >>   f(t) = at + b + sum B_l sin(om_l t) with b fixed
# cSineFixedTrend             >>   f(t) = at + b + sum B_l sin(om_l t) with a, b fixed
#        
# cChebyshevFixedCt           >>   f(t) = c0 + sum(1..N) C_i*T_i(t) where c0 is fixed
# cChebyshevFixedTrend        >>   f(t) = c0 + sum(1..N) C_i*T_i(t) where c0 and C_1 are fixed
#
# FourierFunc =>> cFourierAmplitudesFixedCt
#             =>> cFourierAmplitudes  
#             =>> cSine
#     (Plus)  =>> cSineTrend                 ==>>  cSineFixedTrend
#                                            ==>>  cSineTrendFixedCt
#
# ChebyshevFun   =>>   cChebyshev   ==>>   cChebyshevFixedCt
#                                   ==>>   cChebyshevOdd  
#                                   ==>>   cChebysheEven                              
#                                   ==>>   cChebyshevEvenFixedCt   =>>   cChebyshevFixedTrend
#             
#==============================================================================    

LIST_CUSTOM_FUNC = {}
LIST_CUSTOM_FUNC['cFourierAmplitudesFixedCt']= ""
LIST_CUSTOM_FUNC['cFourierAmplitudes']= ""
LIST_CUSTOM_FUNC['cSine']= ""
LIST_CUSTOM_FUNC['cSineTrend']= ""
LIST_CUSTOM_FUNC['cSineFixedTrend']= ""
LIST_CUSTOM_FUNC['cSineTrendFixedCt']= ""
LIST_CUSTOM_FUNC['cChebyshev']= ""
LIST_CUSTOM_FUNC['cChebyshevFixedCt']= ""
LIST_CUSTOM_FUNC['cChebyshevOdd']= ""
LIST_CUSTOM_FUNC['cChebyshevEven']= ""
LIST_CUSTOM_FUNC['ccChebyshevEvenFixedCt']= ""
LIST_CUSTOM_FUNC['cChebyshevFixedTrend']= ""

class cFourierAmplitudesFixedCt(FourierFunc):
    """
    Purpose:
        Fourrier Series where only the amplitudes (A, B) are the parameters (not c0)
    Update of Parameters:
        _processUpdateParams is overwritten s.t:
            + the function can be updated with an array of value
            + if a dico of parameters is passed it is filtered and only the 
            amplitudes are updated
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        params = self._PreProcessInitParams(params)
        FourierFunc.__init__(self, params, constraints, typeSS, boundaries)


    def _PreProcessInitParams(self, params):
        """
        Some extra custom behavior used during init:
        Number of parameters if provided (in params['nbP']) is used to assess
        the number of Harmonics ('nbH'). Use of nbH induces the initialization 
        of the Fourier Series with nbH Harmonics setup to 0
        """
        self._eff_params_list = ['A', 'B']
        if('nbP' in params.keys()):
            params['nbH'] = type(self).GetNbH(params['nbP'])
        self._nbTotalParams = 2 * params['nbH']
        return params

    def _ProcessUpdateParams(self, params):
        """
        New behavior: 
            if an array is passed split in two (thus should be even) first bit 
            assigned to A second to B
        """
        if(not(isinstance(params, dict))):
            nbH = self._nbHarmonics
            nbP = len(params)
            assert (nbP == 2 * nbH), 'pb len params'
            dico_params = {'A': params[:nbH], 'B':params[nbH:]}
        else:
            dico_params = ut.filter_dico(params, self._eff_params_list)
        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        assert ut.is_even(nb_params), "pb with nbP"
        nbH = int(nb_params / 2)
        return nbH
                 
class cFourierAmplitudes(FourierFunc):   
    """
    Purpose:
        Fourrier Series where only the amplitudes (A, B, c0) are the parameters 
    Update of Parameters:
        _processUpdateParams is overwritten s.t:
            + the function can be updated with an array of value
            + if a dico of parameters is passed it is filtered and only the 
            amplitudes are updated
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        params = self._PreProcessInitParams(params)
        FourierFunc.__init__(self, params, constraints, typeSS, boundaries)

    def _PreProcessInitParams(self, params):
        """
        Some extra custom behavior used during init:
        Number of parameters if provided (in params['nbP']) is used to assess
        the number of Harmonics ('nbH'). Use of nbH induces the initialization 
        of the Fourier Series with nbH Harmonics setup to 0
        """
        self._eff_params_list = ['A', 'B', 'c0']
        if('nbP' in params.keys()):
            params['nbH'] = type(self).GetNbH(params['nbP'])
        self._nbTotalParams = 2 * params['nbH'] + 1
        return params

    def _ProcessUpdateParams(self, params):
        """
        New behavior: 
            if an array is passed split in three (should be odd) first bit is c0
            second and third assigned to A and B
        """
        if(not(isinstance(params, dict))):
            nbH = self._nbHarmonics
            nbP = len(params)  
            assert (nbP == 2 * nbH + 1), 'pb len params'
            dico_params = {'c0': params[0], 'A': params[1:(nbH+1)], 'B':params[(nbH+1):]}
        else:
            dico_params = ut.filter_dico(params, self._eff_params_list)
        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        assert ut.is_odd(nb_params), "pb with nbP"
        nbH = int((nb_params-1) / 2)
        return nbH
                 

class cSine(FourierFunc):   
    """
    Purpose:
        Sine Series (built using FourierSeries) where (B) are the parameters while
        c0 and A are forced to be 0
        
    Update of Parameters:
        _processUpdateParams is overwritten s.t:
            + the function can be updated with an array of value
            + if a dico of parameters is passed it is filtered and only the 
            amplitudes are updated
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        params = self._PreProcessInitParams(params)
        FourierFunc.__init__(self, params, constraints, typeSS, boundaries)

    def _PreProcessInitParams(self, params):
        """
        Some extra custom behavior used during init:
        Number of parameters if provided (in params['nbP']) is used to assess
        the number of Harmonics ('nbH'). Use of nbH induces the initialization 
        of the Fourier Series with nbH Harmonics setup to 0
        """
        self._eff_params_list = ['B']
        if('nbP' in params.keys()):
            params['nbH'] = type(self).GetNbH(params['nbP'])
        self._nbTotalParams = params['nbH']
        return params


    def _ProcessUpdateParams(self, params):
        """
        New behavior: 
            if an array is passed directly feed to C
            if dictionary passed filter it
        """
        if(not(isinstance(params, dict))):
            nbH = self._nbHarmonics
            nbP = len(params)
            assert (nbP == nbH), 'wrong size'
            dico_params = {'B':params}
        else:
            dico_params = ut.filter_dico(params, self._eff_params_list)
        
        return dico_params            
  
    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params    

class cSineTrend(Plus):
    """
    Purpose:
        Linear trend + Sine Series (built using Plus()) 
        f(t) = at + b + sum_l [b_l sin(om_l * t)]
        a and b and B are parameters        

    Update of Parameters:
        _processUpdateParams is overwritten s.t:
            + the function can be updated with an array of value
            + if a dico of parameters is passed it is filtered and only the 
            amplitudes are updated
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        """
        Purpose:
            Initialization can be done by providing:
            + dico with parameters (they have different names) for the linear 
              trend ('a' and 'b') and sin series ('C', 'om', 'phi')
            + array: in this case a/b = array[0/1] C = array[2:]
        """
        self._eff_params_list = ['B', 'a', 'b']
        trend, sin = self._initComponents(params)
        Plus.__init__(self, [sin, trend], constraints, typeSS, boundaries)
    
    
    def UpdateParams(self, params):
        """
        Purpose:
            Update of the 'B' parameters of the fourier series (1st function
            of the collection) and coeffs of the linear function(2nd functions)
        """
        if(isinstance(params, dict)):
            paramsToUpdate = ut.filter_dico(params, self._eff_params_list)
            Plus.UpdateParams(self, params = paramsToUpdate, indexFun = 0)
        else:
            Plus.UpdateParams(self, params = {'a': params[0], 'b':params[1]}, indexFun = 1)
            Plus.UpdateParams(self, params = {'B': params[2:]}, indexFun = 0)
            
    def _initComponents(self, params):
        if(isinstance(params, dict)):
            params_new = dict(params)
            if('nbP' in params.keys()):    
                params_new['nbH'] = type(self).GetNbH(params['nbP'])
                params_new['a'] = params.get('a', 0)
                params_new['b'] = params.get('b', 0)
                del params_new['nbP']
            trend = LinearFunc(params_new)
            sin = cSine(params_new)
        else:
            raise NotImplementedError()
        return trend, sin 

    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params - 2


class cSineFixedTrend(cSineTrend):
    """
    Purpose:
        Same as cSinePlusTrend except that the linear trend parameters are fixed
        (its reflected in the UpdateParams)
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cSineTrend.__init__(self, params, constraints, typeSS, boundaries)
        self._eff_params_list = ['B']
        self._nbTotalParams = self.GetFunction(0)._nbTotalParams

    def UpdateParams(self, params):
        """
        Purpose:
            Update of only the 'B' parameters of the fourier series
        """
        #pdb.set_trace()
        if(isinstance(params, dict)):
            paramsToUpdate = ut.filter_dico(params, self._eff_params_list)
            Plus.UpdateParams(self, params = paramsToUpdate, indexFun = 0)
        else:
            Plus.UpdateParams(self, params = {'B': params}, indexFun = 0)

    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params


class cSineTrendFixedCt(cSineTrend):
    """
    Purpose:
        Same as cSinePlusTrend except that the slope of thelinear trend 
        is fixed (it's reflected in the UpdateParams)
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cSineTrend.__init__(self, params, constraints, typeSS, boundaries)
        self._eff_params_list = ['B', 'a']
        self._nbTotalParams = self.GetFunction(0)._nbTotalParams + 1
    
    def UpdateParams(self, params):
        """
        Purpose:
            Update of only the 'B' parameters of the fourier series
        """
        if(not(isinstance(params, dict))):
            Plus.UpdateParams(self, params = {'a': params[0]}, indexFun = 1)
            Plus.UpdateParams(self, params = {'B': params[1:]}, indexFun = 0)
        else:
            paramsToUpdate = ut.filter_dico(params, self._eff_params_list)
            cSineTrend.UpdateParams(self, params = paramsToUpdate)          

    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params - 1


class cChebyshev(ChebyshevFun):
    """
    Purpose:
        Chebyshev function
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        ChebyshevFun.__init__(self, params, constraints, typeSS, boundaries)
    
    def _ProcessUpdateParams(self, params):
        """
        Purpose:
            Implement update from an array of parameters
        """
        if(not(isinstance(params, dict))):
            dico_params = {'c0': params[0], 'C': params[1:]}
        else:
            dico_params = ut.filter_dico(params, self._eff_params_list)
        return dico_params

    def _PreProcessInitParams(self, params):
        self._eff_params_list = ['c0', 'C']
        if('nbP' in params.keys()):
            params['nbH'] = type(self).GetNbH(params['nbP'])
        return params

    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params - 1

    
class cChebyshevFixedCt(cChebyshev):
    """
    Purpose:
        Chebyshev with fixed constant value fixed (i.e. not considered as a 
        parameter for the updates)
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cChebyshev.__init__(self, params, constraints, typeSS, boundaries)
        self._eff_params_list = ['C']
        self._nbTotalParams -= 1
        
    
    def _ProcessUpdateParams(self, params):
        """
        Purpose:
            Implement update from an array of parameters
        """
        if(not(isinstance(params, dict))):
            dico_params = {'C': params}
        else:
            dico_params = ut.filter_dico(params, self._eff_params_list)
        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        return nb_params

class cChebyshevOdd(cChebyshev):
    """
    Purpose:
        Chebyshev with evn polynomials fixed (i.e. not considered as a 
        parameter for the updates) - odd polynomials are the Tn(-1) <> Tn(1)
    Rmk:
        Can have even components if defined at initialization (bu can't update them)
    #TODO: Implement init with C_odd
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cChebyshev.__init__(self, params, constraints, typeSS, boundaries)
        self._index_odd = np.arange(0, self._nbHarmonics, 2)
        self._eff_params_list = ['C', 'C_odd']
        self._nbTotalParams = len(self._index_odd)
    
    def _ProcessUpdateParams(self, params):
        """
        Purpose:
            Implement update from an array of parameters
        """
        if(isinstance(params, dict)):
            if (hasattr(params, 'C')):
                dico_params = dict(params)
                C_new = np.copy(self.GetParams('C'))
                C_new[self._index_odd] = params['C'][self._index_odd]
                dico_params['C'] = C_new

            dico_params = ut.filter_dico(params, self._eff_params_list)

        else:            
            C_new = np.copy(self.GetParams('C'))
            C_new[self._index_odd] = params
            dico_params = {'C': C_new}

        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        return 2 * nb_params 
    
    
class cChebyshevEven(cChebyshev):
    """
    Purpose:
        Chebyshev with evn polynomials fixed (i.e. not considered as a 
        parameter for the updates) - even polynomials are the Tn(-1) == Tn(1)
    Rmk:
        + Can have odd components if defined at initialization (bu can't update them)
    #TODO: Implement init with C_even
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cChebyshev.__init__(self, params, constraints, typeSS, boundaries)
        self._index_even = np.arange(1, self._nbHarmonics, 2)
        self._eff_params_list = ['C', 'C_even', 'c0']
        self._nbTotalParams = len(self._index_even) + 1
    
    def _ProcessUpdateParams(self, params):
        """
        Purpose:
            Implement update from an array of parameters
        """
        if(isinstance(params, dict)):
            if (hasattr(params, 'C')):
                dico_params = dict(params)
                C_new = np.copy(self.GetParams('C'))
                C_new[self._index_even] = params['C'][self._index_even]
                dico_params['C'] = C_new

            dico_params = ut.filter_dico(params, self._eff_params_list)

        else:            
            C_new = np.copy(self.GetParams('C'))
            C_new[self._index_even] = params[1:]
            dico_params = {'c0':params[0], 'C': C_new}

        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        return 2 * (nb_params -1) # c0 is a param


class cChebyshevEvenFixedCt(cChebyshev):
    """
    Purpose:
        Chebyshev with evn polynomials fixed (i.e. not considered as a 
        parameter for the updates) - even polynomials are the Tn(-1) == Tn(1)
    Rmk:
        + Can have odd components if defined at initialization (bu can't update them)
    #TODO: Implement init with C_even
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        cChebyshev.__init__(self, params, constraints, typeSS, boundaries)
        self._index_even = np.arange(1, self._nbHarmonics, 2)
        self._eff_params_list = ['C', 'C_even']
        self._nbTotalParams = len(self._index_even)
    
    def _ProcessUpdateParams(self, params):
        """
        Purpose:
            Implement update from an array of parameters
        """
        if(isinstance(params, dict)):
            if (hasattr(params, 'C')):
                dico_params = dict(params)
                C_new = np.copy(self.GetParams('C'))
                C_new[self._index_even] = params['C'][self._index_even]
                dico_params['C'] = C_new

            dico_params = ut.filter_dico(params, self._eff_params_list)

        else:            
            C_new = np.copy(self.GetParams('C'))
            C_new[self._index_even] = params
            dico_params = {'C': C_new}

        return dico_params

    @classmethod
    def GetNbH(cls, nb_params):
        return 2 * nb_params # c0 
    
class cChebyshevFixedTrend(cChebyshevEvenFixedCt):
    """
    Purpose:
        Exactly as cChebyshevEvenFixedCt except that initialization can deal with parameters
        'a' and 'b'
        
    """
    def __init__(self, params, constraints = None, typeSS = 'scale', boundaries = None):
        a, b = params.get('a'), params.get('b')
        if(b is not None):
            params['c0'] = b
        cChebyshevEvenFixedCt.__init__(self, params, constraints, typeSS, boundaries)
        self._UpdateLinearTrend(a)
        self._nbTotalParams = len(self._index_even)
    
    def _UpdateLinearTrend(self, a):
        if(a is not None):
            C_new = np.copy(self.GetParams('C'))
            C_new[0] = a
            self.UpdateParams({'C': C_new})
            
    @classmethod
    def GetNbH(cls, nb_params):
        return 2 * nb_params     






#==============================================================================
#TODO: TO BE REIMPLEMENTED AS SUBCLASS OF PARAMETRIZED FUNCTION
# ShiftedAndScaled
# Bounded
# TODO: re-think the way they are implemented.. maybe use composition .. wabout the gradients
#==============================================================================
class ShiftedAndScaled2():
    """
    Purpose:
        Shift and scale a function wrt certain constraints 
        (i.e. f(t0) = y0 and f(t1) = yT)
        
    + For 2 endpoints:
        - typeSS = 'scale'
        - typeSS = 'sin'
        
    + For 1 endpoint:
        - typeSS = 'shift'
        - typeSS = 'scale'
        
    Meant to be inherited in concrete ParametrizedFunctions or collections
    
    Example:
        ShiftedAndScaled([[0,0], [1,5]])
        ShiftedAndScaled([[0,0]], 'shift')
        ShiftedAndScaled([[2,3]]) 
    """
    def __init__(self, constraints = None, typeSS = 'scale'):
        self.__SetSaS(constraints, typeSS)

    def __SetSaS(self, constraints = None, typeSS = 'scale'):
        self._typeSS = typeSS
        self._constraints = constraints
        self.UpdateSaS(self._constraints)

    def UpdateSaS(self, constraints = None, typeSS = None):
        if (constraints is None):
            constraints = self._constraints
        if (typeSS is None):
            typeSS = self._typeSS 
            
        self._scale = 1
        self._shift = 0
            
        if (constraints is not None):
            if not(ut.is_iter(constraints[0])):
                constraints = [constraints]
                
            if(len(constraints) == 2):
                x0, y0 = constraints[0]
                x1, y1 = constraints[1]
                f0, f1 = self.__call__(x0), self.__call__(x1)
                if(typeSS == 'sin'):
                    # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                    assert (y0 == y1), "invalid constraints"
                    self._shift = y0
                    self._scale = lambda t: np.sin(np.pi / (x1 - x0) * (t - x0))
                elif(typeSS == 'sin2'):
                    # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                    assert (y0 == y1), "invalid constraints"
                    self._shift = y0
                    self._scale = lambda t: np.square(np.sin(np.pi * (x1 - x0) * (t - x0)))
                
                else:
                    # f'(t) = shift + scale * f(t)
                    if ((f1-f0)!=0):
                        self._scale = (y1-y0) / (f1-f0)
                    else:
                        print('scale fixed to 1')
                        self._scale=1
                    self._shift = y0 - f0 * self._scale        
            elif(typeSS == 'scale'):
                x0, y0 = constraints[0]
                f0 = self.__call__(x0)

                if (f0 == 0):
                    print('scale fixed to 1')
                    self._scale = 1 
                else:
                    self._scale = y0/f0
            
            elif(typeSS == 'shift'):
                x0, y0 = constraints[0]
                f0 = self.__call__(x0)
                self._shift = y0 - f0
                
    @ut.vectorise_method
    def ShiftAndScale(self, value, t = None):
        if(self._scale is None):
            scale = 1
        elif(hasattr(self._scale, '__call__')):
            scale = self._scale(t)
            
        else:
            scale = self._scale
            
        if(self._shift is None):
            shift = 0
        else:
            shift = self._shift
        
        return value * scale + shift
   
         
class Bounded2():
    """
    Purpose:
        impose boundaries on the output of a function
    Example
    
    """
    def __init__(self, boundaries = None):
        self.__SetupBoundaries(boundaries) #To ensure f(0) = self.norm

    def __SetupBoundaries(self, boundaries = None):
        self._boundaries = boundaries
    
    def UpdateBoundaries(self, boundaries = None):
        self.__SetupBoundaries(boundaries)
    
    @ut.vectorise_method
    def Bound(self, value):
        if ((hasattr(self, '_boundaries')) and (self._boundaries is not None)):
            if value < self._boundaries[0]:
                res = self._boundaries[0]
            elif value > self._boundaries[1]:
                res = self._boundaries[1]
            else:
                res = value
        else:
            res = value
            
        return res

class Overwrite2():
    """
    Purpose:
        Overwrite 
    
    """
    def __init__(self, listX = None, listY=None):
        self._listX = None
        self._listY = None
        self._list_is_interval = None
        self._nbOWPoints = 0
        self.__SetupOW(listX, listY) #To ensure f(0) = self.norm

    def __SetupOW(self, listX, listY):
        if(listX is not None):
            assert len(listX) == len(listY), "listX and listY should have the same length"
            self._listX = listX
            self._listY = listY
            self._list_is_interval = [isinstance(x, tuple) for x in self._listX]
            self._nbPoints = len(listX)
        
    def UpdateOW(self, listX, listY):
        self.__SetupOW(listX, listY)
    
    @ut.vectorise_method
    def OW(self, x):
        res = [False, None]
        for i in self._nbOWPoints:
            if(self._list_is_interval[i]):
                check = (x>= self._listX[i][0]) * (x<= self._listX[i][1]) 
            else:
                check = (x == self._listX[i])

            if(check):
                res = [True, self._listY[i]]
                break
            
        return res








#==============================================================================
# Some testing
#==============================================================================
if __name__ == '__main__':
    import matplotlib.pylab as plt

# --------------------------------------------------------------------------- #
#   Fourier Function 
# --------------------------------------------------------------------------- #        
    
    # Build some Fourier functions
    four_params= {'A': [1,0], 'B':[0,0], 'c0':0, 'om':2*np.pi, 'Phi':0}
    fourFun = FourierFunc(params = four_params)

    x = np.arange(0, 1,0.001)
    fourFun.PlotFunction(x)
    fourFun.UpdateParams({'Phi':[np.pi/2, np.pi/2]})
    fourFun.PlotFunction(x)
    
    
    # 
    rep1 = repr(fourFun)
    repfun = ParametrizedFunctionFactory.Repr2Fun(rep1)
    repfun.PlotFunction(x)
    rep2 = fourFun.Fun2Dico()
    repfun2 = ParametrizedFunctionFactory.Repr2Fun(rep2)
    repfun2.PlotFunction(x)

    
    
# --------------------------------------------------------------------------- #
#   Collection of functions
# --------------------------------------------------------------------------- #          
    fourFun1 = FourierFunc(params = {'A': [1,0], 'B':[0,1], 'c0':0, 'om':2*np.pi+0.3, 'Phi':0})
    fourFun2 = FourierFunc(params = {'A': [1,0], 'B':[0,1], 'c0':0, 'om':2*np.pi+0.3, 'Phi':0}, constraints = [[0,0.5], [1,0.7]])
    linFun= LinearFunc({'a': 1, 'b':0})
    fourFun1.PlotFunction(x)
    fourFun2.PlotFunction(x)
    linFun.PlotFunction(x)
    sumFun = Plus([fourFun1], boundaries = [0,1])
    productFun = Times([sumFun, linFun])
    sumFun.PlotFunction(x)
    productFun.PlotFunction(x)
    sumFun.AddNewFunctions([fourFun2], order = 0)
    sumFun.PlotFunction(x) 
    #productFun.UpdateParams({'B':[0,0]}, indexFun = 0)
    #productFun.PlotFunction(x)
    
    rep1 = repr(productFun)
    FunCopy = CollectionParametrizedFunctionFactory.Repr2Fun(rep1)
    FunCopy.PlotFunction(x)
    productFun.PlotFunction(x)

# --------------------------------------------------------------------------- #
#   Test Chebyshev
# --------------------------------------------------------------------------- #     
    cheb_params= {'C': [0,1,0,0], 'c0':1, 'domain':[0,1]}
    chebFun1 = ChebyshevFun(params = cheb_params)
    ch1 = np.array([chebFun1(t) for t in x])
    cheb_params= {'C': [0,0,0,1], 'c0':0, 'domain':[0,1]}
    chebFun2 = ChebyshevFun(params = cheb_params)
    ch2 = np.array([chebFun2(t) for t in x])
    lin_params= {'a': 1, 'b':0}
    linFun= LinearFunc(lin_params)
    lf1 = np.array([linFun(t) for t in x])

    sumFun = Plus([chebFun1])
    sf1 = np.array([sumFun(t) for t in x])
    productFun = Times([sumFun, linFun])
    pf1 = np.array([productFun(t) for t in x])
    
    
    ff2 = np.array([fourFun2(t) for t in x])
    sumFun.AddNewFunctions(fourFun2, order = 0)
    sf2 = np.array([sumFun(t) for t in x])
    pf2 = np.array([productFun(t) for t in x])    
    
    plt.plot(x, ch1)
    plt.plot(x, ch2)
    plt.plot(x, lf1)
    plt.plot(x, sf1)
    plt.plot(x, ff2)
    plt.plot(x, sf2)
    plt.plot(x, pf1)
    plt.plot(x, pf2)
    