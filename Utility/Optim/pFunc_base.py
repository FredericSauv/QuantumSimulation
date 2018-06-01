#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:18 2018

@author: fred
Strongly inspired by sklearn.gaussian_process.kernels
"""

if(__name__ == '__main__'):
    import sys
    sys.path.insert(0, '../')
    import Helper as ut
else:
    from .. import Helper as ut

import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pylab as plt
import pdb 

import sklearn.gaussian_process.kernels as ker


#==============================================================================
# parametrized function base class
# 
#==============================================================================
class pFunc_base():
    """ Abstract class for non-nested parametric functions. Still potential nested 
    behavior is implemented here but fully exploited in the collection subclass 
    (in order not to re-implement the same methods in thse subclass). Nested 
    behavior is flagged by 'deep' parameter or the check of '_FLAG_TYPE' ('base'
    for non nested functions and 'collection' for potentially nested structure)
        
        parameters: a list of elements
        
        param_bounds: represent either the bounds (as a 2-uplet) or a possible 
            fixing of the hyperparameters (False when parameters is fixed) or True
            if he is free 
                i.e. [(-2, 5), False, True] means that the first element of
        
        theta: flat np.array of the values of non-fixed hyperparameters
    
    
    e.g. for a two parameters ('weights' and 'bias') function with bias fixed 
    the structure is the following (notice everything is stored as a list/ array
    even when )
    self._LIST_PARAMS = ['weights', 'bias']
    self.__weights = np.array([3, 6])
    self.__bias = np.array([1])
    self.__weights_bounds = [(1e-5, 1000), (1e-5, 1000)]
    self.__bias = [False]
    """

    # may be usefull to diff a function to a collection (nested structure)
    _FLAG_TYPE = 'base'     
    _LIST_PARAMETERS = []
    _DEF_BOUNDS = (1e-10, 1e10)
    def __init__(self, params = None, fix_params = False, **args_behavior):
        pass

        
        
#-----------------------------------------------------------------------------#
# Parameters management
#-----------------------------------------------------------------------------#
    def _setup_params_and_bounds(self, param_name, val, bounds):
        """ 
        """
        setattr('__'+param_name, val)
        bounds_processed = self._process_bounds(param_name, bounds)
        setattr('__'+param_name+'_bounds', bounds_processed)
        
        
    def _process_bounds(self, param_name, bounds):
        """ Process the bounds to match specifications, i.e. a list with the same
        length as the number of values for the parameter, each element being either
        False or a pair of values)
        """
        l = self.n_elements_one_param(param_name)
        if(ut.is_iter(bounds)):
            if(len(bounds) != l):
                raise ValueError('Length of parameter values (%s) and bounds (%s). '
                    'dont match ' % (l, len(bounds)))
            res = [self._process_one_val_bound(b) for b in bounds]
        else:
            res_one = self._process_one_val_bound(bounds)
            res = [res_one for _ in range(l)]
        
        return res
    
    def _process_one_val_boud(self, val):
        """ if True is passed use default bounds, if False keep False, if none it
        should be a pair of elements with the first being the min"""
        if(val == True):
            res = self._DEF_BOUNDS
        elif(val == False):
            res = False
        else:
            if(len(val) != 2):
                raise ValueError('Bound value is not recognized. '% (str(val)))
            if(val[0] > val[1]):
                raise ValueError('Bound values are inverted '% (str(val)))
            res = val
        return res
        
        
        
    def _check_integrity(self):
        """ Ensure that for each parameter ('abc') there is indeed a list of 
        values (which is at self.__abc) and a list of bounds (self.__abc_bounds), 
        that they have the same size and nothing is outside the boundaries
        """
        
        for param_name in self._LIST_PARAMETERS:
            param = self._get_one_param(param_name)
            bounds = self._get_one_param(param_name)
            if(len(bounds) != len(param)):
                raise ValueError('Length of parameter values (%s) and bounds (%s). '
                    'dont match ' % (len(param), len(bounds)))                
            
            for n, val in enumerate(param):
                if(bounds[n] != False):
                    if((val<bounds[n][0]) or (val>bounds[n][1])):
                        raise ValueError('Value %s of parameter %s doesnt comply'
                                'with bounds ' % (n, param_name, str(bounds)))
                
    
    def get_params(self, deep = True):
        """Get parameters (and bounds) of the function.
        
        Returns
        -------
        params : dictionary 
        """
        params = dict()        
        for p in self._LIST_PARAMETERS:
            params[p] = self._get_one_param(p)
            params[p+'_bounds'] = self._get_one_bound(p)
            if(deep and self._FLAG == 'collection'):
                for n, f_nested in enum(params[p]):
                    params_nested = f_nested.get_params(deep)
                    params.update(('f'+str(n)+'__'+key, val)for key, val in params_nested)
                    
        return params


    def set_params(self, **params):
        """Set the parameters of this kernel.
        
        Parameters
        ----------
        params: should be h or h_bounds with h in _LIST_HYPERPARAMETERS
        
        Remark
        ------
        (1) To pass the check_integrity test when a parameter is updated with a new 
        number of elements a new bounds should be provided
        (2) Nested behavior has delibarately not been implemented. Is there 
        a case where it could be needed
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep = False)
        for key, value in enumerate(params):
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for function %s. '
                    'Check the list of available parameters '
                    'with `kernel.get_params().keys()`.' % (key, self))
            else:
                setattr(self, '__'+key, value)    
        self._check_integrity()
        
        
    @property
    def n_parameters(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return len(self._LIST_PARAMETERS)
    
    @property
    def n_theta(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def n_elements_one_param(self, param_name):
        """ return the number of elements of one parameter"""
        p = self._get_one_param(param_name)
        return len(p)
    
    @property
    def n_theta_one_param(self, param_name):
        """Get the number of free values for one parameter."""
        res = len(self._get_one_param_theta())
        return res
        

    ### Recall thetas are the free parameter values
    @property
    def theta(self):
        """Returns the (flattened) non-fixed hyperparameters.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, parameters values of the function (ordered following
            self._LIST_PARAMETERS)
        """
        tmp = [self._get_one_param_theta(p) for p in self._LIST_PARAMETERS]
        return np.hstack(tmp)
        
    
    @theta.setter
    def theta(self, theta):
        """Sets the non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for p in self._LIST_PARAMETERS:
            nb_to_update =  self.n_theta_one_param(p)
            if nb_to_update > 0:
                mask_free = self._get_one_free_mask(p)

                if(self._FLAG_TYPE == 'base'):    
                    params[p][mask_free] = theta[i:i + nb_to_update]
                    i += nb_to_update

                elif(self._FLAG_TYPE == 'collection'):
                    list_sub_free = [p[n] for n, pp in enumerate(p) if mask_free[n]]
                    for sub in list_sub_free:
                        nb_to_update_sub = sub.n_theta()
                        sub.theta = theta[i:i + nb_to_update]
                        i += nb_to_update_sub
                        
        # check all the values in theta have been used
        if i != len(theta):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))
        self.set_params(**params)


    @property
    def theta_bounds(self):
        """Returns the bounds on the theta.

        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = []
        for p in self._LIST_HYPERPARAMETERS:
            mask_free = self._get_one_free_mask(p)
            if(self._FLAG_TYPE == 'base'): 
                bounds += self._get_one_bound(p)[mask_free]
                
            elif(self._FLAG_TYPE == 'collection'):
                list_sub_free = [p[n] for n, pp in enumerate(p) if mask_update[n]]
                for sub in list_sub_free:
                    bounds += self._theta_bound(sub)
        return bounds



    def _get_one_param(self, param_name, deep = True):
        """Get one parameter by name."""
        return getattr(self, param_name)
    
    def _get_one_bound(self, param_name):
        """Get bounds associated to one parameter by name (of the param)."""
        bounds = getattr(self, '__' + param_name + '_bounds')
        return bounds

    def _get_one_param_theta(self, param_name):
        """Get the thetas for one parameter by name."""
        res = self._get_one_param(param_name)[self._get_one_free_mask(param_name)]
        return res
        
    def _get_one_free_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = (0,5)
        it means that this value is free and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return np.array([not(b == False) for b in bounds])


    def _get_one_fixed_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = False
        it means that this value is fixed and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return np.array([b == False for b in bounds])
   

#-----------------------------------------------------------------------------#
# IO
#-----------------------------------------------------------------------------#
    def __add__(self, b):
        if not isinstance(b, pFunc_base):
            return Sum([self, ConstantFunc(b)])
        return Sum([self, b])

    def __radd__(self, b):
        if not isinstance(b, pFunc_base):
            return Sum([ConstantFunc(b), self])
        return Sum([b, self])

    def __mul__(self, b):
        if not isinstance(b, pFunc_base):
            return Product([self, ConstantFunc(b)])
        return Product([self, b])

    def __rmul__(self, b):
        if not isinstance(b, pFunc_base):
            return Product([ConstantFunc(b), self])
        return Product([b, self])

    def __pow__(self, b):
        raise NotImplementedError()
    
    def __dot__(self, b):
        if not isinstance(b, pFunc_base):
            raise NotImplementedError()
        return Compo([self, b])
    
    def __rdot__(self, b):
        if not isinstance(b, pFunc_base):
            raise NotImplementedError()
        return Compo([b, self])

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}(**{1})".format(self.__class__.__name__, self.get_params())                             

    @abstractmethod
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        
    def clone(self, theta):
        """Returns a clone of self with given hyperparameters theta. """
        cloned = eval(repr(self))
        return cloned
    
    @classmethod
    def build_from_various(self, representation):
        """ Build a function from a str / dico 
        potentially add others 
        (can be called from an other module to ensure access to all the classes)
        """
        if ut.is_str(representation):
            func = eval(representation)
            
        elif ut.is_dico(representation):
            name_func = representation.pop('name_func')
            func = eval(name_func)(**representation)
            
        else:
            raise NotImplementedError()
    
        return func

#-----------------------------------------------------------------------------#
# Some extra capabilities
#-----------------------------------------------------------------------------#
    def plot_function(self, range_x, **args_plot):
        """ plot the function
        """
        y = [self.__call__(x) for x in range_x]
        plt.plot(np.array(range_x), np.array(y), **args_plot)
    
    
    def fluence(self, range_x):
        """ Fluence as sum_x f(x)**2 dx / (x_max - xmin)
        """
        time_step = np.diff(range_x)
        val_square = np.square(np.abs(self(range_x)))
        res = np.sum(np.array(val_square[:-1] * time_step))
        res = res/(range_x[-1] - range_x[0])
        return res
    
    def smoothness(self, range_x):
        """ Smoothness as avg (f(x) - f(x + dx) ** 2 / (x_max - x_min)
        """
        step = np.diff(range_x)
        diff_val_square = np.square(np.diff(self(range_x)))
        res = np.sum(np.array(diff_val_square / step))
        res = res/(range_x[-1] - range_x[0])
        return res
    

    @staticmethod
    def l2_dist(func1, func2):
        """ 
        """
        raise NotImplementedError()

    @staticmethod
    def l2_norm(func1, range_x):
        """ 
        """
        raise NotImplementedError()
    
    @staticmethod
    def l2_dist_list(list_func, range_x, funcRef = None):
        """ Compute the L2 distance of a list of functions to the first one
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        if(func_ref is None):
            func_ref = list_func[0]
            
        l2_dists = [pFunc_base.l2_dist(func_ref, f) for f in listFunc] 
        return l2_dists

         
class pFunc_collec(pFunc_Base):
    """Abstract collection of <pFunc_Base> or <pFunc_collect> which can be used 
    to represent any specific functions composition. Bounds in these context is 
    a boolean list indicating if the underlying objects are treated as free (True)
    or fixed (False).
    
    e.g. for a collection of one functions 'f1' and a collection 'c2' 
    with f1 explictly fixed independently of the fixing parameters details of 
    f1 the structure is the following:
        
    self._LIST_PARAMS = ['list_func']
    self.__list_func = [<pFunc_base> f1, <pFunc_collec> c1]
    self.__list_func_bounds = [False, True]
    self.__bias = [False] 
    """
    _FLAG_TYPE = 'collection'
    _LIST_PARAMETERS = ['list_func']
    
    def __init__(self, list_func, list_func_bounds = None):
        if(list_func_bounds is None):
            list_func_bounds = True
        self._setup_params_and_bounds('list_func', list_func, 'function')
        
        
    def _process_one_val_bound(self, val):
        """ if False keep False (means fixed functions), """
        if(val not in [True, False]):
            raise ValueError('For composition bounds expected are iether True 
                '(free function) or False (fixed function) not %s' % (str(val)))
        return val
    
    @property    
    def list_func(self):
        self._get_one_param('list_func')
        
    def _check_integrity(self):
        """ Ensure that for each parameter ('abc') there is indeed a list of 
        values (which is at self.__abc) and a list of bounds (self.__abc_bounds), 
        that they have the same size and nothing is outside the boundaries
        """
        for f in self.list_func:
            if(not(isinstance(f, pFunc_collec) or isinstance(f, pFunc_base))):
                raise ValueError('type %s while expecting pFunc_base or collection'
                    ' ' % (str(type(f)))



    def _get_one_param(self, param_name, deep = True):
        """Get one parameter by name."""
        return getattr(self, param_name)
    
    def _get_one_bound(self, param_name):
        """Get bounds associated to one parameter by name (of the param)."""
        bounds = getattr(self, '__' + param_name + '_bounds')
        return bounds

    def _get_one_param_theta(self, param_name):
        """Get the thetas from each func (or nested collection). This is nested by nature
        """
        if(param_name != 'list_func'):
            raise NotImplementedError("In collections only one parameter allowed:"
                             " it should be list_func; not %s"% (param_name))
        l_func = self._get_one_param(param_name)
        l_free = self._get_one_free_mask(param_name)
        res = np.hstack([f.theta for i, f in range(len(lst)) if lfree[i]])
        return res
        
    def _get_one_free_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = (0,5)
        it means that this value is free and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return np.array([not(b == False) for b in bounds])


    def _get_one_fixed_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = False
        it means that this value is fixed and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return np.array([b == False for b in bounds])
   


    # ----------------------------------------------------------------------- #
    #   I/O
    # ----------------------------------------------------------------------- #
    def to_dico(self):
        """ Output a dico with vital infos .. is it necessary??"""
        raise NotImplementedError()
    
    def __repr__(self):
        name_collec = self.__class__.__name__
        list_func = str([repr(f) for f in self.list_func])
        list_func_bounds = str(self.list_func_bounds)
        return "{0}(list_func={1}, list_func_bounds={2})".format(name_collec, 
                list_func, list_func_bounds) 
            
    
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """return a list of each evaluation function."""
        list_pfunc = self._get_one_param('list_func')
        return [f(variable) for f in list_func]
    
    ### New methods    
    @property
    def list_func(self):
        return self._get_one_param('list_func')
    
    @property
    def list_func_bounds(self):
        return self._get_one_bound('list_func')
    
    def add_new_function(self, pfunc, pfunc_bound= True, index = None):
        """ Add new <pFunc_base> or <pFunc_collect> to the collection"""
        raise NotImplementedError()

    def replace_function(self, pfunc, index = -1):
        """ Replace a <pFunc_base> or <pFunc_collect> the collection """
        raise NotImplementedError()
        
    def remove_function(self, index = 0):
        """ Remove a <pFunc_base> or <pFunc_collect> the collection """
        raise NotImplementedError()
            
    def get_function(self, index = 0):
        """ Return the index-th <pFunc_base> or <pFunc_collect> of the collection"""
        return self._get_one_param('list_func')[index]
        
    
# --------------------------------------------------------------------------- #
#   Implementations
# --------------------------------------------------------------------------- #
class Product(pFunc_collec):
    """Product of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) * f2(t) * c1(t) """
    

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.product(pFunc_collec(X, Y, eval_gradient))
        
    
class Sum(CollectionParametrizedFunctionFactory):        
    """Sum of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) + f2(t) + c1(t) """
    
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.sum(pFunc_collec(X, Y, eval_gradient))
    
 
class Composition(CollectionParametrizedFunctionFactory):
    """Composition of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(f2(c1(t)))) """
    @ut.vectorise_method

    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        list_pfunc = self._get_one_param('list_func')
        for f in list_pfunc:
            X = f(X, Y, eval_gradient=False)
        return X
        