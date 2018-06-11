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
from numpy import array
import numpy.polynomial.chebyshev as cheb
import functools as ft
import matplotlib.pylab as plt
import pdb 

# import sklearn.gaussian_process.kernels as ker


#==============================================================================
# parametrized function base classes
#==============================================================================
class pFunc_base():
    """ Abstract class for non-nested parametric functions. Still potential nested 
    behavior is implemented here but fully exploited in the collection subclass 
    (in order not to re-implement the same methods in thse subclass). Nested 
    behavior is flagged by 'deep' parameter or the check of '_FLAG_TYPE' ('base'
    for non nested functions and 'collection' for potentially nested structure)
        
        parameters: a list of elements
        
        param_bounds: represent either the bounds (as a **2-uplet**) or a possible 
            fixing of the hyperparameters (**False** when parameters is fixed) or use 
            **True** if param element(s) is(are) free and happy to use default boundaries
            or **None** if param(s) is (are) free and one doesn't want to put boundaries
            i.e. [(-2, 5), False, True, None] means that the first element of
        
        theta: flat np.array of the values of non-fixed hyperparameters
    
    
    e.g. for a two parameters ('weights' and 'bias') function with bias fixed 
    the structure is the following (notice everything is stored as a list/ array
    even when )
    self._LIST_PARAMS = ['weights', 'bias']
    self.__weights = np.array([3, 6])
    self.__bias = np.array([1])
    self.__weights_bounds = [(1e-5, 1000), (1e-5, 1000)]
    self.__bias = [False]
    
    TODO:
        + use of different default bounds (e.g. _DEF_POS_BOUNDS)
        + test (D-CRAB setups)

    """

    _FLAG_TYPE = 'base'     
    _LIST_PARAMETERS = []
    _NB_ELEM_PER_PARAMS = []
    _DEF_BOUNDS = (-1e10, 1e10)
    _DEF_POS_BOUNDS = (1e-10, 1e10)

    def __init__(self, **args_func):
        """ Look for all the parameters specified in _LIST_PARAMETERS and take by default bounds 
        associated to these params to be None (i.e they are no fixed but no bounds are associated) 
        """
        if(args_func.get('debug')):
            pdb.set_trace()

        if(args_func.get('fix_all')):
            build_fixed = True
        else:
            build_fixed = False

        for p in self._LIST_PARAMETERS:
            if build_fixed:
                self._setup_params_and_bounds(p, args_func[p], False)
            else:
                self._setup_params_and_bounds(p, args_func[p], args_func.get(p+'_bounds', None))

        self._check_integrity()
    #-----------------------------------------------------------------------------#
    # Parameters management
    #-----------------------------------------------------------------------------#
    @classmethod
    def help(cls):
        """ Print the name of parameter expected """
        print(cls._LIST_PARAMETERS)


    def fix_all(self):
        """ Fix all the parameters"""
        for param_name in self._LIST_PARAMETERS:
            bounds_processed = self._process_bounds(param_name, False)
            setattr(self, '__'+param_name+'_bounds', bounds_processed)

    def _setup_params_and_bounds(self, param_name, vals, bounds):
        """ 
        """
        vals_processed = self._process_vals(param_name, vals)
        setattr(self, '__'+param_name, vals_processed)
        bounds_processed = self._process_bounds(param_name, bounds)
        setattr(self, '__'+param_name+'_bounds', bounds_processed)

    def _process_vals(self, param_name, vals):
        """ just ensure that cal is an iterable (if single value passed return 
        a list)
        """
        if(not(ut.is_iter(vals))):
            vals = [vals]
        return array(vals)
        
    def _process_bounds(self, param_name, bounds):
        """ Process the bounds associated to a parameter which has l elements.
        bounds can be:
            + a list with l potentially distinct individual bounds
            + an individual bound (which will be used for each of the l elements)
            
        TODO: CAN DO BETTER .. TOO MANY IF ELSE ETC..
        """
        l = self.n_elements_one_param(param_name)
        if(ut.is_iter(bounds)):
            if(len(bounds)  == l):
                if(l!=2):
                    res = [self._process_individual_bound(b) for b in bounds]
                else:
                    try:
                        res = [self._process_individual_bound(b) for b in bounds]
                    except:
                        res_one = self._process_individual_bound(bounds)
                        res = [res_one for _ in range(l)]

            elif(len(bounds)  == 2):
                # slightly tricky as l can be = 2
                res_one = self._process_individual_bound(bounds)
                res = [res_one for _ in range(l)]

            else:
                raise ValueError('Bounds length (%s) is not recognized. '% (len(bounds)))
        else:
            res_one = self._process_individual_bound(bounds)
            res = [res_one for _ in range(l)]
        
        return res
    
    def _process_individual_bound(self, val):
        """ if True is passed use default bounds, if False keep False, if none it
        should be a pair of elements with the first being the min
        TODO: could deal with strings
        """
        if(val == True):
            res = self._DEF_BOUNDS
        elif(val in [False, None]):
            res = val
        else:
            if(len(val) != 2):
                raise ValueError('Bound value is not recognized. '% (str(val)))
            if(val[0] > val[1]):
                raise ValueError('Bound values are inverted '% (str(val)))
            res = val
        return res
        
        
        
    def _check_integrity(self):
        """ For each parameter ('abc') ensure that:
        + there is indeed a list of values (which is at self.__abc) 
         and a list of bounds (self.__abc_bounds), with the same size.
        + nothing is outside the boundaries,
        + if an int has been stored in _NB_ELEM_PER_PARAMS verif that it matches,
        if it's a string, ensure each param with the same string attached have the
        same number of element
        """
        list_flag =[]
        list_nb_el = []
        
        for n, param_name in enumerate(self._LIST_PARAMETERS):
            param = self._get_one_param(param_name)
            bounds = self._get_one_bound(param_name)
            el_exp = self._NB_ELEM_PER_PARAMS[n]
            nb_el = len(param)

            if(len(bounds) != nb_el):
                raise ValueError('Length of parameter values (%s) and bounds (%s). '
                    'dont match ' % (nb_el, len(bounds)))                

            if(ut.is_str(el_exp)):
                if(el_exp not in list_flag):
                    list_flag.append(el_exp)
                    list_nb_el.append(nb_el)
                nb_el_exp = list_nb_el[list_flag.index(el_exp)]
            
            else:
                nb_el_exp = el_exp

            if ut.is_x_not_in_y(x=nb_el, y=nb_el_exp):
                    raise ValueError('Length of parameter values expected for %s' 
                    'is (%s). '% (param_name, nb_el))

            for n, val in enumerate(param):
                if(bounds[n] not in [False, None]):
                    if((val<bounds[n][0]) or (val>bounds[n][1])):
                        raise ValueError('Value %s of parameter %s doesnt comply'
                                'with bounds %s' % (n, param_name, str(bounds[n])))
                
    
    def get_params(self, deep = True, bounds = True):
        """Get parameters (and bounds) of the function.
        
        Returns
        -------
        params : dictionary 
        """
        params = dict()        
        for p in self._LIST_PARAMETERS:
            params[p] = self._get_one_param(p)
            if(bounds):
                params[p + '_bounds'] = self._get_one_bound(p)
            if(deep and self._FLAG_TYPE == 'collection' and p == 'list_func'):
                for n, sub_obj in enumerate(params[p]):
                    sub_params = sub_obj.get_params(deep, bounds)
                    params.update(('f' + str(n) + '__' + key, val) for key, val in sub_params)
                    
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
        (2) Nested behavior has been implemented. Is there a case where it is needed
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep = True)       
        for key, value in params.items():
            split = key.split('__', 1)

            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                index_subobj = int(name.replace('f', '', 1))
                list_subobj = self._get_one_param('list_func')
                if index_subobj > len(list_subobj):
                    raise ValueError('Looking for the %s -th nested function but'
                                    'there is only %s functions . ' %
                                     (index_subobj, len(list_subobj)))

                sub_object = list_subobj[index_subobj]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for function %s. '
                                     'Check the list of available parameters '
                                     'with `cls.print_params_name()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, '__' + key, value)
        
        
    @property
    def n_parameters(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return len(self._LIST_PARAMETERS)
    
    def n_elements_one_param(self, param_name):
        """ return the number of elements of one parameter"""
        p = self._get_one_param(param_name)
        return len(p)


    ### Recall thetas are the free parameter values // recursive behavior by default 
    @property
    def n_theta(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]
    
    def n_theta_one_param(self, param_name):
        """Get the number of free values for one parameter."""
        res = len(self._get_one_param_theta())
        return res
        
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
            The non-fixed, hyperparameters of the kernel
        """
        n_theta_avail = self.n_theta
        n_theta_suggest = len(theta)
        if(n_theta_avail != n_theta_suggest):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (n_theta_avail, n_theta_suggest))            

        i = 0
        if(self._FLAG_TYPE == 'base'):   
            params = self.get_params(bounds = False)
            for p_name in self._LIST_PARAMETERS:
                mask_free = self._get_one_free_mask(p_name)
                nb_to_update = np.sum(mask_free)
                if nb_to_update > 0:    
                    params[p_name][mask_free] = theta[i:i + nb_to_update]
                    i += nb_to_update

            #That's wehere the updates really happen
            self.set_params(**params)

        elif(self._FLAG_TYPE == 'collection'):
            #Delegate updates to subobj
            for sub in self.list_func_free:
                nb_to_update_sub = sub.n_theta
                sub.theta = theta[i:i + nb_to_update_sub]
                i += nb_to_update_sub

        # check all the values in theta have been used
        if i != len(theta):
            raise ValueError("Something went wrong with nb of theta updated. "
                             " Should be %d; but it is only %d"
                             % (n_theta_avail, i))

        self._check_integrity()

    @property
    def theta_bounds(self):
        """Returns the bounds on the theta.

        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = []
        for p in self._LIST_PARAMETERS :
            bounds += self._get_one_param_theta_bounds(p) 
        return bounds


    def _get_one_param_theta(self, param_name, deep = True):
        """Get the thetas for one parameter by name. Recursive behavior by def"""
        if(deep and self._FLAG_TYPE == 'collection' and param_name == 'list_func'):
            theta = []
            for sub in self.list_func_free:
                theta.append(sub.theta)
        else:
            vals = self._get_one_param(param_name)
            mask = self._get_one_free_mask(param_name)
            theta = [v for n, v in enumerate(vals) if mask[n]]
        if (len(theta) == 0):
            res = np.array(theta)
        else:
            res = np.hstack(theta)
        return res

    def _get_one_param_theta_bounds(self, param_name, deep = True):
        """Get the thetas for one parameter by name. Recursive behavior by def"""
        if(deep and self._FLAG_TYPE == 'collection' and param_name == 'list_func'):
            bounds_free = []
            for sub in self.list_func_free:
                bounds_free += sub.theta_bounds
        else:
            bounds = self._get_one_bound(param_name)
            mask = self._get_one_free_mask(param_name)
            bounds_free = [b for n, b in enumerate(bounds) if mask[n]]

        return bounds_free


    def _get_one_param(self, param_name):
        """Get one parameter by name. No nested behavior"""
        return getattr(self, '__' + param_name)
    
    def _get_one_bound(self, param_name):
        """Get bounds associated to one parameter by name (of the param).
        No nested behavior"""
        return getattr(self, '__' + param_name + '_bounds')
  
    def _get_one_free_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = (0,5)
        it means that this value is free and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return array([not(b == False) for b in bounds])


    def _get_one_fixed_mask(self, param_name):
        """Get the fixed mask associated to one parameter (i.e. if bound = False
        it means that this value is fixed and the value in the mask is True)
        , by name (of the param)."""
        bounds = self._get_one_bound(param_name)
        return array([b == False for b in bounds])
   

    #-----------------------------------------------------------------------------#
    # IO
    #-----------------------------------------------------------------------------#
    def __add__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False})
            return Sum([self, bFunc])
        return Sum([self, b])

    def __radd__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False})
            return Sum([bFunc, self])
        return Sum([b, self])

    def __mul__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False})
            return Product([self, bFunc])
        return Product([self, b])

    def __rmul__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False})
            return Product([bFunc, self])
        return Product([b, self])

    def __pow__(self, b):
        raise NotImplementedError()
    

    def __eq__(self, b):
        """ doesn't deal well with the type pFunc_fromcllable"""
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}(**{1})".format(self.__class__.__name__, repr(self.get_params(deep = False)))                             

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        
    def clone(self, theta = None):
        """Returns a clone of self with given hyperparameters theta. """
        cloned = eval(repr(self))
        if(theta is not None):
            cloned.theta = theta
        return cloned
    
    @classmethod
    def build_from_various(self, representation):
        """ Build a function from a str / dico 
        potentially add others 
        (can be called from an other module to ensure access to all the classes)
        TODO: maybe incorporate/merge pFuncZoo here
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
        y = self.__call__(range_x)
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
    def l2_dist_list(list_func, range_x, func_ref = None):
        """ Compute the L2 distance of a list of functions to the first one
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        if(func_ref is None):
            func_ref = list_func[0]
            
        l2_dists = [pFunc_base.l2_dist(func_ref, f) for f in list_func] 
        return l2_dists

class pFunc_fromcallable(pFunc_base):
    """ wrap a callable object (function, lambda function) to have the (min)
    capabilities of pFunc
    TODO: haven't been tested used to yet but may be usefull
    """

    _FLAG_TYPE = 'callable'
    def __init__(**args_func):
        self._callable = args_func['callable']
    
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        res = self._callable(X)
        return res

    def __repr__(self):
        return "{0}(callable = {1})".format(self.__class__.__name__, repr(self._callable)) 

class pFunc_collec(pFunc_base):
    """Abstract collection of <pFunc_Base> or <pFunc_collect> which can be used 
    to represent any specific functions composition. Bounds in these context is 
    a boolean list indicating if the underlying objects are treated as free (True)
    or fixed (False). It is an iterable object (i.e can do for f in pFunc_collec
    where f will be the functions stored in 'list_func')
    
    e.g. for a collection of one functions 'f1' and a collection 'c2' 
    with f1 explictly fixed independently of the fixing parameters details of 
    f1 the structure is the following:
        
    self._LIST_PARAMS = ['list_func']
    self.__list_func = [<pFunc_base> f1, <pFunc_collec> c1]
    self.__list_func_bounds = [False, True]
    self.__bias = [False] 

    TODO: could we provide capabilities to deal with non pFunc function (should be 
    fine with theta related getter/setter )
    """
    _FLAG_TYPE = 'collection'
    _LIST_PARAMETERS = ['list_func']
    
    def __init__(self, list_func, list_func_bounds = True):
        self._setup_params_and_bounds('list_func', list_func, list_func_bounds)
        self.__init_counter = 0
    
    def __iter__(self):
        """ iterable capability"""
        i = self.__init_counter
        while i < self.n_func:
            yield self.get_function(i)
            i+=1

    def __getitem__(self, index):
        """ index capability"""
        return self.get_function(i)
        
    def _process_individual_bound(self, val):
        """ if False keep False (means fixed functions), """
        if(val not in [True, False]):
            raise ValueError('For composition bounds expected are iether True' 
                '(free function) or False (fixed function) not %s' % (str(val)))
        return val
    
    def _process_vals(self, param_name, vals):
        """ just ensure that cal is an iterable (if single value passed return 
        a list)
        """
        if(not(ut.is_iter(vals))):
            vals = [vals]
        return array(vals)
        
    def _check_integrity(self):
        """ Ensure that for each parameter ('abc') there is indeed a list of 
        values (which is at self.__abc) and a list of bounds (self.__abc_bounds), 
        that they have the same size and nothing is outside the boundaries
        """
        for f in self.list_func:
            if(not(isinstance(f, (pFunc_collec, pFunc_base, pFunc_fromcallable)))):
                raise ValueError('type %s while expecting pFunc_base or collection'
                    ' ' % (str(type(f))))
            f._check_integrity()

    # ----------------------------------------------------------------------- #
    #   I/O
    # ----------------------------------------------------------------------- #     
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """return a list of each evaluation function."""
        return [f(X, Y=Y, eval_gradient=eval_gradient) for f in self.list_func]
    
    ### New properties / methods
    @property
    def n_func(self):
        return len(self.list_func)

    @property
    def list_func(self):
        return self._get_one_param('list_func')
    
    @property
    def list_func_bounds(self):
        return self._get_one_bound('list_func')
    
    @property    
    def list_func_free_mask(self):
        return self._get_one_free_mask('list_func')

    @property    
    def list_func_free(self):
        list_func = self.list_func
        mask = self.list_func_free_mask
        return [f for n, f in enumerate(list_func) if mask[n]]

        
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
        
class pFunc_wrapper(pFunc_base):
    """ instead of params we have here hyperparams """
    _LIST_HYPERPARAMETERS=[]
    _NB_ELEM_PER_HYPERPARAMS = []

    def __init__(self, **args_wrapper):
        """ Look for all the hyperparameters specified in _LIST_HYPERPARAMETERS  
        use None if they are not specified"""
        if(args_wrapper.get('debug')):
            pdb.set_trace()

        for hp in self._LIST_HYPERPARAMETERS:
            setattr(self, '__'+hp, args_wrapper.get(hp))

    #-----------------------------------------------------------------------------#
    # hyper-parameters management (straight-forward)
    #-----------------------------------------------------------------------------#
    @classmethod
    def help(cls):
        print(cls._LIST_HYPERPARAMETERS)
    
    def get_hyperparams(self):
        """Return a dico with the hyperparameters """
        params = dict()        
        for hp in self._LIST_HYPERPARAMETERS:
            params[hp] = self._get_one_param(hp)
        return params

    def set_hyperparams(self, **hyperparams):
        """Return a dico with the hyperparameters """
        valid_params = self.get_hyperparams()         
        for key, value in hyperparams.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for function %s. '
                        'Check the list of available parameters '
                        'with `cls.print_params_name()`.' %
                        (key, self.__class__.__name__))
            setattr(self, '__' + key, value)

    def update_dynamically(self, func):
        """ update the attributes of the object dynamically based on a provided func
        e.g. when something changes (parameters update) in func """
        raise NotImplementedError()

    #-----------------------------------------------------------------------------#
    # IO
    #-----------------------------------------------------------------------------#
    def wrap(self, func):
        """ Wrapping a function returns a new type still inheriting from pFunc_base"""
        return pFunc_wrapped(func, self)

    def __mul__(self, b):
        """ wrapper * func >> wrapped_function"""
        if not isinstance(b, pFunc_base):
            raise NotImplementedError()
        return self.wrap(b)
    
    # doen't haved to be vectorized (i.e. wrapped) as it will be always called
    # from pFunc_wrapped with scalar values 
    def __call__(self,  X, func, Y=None, eval_gradient=False):
        """ call rely on a func"""
        raise NotImplementedError

    def __repr__(self):
        return "{0}(**{1})".format(self.__class__.__name__, self.get_hyperparams())   

class pFunc_wrapped(pFunc_base):
    """ fusion of pfunc and pFunc_wrapped"""
    def __init__(self, func, wrapper):
        self._func = func
        self._class_func = func.__class__
        self._wrapper = wrapper
        self.update_wraper_hyperparams()

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        res = self._wrapper(X, self._func, Y, eval_gradient)
        return res

    def __repr__(self):
        return "{0}({1},{2})".format(self.__class__.__name__, 
            repr(self._func), repr(self._wrapper))

    #-----------------------------------------------------------------------------#
    # hyper-parameters management (delegate everything to the wrapper)
    #-----------------------------------------------------------------------------#    
    def get_hyperparams(self):
        """Return a dico with the hyperparameters of the wrapper"""
        return self._wrapper.get_hyperparams()

    def set_hyperparams(self, **hyperparams):
        """Return a dico with the hyperparameters """
        self._wrapper.set_hyperparams(**hyperparams)

    #-----------------------------------------------------------------------------#
    # Parameters management (delegate everything to the underlying function)
    #-----------------------------------------------------------------------------#
    def update_wraper_hyperparams(self):
        self._wrapper.update_dynamically(self._func)
    
    def print_param_names(self):
        """ Print the name of parameter expected """
        print(self._class_func._LIST_PARAMETERS)

    def fix_all(self):
        """ Fix all the parameters"""
        self._func.fix_all()

    def get_params(self, deep = True, bounds = True):
        """Get parameters (and bounds) of the underlying function"""
        return self._func.get_params(deep, bounds)


    def set_params(self, **params):
        """Set the parameters of the underlying function: triggers update of 
        the attributes of the wrapper """
        self._func.set_params(**params)
        self.update_wraper_hyperparams()
        
        
    @property
    def n_parameters(self):
        """Returns the number of non-fixed hyperparameters of the underlying function"""
        self._func.n_parameters
    
    def n_elements_one_param(self, param_name):
        """ return the number of elements of one parameter of the underlying function"""
        self._func.n_elements_one_param(param_name)


    ### Recall thetas are the free parameter values // recursive behavior by default 
    @property
    def n_theta(self):
        """Returns the number of non-fixed hyperparameters of the of the underlying function"""
        return self._func.n_theta
    
    def n_theta_one_param(self, param_name):
        """Get the number of free values for one parameter of the underlying function"""
        return self._func.n_theta_one_param(param_name)
        
    @property
    def theta(self):
        """Returns the (flattened) non-fixed hyperparameters of the underlying function """
        return self._func.theta  
    
    @theta.setter
    def theta(self, theta):
        """Sets the non-fixed hyperparameters of the underlying function"""
        self._func.theta = theta
        self.update_wraper_hyperparams()

                            
 
#==============================================================================
# Implementations
#==============================================================================   
# --------------------------------------------------------------------------- #
#   Wrappers
#       >> Owritten  Bounded  LTransformed
#
# TODO: May not need the wrapper workaround but could be implementent as pFunc
# e.g. owFunc with ow_X_min ow_X_max ow_Y as params
# e.g. sinEnveloppeFun is just FourrierFunc with one harmonic etc..
# Then need toassociate a symbol for composition
# --------------------------------------------------------------------------- #
class OWriter(pFunc_wrapper):
    """ Allow for overwritting results of a function when X belongs to certain
    values i.e. """
    def __init__(self, **args_wrapper):
        self._LIST_HYPERPARAMETERS += ['ow_X', 'ow_Y']
        self._NB_ELEM_PER_HYPERPARAMS += ['ow', 'ow']
        pFunc_wrapper.__init__(self, **args_wrapper)

    def update_dynamically(self, func):
        """ don't need to maintain attribute can do the bounding on the fly 
        in __call__() """
        pass

    def __call__(self, X, func, Y=None, eval_gradient=False):
        res_func = None
        ow_X = self._get_one_param("ow_X")
        ow_Y = self._get_one_param("ow_Y")
        if(ow_X is not None):
            for n, int_x in enumerate(ow_X):
                if ut.is_x_in_y(X, int_x):
                    res_func = ow_Y[n]
                    break

        if res_func is None:
            res_func = func(X, Y, eval_gradient)
        return res_func
    
class Bounder(pFunc_wrapper):
    """ Allow for bounding the results of a function   """
    def __init__(self, **args_wrapper):
        self._LIST_HYPERPARAMETERS += ['bounds_min', 'bounds_max']
        self._NB_ELEM_PER_HYPERPARAMS += [1,1]
        pFunc_wrapper.__init__(self, **args_wrapper)

    def update_dynamically(self, func):
        """ don't need to maintain attribute can do the bounding on the fly 
        in __call__() """
        pass

    def __call__(self, X, func, Y=None, eval_gradient=False):
        res_func = func(X, Y, eval_gradient)
        M = self._get_one_param("bounds_max")
        m = self._get_one_param("bounds_min")
        if (m is not None):
            if m > res_func:
                res_func = m
        if (M is not None):
            if M < res_func:
                res_func = M

        return res_func

class LTransformer(pFunc_wrapper):
    """ Allow to linearly transformed the results of a function   """
    def __init__(self, **args_wrapper):
        self._LIST_HYPERPARAMETERS += ['lt_type', 'lt_constraints']
        self._NB_ELEM_PER_HYPERPARAMS += [1,(1,2)]
        self._reset_lt()
        pFunc_wrapper.__init__(self, **args_wrapper)

    @property
    def n_constraints(self):
        ct = self._get_one_param("lt_constraints")
        if ct is None:
            return 0
        else:
            assert ut.is_iter(ct[0]), "wrong format"
            return len(ct)

    def _reset_lt(self):
        """ reset underlying scale and shift values"""
        self._lt_scale = 1
        self._lt_shift = 0

    def update_dynamically(self, func):
        """ update self._lt_scale and self._lt_shift based on func """  
        ct = self._get_one_param("lt_constraints")
        lt_type = self._get_one_param("lt_type")
        new_scale, new_shift = 1, 0

        
        if(self.n_constraints == 1):
            x0, y0 = ct[0]
            f0 = func(x0)
            if(lt_type == 'scale'):
                if (f0 == 0):
                    print('scale fixed to 1')
                    new_scale = 1 
                else:
                    new_scale = y0/f0
            elif(lt_type == 'shift'):
                new_shift = y0 - f0
            else:
                raise NotImplementedError()
                
        elif(self.n_constraints == 2):
            x0, y0 = ct[0]
            x1, y1 = ct[1]
            f0, f1 = func(x0), func(x1)
            if(lt_type == 'sin'):
                # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                assert (y0 == y1), "invalid constraints: y's shouldn't be the same"
                new_shift = y0
                new_scale = lambda t: np.sin(np.pi / (x1 - x0) * (t - x0))
            
            elif(lt_type == 'sin2'):
                # f'(t) = shift + sin(Pi * (x1 - x0) * (t-x0)) * f(t)
                assert (y0 == y1), "invalid constraints: y's shouldn't be the same"
                new_shift = y0
                new_shift = lambda t: np.square(np.sin(np.pi * (x1 - x0) * (t - x0)))
            
            elif(lt_type == 'scale&shift'):
                # f'(t) = shift + scale * f(t)
                if ((f1-f0) != 0):
                    new_scale = (y1-y0) / (f1-f0)
                else:
                    print('scale fixed to 1')
                    new_scale = 1
                new_shift = y0 - f0 * self._scale        
            else:
                raise NotImplementedError()

        elif(self.n_constraints > 2):
            raise NotImplementedError()
        self._lt_scale = new_scale
        self._lt_shift = new_shift

    def __call__(self, X, func, Y=None, eval_gradient=False):
        if(hasattr(self._lt_scale, '__call__')):
            scale = self._lt_scale(X)
        else:
            scale = self._lt_scale
        shift = self._lt_shift
        res_func = func(X, Y=Y, eval_gradient=eval_gradient)
        return res_func * scale + shift

class Enrober(OWriter, Bounder, LTransformer):
    """ Wrapper combining behavior of Bounder, LTransformer and OWriter. Mostly 
    define order in which underlying methods are sequentially applied: (1) linear
    -transformation 
    ... NOT REALLY NICE IMPLEM ... COULD BE MORE ...
    """
    def __init__(self, **args_wrapper):
        LTransformer.__init__(self, **args_wrapper)
        Bounder.__init__(self, **args_wrapper)
        OWriter.__init__(self, **args_wrapper)
        

    def update_dynamically(self, func):
        """ Update is only required for LTransformer"""
        LTransformer.update_dynamically(self, func)

    def __call__(self, X, func, Y=None, eval_gradient=False):
        func1 = lambda x, y, e: LTransformer.__call__(self, X=x,  func = func, Y=y, eval_gradient= e)
        func2 = lambda x, y, e: Bounder.__call__(self, X=x, func = func1, Y=y, eval_gradient= e)
        func3 = lambda x, y, e: OWriter.__call__(self, X=x, func = func2, Y=y, eval_gradient= e)
        res = func3(X, Y, eval_gradient)
        return res

# --------------------------------------------------------------------------- #
#   Collections
#       >> Product  Sum  Composition
# --------------------------------------------------------------------------- #
class Product(pFunc_collec):
    """Product of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) * f2(t) * c1(t) """
    

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.product(pFunc_collec.__call__(self, X, Y = Y, eval_gradient = eval_gradient))
        
    
class Sum(pFunc_collec):        
    """Sum of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) + f2(t) + c1(t) """
    
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.sum(pFunc_collec.__call__(self, X, Y = Y, eval_gradient = eval_gradient))
    
 
class Composition(pFunc_collec):
    """Composition of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(f2(c1(t)))) """
    
    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        list_pfunc = self._get_one_param('list_func')
        for f in list_pfunc:
            X = f(X, Y, eval_gradient=False)
        return X
        

# --------------------------------------------------------------------------- #
#   Base functions
#       >> IdFunc StepFunc ExpRamp SquareExp FourierFunc 
#       >> LinearFunc ConstantFunc ChebyshevFun
# --------------------------------------------------------------------------- #
class IdFunc(pFunc_base):
    """ params : {'F' = [f_1, .., f_N], 'Tstep' = [T_1, .., T_N], F0 = f_0}
    f(t) = (1) f_0 (if t<t_1)  (2) f_n (t in [t_(n-1), t_n[) (3) f_N (if t>=T_N) 
    """     
    _LIST_PARAMETERS = []
    _NB_ELEM_PER_PARAMS = []

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """ identity"""
        return X


class StepFunc(pFunc_base):
    """ params : {'F' = [f_1, .., f_N], 'Tstep' = [T_1, .., T_N], F0 = f_0}
    f(t) = (1) f_0 (if t<t_1)  (2) f_n (t in [t_(n-1), t_n[) (3) f_N (if t>=T_N) 
    """     
    _LIST_PARAMETERS = ['F', 'F0', 'Tstep']
    _NB_ELEM_PER_PARAMS = ['a', 1, 'a']

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        F, f0, Tstep = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        if(X < Tstep[0]):
            res = f0[0]
        else:
            idx = Tstep.searchsorted(X + 1e-15) - 1
            res = F[idx]
        return res


class ExpRampFunc(pFunc_base):
    """ params : {'a' = ymax, 'T':T, 'l' = l}
        f(t) = ampl * (1 - exp(t/ T * l)) / (1 - exp(l)) 
        ** The more r is the more convex it is
    """        
    _LIST_PARAMETERS = ['a', 'T', 'l']
    _NB_ELEM_PER_PARAMS = [1, 1, 1]
    

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        a, T, l = self.__a[0], self.__T[0], self.__l[0]
        res = a * (1 - np.exp(X / T * l)) / (1 - np.exp(l)) 
        return res


class SquareExponential(pFunc_base):
    """ params : {'a' = ymax, 'mu':T, 'l' = l} >> ymax^2 exp^{(x-mu)^2/l^2}
        TODO: Implement the multiDim + Kernel version // in eval and assert
    """
    pass


class FourierFunc(pFunc_base):
    """ params :{'A' = [a_1, .., a_N], 'B' = [b_1, .., b_N], 'omegas' = [om_0, 
        .., om_N], 'Phi' = [phi_0, phi_N], 'c0' = c0)
    >> f(t) = c0 + Sum(1..N) {a_i*cos(om_i t + phi_i) + b_i*cos(om_i t + phi_i)}
    """
    _LIST_PARAMETERS = ['A', 'B', 'om', 'phi', 'c0']
    _NB_ELEM_PER_PARAMS = ['a', 'a', 'a', 'a', 1]

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        A, B, om, phi, c0 = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        res = [A[i] * np.cos(omi * X + phi[i]) + B[i] * np.sin(omi * X + phi[i]) for i, omi in enumerate(om)]        
        res = (c0[0] + np.sum(res)) 
        return res


class LinearFunc(pFunc_base):
    """ params :{'bias':b, 'w'} >> f(t) = w.t + b
    """
    _LIST_PARAMETERS = ['w', 'bias']
    _NB_ELEM_PER_PARAMS = [1, 1]


    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        w, bias = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        res = bias[0] + w[0] * X
        return res    
    

class ConstantFunc(pFunc_base):
    """ params: {'c0':c0} >> f(t) = c0
    """
    _LIST_PARAMETERS = ['c0']
    _NB_ELEM_PER_PARAMS = [1]

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        return self._get_one_param('c0')[0]
    

class ChebyshevFunc(pFunc_base):
    """ params = {'c0', 'C' = [c_1, .., c_N], domain, window)
    >> f(t) = [c0 + Sum(1..N) {C_i* T_i(t)] where T_i is the Chebyshev polynomial 
        of order i defined using domain and window
    """
    _LIST_PARAMETERS = ['C', 'c0', 'domain', 'window']
    _NB_ELEM_PER_PARAMS = [None, 1, 2, 2]


    def _check_integrity(self):
        """ Additional feature: build and maintain the underlying numpy.polynomial.chebyshev
        """
        pFunc_base._check_integrity(self)
        C, c0, domain, window = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        coeffs = np.concatenate(([c0], C))
        self._chebFunc = cheb.Chebyshev(coeffs, domain, window)

    @ut.vectorise_method
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        res = self._chebFunc(X)
        return res





#==============================================================================
# Some testing
#==============================================================================
if __name__ == '__main__':

# --------------------------------------------------------------------------- #
#   Step Func
# --------------------------------------------------------------------------- #
    x = np.arange(0, 1,0.001)
    y = np.random.sample(len(x))
    f0 = 0  
    sf_dico = {'Tstep': x, 'F':y, 'F0':0, 'F0_bounds':(-10, 10), 'Tstep_bounds':(-10, 10)}
    sf1 = StepFunc(**sf_dico)
    sf1.plot_function(x)

    
# --------------------------------------------------------------------------- #
#   Fourier Function 
# --------------------------------------------------------------------------- #        
    
    # Build some Fourier functions
    T_four = 1
    nb_h = 2
    om = np.random.uniform(0, 4 * np.pi / T_four, nb_h)
    four_dico = {'A': [1,0], 'B':[0,0], 'c0':3, 'om':om, 'phi':np.zeros(nb_h)}
    ff1 = FourierFunc(**four_dico)

    # Test different functionalities
    ff1.plot_function(x)
    ff1.get_params()
    print(ff1.theta)
    repr(ff1)
    
    ff2 = ff1.clone()
    ff2.theta = np.array([1,0,0,0,2*np.pi, 4*np.pi, 0, 0,3])
    #set_params
    #set_theta
    ff2.plot_function(x)
    

# --------------------------------------------------------------------------- #
#    CRAB testing
# --------------------------------------------------------------------------- #
    T = 1
    x = np.arange(-1, 2, 0.01)
    xx =np.arange(0, 1, 0.01)
    om = [2 * np.pi/T, 4 * np.pi/T]
    A = np.random.sample(2)
    B = np.random.sample(2)
    dico_f2h = {'om':om, 'A': A, 'B': B, 'phi':[0,0], 'c0':0,'om_bounds':False, 
    'A_bounds': (-1,1), 'B_bounds': (-1,1), 'phi_bounds':False, 'c0_bounds':False}
    dico_linear = {'w':1, 'bias':0, 'fix_all': True}
    dico_sin_wrap = {'lt_type':'sin', 'lt_constraints':[[0,0], [T,0]]}
    dico_oa_wrap = {'bounds_min':0, 'bounds_max':1, 'ow_Y':[0, 1], 'ow_X':[(-100,0),(1, 100)]}
           
    f2h = FourierFunc(**dico_f2h)
    guess = LinearFunc(**dico_linear) 
    oa_wer = Enrober(**dico_oa_wrap)
    sin_wer = Enrober(**dico_sin_wrap)

    better = oa_wer * (guess * (1 + (sin_wer * f2h)))

    
    better2 = better.clone()
    better2.theta = [1,1,0,0]

    better.plot_function(xx)    
    better2.plot_function(xx)

    
