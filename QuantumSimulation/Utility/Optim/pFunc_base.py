#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:18 2018

@author: fred
Inspired by sklearn.gaussian_process.kernels
"""

if(__name__ == '__main__'):
    import sys
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
else:
    from .. import Helper as ut
    
import numpy as np
from numpy import array, inf #used for eval within the module
import numpy.polynomial.chebyshev as cheb
import matplotlib.pylab as plt
import pdb 
import logging
logger = logging.getLogger(__name__)



#==============================================================================
# parametrized function base classes
#==============================================================================
class pFunc_base():
    """ Abstract class for non-nested parametric functions. Still potential nested 
    behavior is implemented in the methods but fully exploited in the collection subclass 
    (in order not to re-implement the same methods in these subclasses). Nested 
    behavior is flagged by 'deep' parameter or the check of '_FLAG_TYPE' ('base'
    for non nested functions and 'collection' for potentially nested structure)
        
        param: an array of elements (in general numerics but still can accomodate non
                numeric elements .. not advised though)
        
        param_bounds: represent the bounds associated to the elements of param
                It is thus a list of the same length with elements which can be:
                **2-uplet**: in this case (min, max), **False**: means the element
                is fixed if **True** is passed it will result in _DEF_BOUNDS
                or **None** if param element is free and doesn't have boundaries
            i.e. [(-2, 5), False, _DEF_BOUNDS, None] 
        
        theta: flat np.array of the values of the non-fixed elements of the (potentially
                several parameters)
    
        theta_bounds: list of bounds associated to theta

    e.g. for a two parameters ('weights' and 'bias') function with bias fixed 
    the structure is the following (notice everything is stored as a list/ array
    even when )
    self._LIST_PARAMS = ['weights', 'bias']
    self.__weights = np.array([3, 6])
    self.__bias = np.array([1])
    self.__weights_bounds = [(1e-5, 1000), (1e-5, 1000)]
    self.__bias = [False]
    
    TODO:
        + implement power
        + Finish using name (i.e. get attribute and get by name) s.t. 
        it appears in repr + probably use a flag use_name 
        + use of different default bounds (e.g. _DEF_POS_BOUNDS) - maybe not
        + test (D-CRAB setups)
        + Bug when try to update theta (in the case it's an array of integer) with a float
         >> enforce type (same way as bounds or parameters)
        + default paarmeters for functions
        + ENFORCE TYPE
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
                self._setup_params_and_bounds(p, args_func[p], args_func.get(p+'_bounds', False))
        if(args_func.get('name') is not None):
            self._name = args_func.get('name')
        else:
            name_func = str(self.__class__).split('.')[-1].split("'")[0]
            self._name =name_func + str(np.random.randint(0, 1e6))

        self._check_integrity()
    
#    def __hash__(self):
#        return id(self)
        
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
        """ process and store parameters and their bounds"""
        vals_processed = self._process_vals(param_name, vals)
        setattr(self, '__'+param_name, vals_processed)
        bounds_processed = self._process_bounds(param_name, bounds)
        setattr(self, '__'+param_name+'_bounds', bounds_processed)

    def _process_vals(self, param_name, vals):
        """ just ensure that cal is a list  (if single value passed return 
        a list)
        """
        if(not(ut.is_iter(vals))):
            vals = [vals]
        return array(vals)
        
    def _process_bounds(self, param_name, bounds):
        """ Process the bounds associated to a parameter (with l elements).
        bounds can be provided as :
            + a list with l  distinct individual bounds
            + an individual bound (which will be duplicated to form a list
            with l elements)
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
                    if(val<bounds[n][0]): 
                        logger.warning('Value %s of parameter %s doesnt comply'
                                ' with bounds %s ==> OWritten' % (n, param_name, str(bounds[n])))
                        param[n] = bounds[n][0]
                    if(val>bounds[n][1]):
                        logger.warning('Value %s of parameter %s doesnt comply'
                                ' with bounds %s ==> OWritten' % (n, param_name, str(bounds[n])))
                        param[n] = bounds[n][1]
    
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
                    params.update({'f' + str(n) + '__' + key: val for key, val in sub_params.items()})
                    
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
    def name(self):
        return self._name
    @property
    def all_names(self):
        return self._name
    
    def get_func_by_name(self, name):
        """ return a pFunc if found, None if not found"""
        if(name == self.name):
            res = self
        else:
            res = None
        return res

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
                    params[p_name][mask_free] = np.array(theta[i:i + nb_to_update])
                    i += nb_to_update

            #That's wehere the updates really happen
            self.set_params(**params)

        elif(self._FLAG_TYPE == 'collection'):
            #Delegate updates to subobj
            for sub in self.list_func_free:
                nb_to_update_sub = sub.n_theta
                if(nb_to_update_sub > 0):
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
   
    def randomized_theta(self):
        """ randomize (uniformly) theta values, based on their bounds"""
        theta_bounds = self.theta
        theta_random = [random_from_bound(b) for b in theta_bounds]
        self.theta = theta_random
        
    

    #-----------------------------------------------------------------------------#
    # standard operations between pFunc
    #-----------------------------------------------------------------------------#
    def __add__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False)
            return Sum([self, bFunc])
        return Sum([self, b])
    
    def __sub__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b) * (-1)
            else:
                bFunc = ConstantFunc(c0 = -b, c0_bounds = False)
            return Sum([self, bFunc])
        return Sum([self, b])

    def __radd__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False)
            return Sum([bFunc, self])
        return Sum([b, self])
    
    def __mul__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False)
        elif( isinstance(b, pFunc_wrapper)):
            raise SystemError("{1} not allowed to act on  pFunc_wrapper on the" 
                "left".format(self.__class__))
        else:
            bFunc = b
        return Product([self, bFunc])

    def __rmul__(self, b):
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False)
            res = Product([bFunc, self])

        if(isinstance(b, pFunc_wrapper)):
            res = Composition([b, self])            
        else:
            res = Product([b, self])

        return res

    def __pow__(self, b):
        if(type(b) in [int, float]):
            bFunc = PowerFunc(power=b)
            res = Composition([bFunc, self])
        else:
            raise SystemError("{1} not allowed with power operations" .format(type(b)))
        return res

#    def __eq__(self, b):
#        """ doesn't deal well with the type pFunc_fromcllable"""
#        if type(self) != type(b):
#            return False
#        params_a = self.get_params()
#        params_b = b.get_params()
#        for key in set(list(params_a.keys()) + list(params_b.keys())):
#            if np.any(params_a.get(key, None) != params_b.get(key, None)):
#                return False
#        return True

    def __repr__(self):
        return "{0}(**{1})".format(self.__class__.__name__, repr(self.get_params(deep = False)))                             

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        
    def clone(self, theta = None):
        """Returns a clone of self with given hyperparameters theta. """
        cloned = eval(repr(self))
        if(theta is not None):
            cloned.theta = theta
        return cloned
    
    @classmethod
    def build_pfunc(cls, representation):
        """ Build a function from a str / dico / potentially add others """
        if ut.is_str(representation):
            try:
                func = eval(representation)
            except:
                bf = 'cls.build_pfunc('
                af = ')'
                st = ut.parse_enclose_with_counter(representation , before = bf, after = af)
                func = eval(st)
                
        elif ut.is_dico(representation):
            name_func = representation['name_func']
            func = eval(name_func)(**representation)
            
        else:
            raise SystemError("build_custom_func can build a function from an "
                              "object  of tye {0}".format(cls.__class__))
    
        return func

    #-----------------------------------------------------------------------------#
    # Some extra capabilities
    #     Visualization >> plot_func
    #        >> fluence / smoothness / 
    #     Fitting >> fit_to_target_func
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
    def square_dist(func1, func2, range_x):
        """ avg_x (func1(x) - func2(x))^2 """
        res = np.average(np.square(np.squeeze(func1(range_x)) - np.squeeze(func2(range_x))))
        return res
    
    def square_dist_to(self, func1, range_x):
        return pFunc_base.square_dist(self, func1, range_x)


    @staticmethod
    def l2_norm(func1, range_x):
        """ """
        raise NotImplementedError()
    
    @staticmethod
    def l2_dist_list(list_func, range_x, func_ref = None):
        """ Compute the L2 distance of a list of functions to the first one
        distance = 1/T * sum [f(t) - g(t)]^2 dt * normalization
        (last time is not taken into account)
        """
        if(func_ref is None):
            func_ref = list_func[0]
            
        l2_dists = [pFunc_base.square_dist(func_ref, f) for f in list_func] 
        return l2_dists
    
    def fit_to_target_func(self, func_target, list_x, algo = 'DE', copy = False, **args_optim):
        """ fit the free parameters of the functionn to match (in the Least 
        Square Error sense)
        TODO: gradient based methods """
        if (copy):
            func_to_fit = self.clone()
        else:
            func_to_fit = self
        return_fit = args_optim.pop('return_fit', False)
        bounds = func_to_fit.theta_bounds
        if not(_is_fully_bounded(bounds)):
            SystemError("Can't fit with unknown bounds")
            
        def SquareErrors(params):
            func_to_fit.theta = params
            res = pFunc_base.square_dist(func_to_fit, func_target, list_x)
            return res
        
        if(algo == 'DE'):
            import scipy.optimize
            resDE =  scipy.optimize.differential_evolution(SquareErrors, bounds, **args_optim)
        else:
            raise NotImplementedError()
        fit = SquareErrors(resDE['x'])
        logger.info(fit)
        if(return_fit):
            return func_to_fit, fit
        else:
            return func_to_fit
                
    #-----------------------------------------------------------------------------#
    # I/O
    #-----------------------------------------------------------------------------#
    def save_to_file(self, file_name):
        pFunc_base.save_a_func_to_file(self, file_name, 'w')
    
    @staticmethod
    def save_a_func_to_file(func, file_name, type_write = 'w'):
        ut.write_to_file_for_eval(func, file_name, type_write)
    
    @staticmethod
    def read_func_from_file(file_name):
        evaluator = lambda ev: eval(ev)    
        func1 = ut.eval_from_file(file_name, evfunc = evaluator)
        return func1

class pFunc_fromcallable(pFunc_base):
    """ wrap a callable object (function, lambda function) to have the (min)
    capabilities of pFunc"""
    _FLAG_TYPE = 'callable'
    def __init__(self, callable_obj):
        pFunc_base.__init__(self)
        self._callable = callable_obj

    
    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        res = self._callable(X)
        return res

    def __repr__(self):
        return "{0}(callable_obj = {1})".format(self.__class__.__name__, repr(self._callable)) 


class pFunc_wrapper(pFunc_base):
    """ Rules for wrapper:
        (1) * as a Composition rather than a product
        (2) can only be at the right """
    def __mul__(self, b):
        """ Composition instead of product """
        if not isinstance(b, pFunc_base):
            if(ut.is_callable(b)):
                bFunc = pFunc_fromcallable(b)
            else:
                bFunc = ConstantFunc(c0 = b, c0_bounds = False)
        elif(not isinstance(b, self.__class__)):
            bFunc = b
        else:
            raise SystemError("pFun_wrapper can't act on the left (by *)"
                " of a {1} object".format(b.__class__))

        return Composition(list_func = [self, bFunc])

    def __rmul__(self, b):
        """ more exactly it should not happen"""
        raise SystemError("pFun_wrapper can't act on the right (by *)")

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
    """
    _FLAG_TYPE = 'collection'
    _LIST_PARAMETERS = ['list_func']
    
    def __init__(self, list_func, list_func_bounds = True, **args_func):
        self._setup_params_and_bounds('list_func', list_func, list_func_bounds)
        if(args_func.get('name') is not None):
            self._name = args_func.get('name')
        else:
            name_func = str(self.__class__).split('.')[-1].split("'")[0]
            self._name =name_func + str(np.random.randint(0, 1e6))
        
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

    @property
    def name(self):
        return self._name
    
    @property
    def all_names(self):
        names = [self.name]
        for f in self.list_func:
            name_tmp = f.all_names
            if(ut.is_list(name_tmp)):
                names += name_tmp
            else:
                names.append(name_tmp)
        
        return names
    
    def get_func_by_name(self, name):
        """ return a pFunc if found, None if not found"""
        if(name == self.name):
            res = self
        else:
            sub = [f.get_func_by_name(name) for f in self.list_func]
            found = [s is not None for s in sub]
            n_found = np.sum(found)
            if(n_found == 0):
                res = None
            elif(n_found == 1):
                res = sub[found.index(True)]
            else:
                raise SystemError('{0} found {1} times in {2}'.format(name, n_found, self.name))
        return res

    # ----------------------------------------------------------------------- #
    #   I/O
    # ----------------------------------------------------------------------- #     
    @ut.extend_dim_method(0, True)
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

class pFunc_List(pFunc_collec):
    def __init__(self, list_func, list_func_bounds = True, **args_func):
        pFunc_collec.__init__(self, list_func, list_func_bounds = True, **args_func)
        self.__init_counter = 0
        
    def __iter__(self):
        """ iterable capability"""
        i = self.__init_counter
        while i < self.n_func:
            yield self.get_function(i)
            i+=1

    def __getitem__(self, index):
        """ index capability"""
        return self.get_function(index)
    
    def __len__(self):
        return self.n_func

#==============================================================================
# Implementations
#==============================================================================       
# --------------------------------------------------------------------------- #
#   Collections
#       >> Product  Sum  Composition
# --------------------------------------------------------------------------- #
class Product(pFunc_collec):
    """Product of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) * f2(t) * c1(t) """
    

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.product(pFunc_collec.__call__(self, X, Y = Y, eval_gradient = eval_gradient))
    
class Sum(pFunc_collec):        
    """Sum of several pFuncs (<pFunc_base> or <pFunc_collec>) 
    [f1, f2, c1](t) >> f1(t) + f2(t) + c1(t) """
    
    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function."""
        return np.sum(pFunc_collec.__call__(self, X, Y = Y, eval_gradient = eval_gradient))
 
class Composition(pFunc_collec):
    """Composition of several pFuncs (<pFunc_base> or <pFunc_collec>)
    g = Composition(list_func = [f1, f2, f3])
    >>> g(x) = f1(f2(f3(x)))) """
    
    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the function by starting from the last function of the list
        Y is used here to keep track of the input (which is otherwised lost)
        this feature (workaround) is used in OWYWrapper.. Maybe Y should be renamed 
        X0"""
        list_pfunc = self._get_one_param('list_func')
        if(Y is None):
            Y =X
        for f in reversed(list_pfunc):
            X = f(X, Y=Y, eval_gradient=False)
        return X
        

# --------------------------------------------------------------------------- #
#   Base 
#       >> IdFunc StepFunc ExpRamp SquareExp FourierFunc 
#       >> LinearFunc ConstantFunc ChebyshevFun
# --------------------------------------------------------------------------- #
class IdFunc(pFunc_base):
    """ params : {}
    >> f(x) = x
    """     
    _LIST_PARAMETERS = []
    _NB_ELEM_PER_PARAMS = []

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """ identity"""
        return X

class PowerFunc(pFunc_base):
    """ PowerFunc(power=p)
    >> f(x) = x ** p 
    """
    _LIST_PARAMETERS = ['power']
    _NB_ELEM_PER_PARAMS = [1]

    def __call__(self, X, Y=None, eval_gradient=False):
        """ identity"""
        return X ** self._get_one_param('power')[0]

class StepFunc(pFunc_base):
    """ params : {'F' = [f_1, .., f_N], 'Tstep' = [T_1, .., T_N], F0 = f_0}
    >> f(t) = (1) f_0 (if t<t_1)  (2) f_n (t in [t_(n-1), t_n[) (3) f_N (if t>=T_N) 
    """     
    _LIST_PARAMETERS = ['F', 'F0', 'Tstep']
    _NB_ELEM_PER_PARAMS = ['a', 1, 'a']

    @ut.extend_dim_method(0, True)
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
    
class PWL(pFunc_base):
    """ piecewiselinear
    params : {'F' = [f_1, .., f_N], 'Tstep' = [T_1, .., T_N], F0 = f_0}
    >> f(t) = (1) f_0 (if t<t_0)  (2) f_(n-1) + (f_(n) - f_(n-1)) * (t - t_(n-1))  
    (t in [t_(n-1), t_n[) (3) f_N (if t>=T_N) 
    """     
    _LIST_PARAMETERS = ['F', 'T']
    _NB_ELEM_PER_PARAMS = ['a', 'a']

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        F, T= (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        if(X < T[0]):
            res = F[0]
        elif(X >= T[-1]):
            res = F[-1]
        else:
            idx = T.searchsorted(X + 1e-15) - 1
            res = F[idx] + (F[idx+1] - F[idx]) * (X - T[idx])/(T[idx+1] - T[idx])
        return res

class ExpRampFunc(pFunc_base):
    """ params : {'a' = ymax, 'T':T, 'l' = l}
        f(t) = ampl * (1 - exp(t/ T * l)) / (1 - exp(l)) 
        ** The more r is the more convex it is
    """        
    _LIST_PARAMETERS = ['a', 'T', 'l']
    _NB_ELEM_PER_PARAMS = [1, 1, 1]
    

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        a, T, l = self.__a[0], self.__T[0], self.__l[0]
        res = a * (1 - np.exp(X / T * l)) / (1 - np.exp(l)) 
        return res

class GRBFFunc(pFunc_base):
    """ Gaussian Radial Basis Functions params : {'A':, 'x0':, 'l' = l} 
    >> f(x) = sum A[k] exp^{(x-x0[k])^2/(2*l^2)} """
    _LIST_PARAMETERS = ['A', 'x0', 'l']
    _NB_ELEM_PER_PARAMS = ['a', 'a', 'a']
    
    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        A, x0, l = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        res = np.dot(A, np.exp(-np.square(x0 - X)/(2*np.square(l))))
        return res


class FourierFunc(pFunc_base):
    """ params :{'A' = [a_1, .., a_N], 'B' = [b_1, .., b_N], 'omegas' = [om_0, 
        .., om_N], 'Phi' = [phi_0, phi_N], 'c0' = c0)
    >> f(t) = c0 + Sum(1..N) {a_i*cos(om_i t + phi_i) + b_i*cos(om_i t + phi_i)}
    """
    _LIST_PARAMETERS = ['A', 'B', 'Om', 'phi', 'c0']
    _NB_ELEM_PER_PARAMS = ['a', 'a', 'a', 'a', 1]

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        A, B, Om, phi, c0 = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        res = [A[i] * np.cos(omi * X + phi[i]) + B[i] * np.sin(omi * X + phi[i]) for i, omi in enumerate(Om)]        
        res = (c0[0] + np.sum(res)) 
        return res


class LinearFunc(pFunc_base):
    """ params :{'bias':b, 'w'} >> f(t) = w.t + b
    """
    _LIST_PARAMETERS = ['w', 'bias']
    _NB_ELEM_PER_PARAMS = [1, 1]


    @ut.extend_dim_method(0, True)
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

    @ut.extend_dim_method(0, True)
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

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        res = self._chebFunc(X)
        return res

class LogisticFunc(pFunc_base):
    """ params = {'L':1, 'k':1, 'x0':0)
    >> f(t) = L / (1 + e^{-k(x-x0)})
    """
    _LIST_PARAMETERS = ['L', 'x0', 'k']
    _NB_ELEM_PER_PARAMS = [1, 1, 1]

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        """         
        """
        L, x0, k = (self._get_one_param(p) for p in self._LIST_PARAMETERS)
        res = L[0]/(1+np.exp(-k[0]*(X-x0[0])))
        return res


# --------------------------------------------------------------------------- #
#   Wrapper (cf. pFunc_wrapper docstring)
#       >> OWriterYWrap BoundWrap 
# --------------------------------------------------------------------------- #
class OwriterYWrap(pFunc_wrapper):
    """ Carefull there is a little bit of workaround here and names of variable may be confusing..
    
    It's meant to be used as OWriterYFunc1() * Func2() >> Func3() with:
    Func3(input) =
     + output_ow[l] when input in [input_min[l], input_max[l]] 
     + Func2(input) o.w.
    """
    _LIST_PARAMETERS = ['input_min', 'input_max', 'output_ow']
    _NB_ELEM_PER_PARAMS = ['ow', 'ow', 'ow']

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y, eval_gradient=False):
        """ Carefull (X is understood as ouput Y as input)"""
        res_func = X
        input_min = self._get_one_param("input_min")
        input_max = self._get_one_param("input_max")
        ow = self._get_one_param("output_ow")
        
        if(input_min is not None):
            for n, i_min in enumerate(input_min):
                if (Y >= i_min) and (Y<=input_max[n]):
                    res_func = ow[n]
                    break

        return res_func


class BoundWrap(pFunc_wrapper):
    """ Allow for bounding the results of a function   """
    _LIST_PARAMETERS = ['bounds_min', 'bounds_max']
    _NB_ELEM_PER_PARAMS = [1,1]

    @ut.extend_dim_method(0, True)
    def __call__(self, X, Y=None, eval_gradient=False):
        M = self._get_one_param("bounds_max")[0]
        m = self._get_one_param("bounds_min")[0]
        return np.clip(X, m, M)

def eval_with_pFunc(string):
    """ eval benifying from the classes/methods of this module"""
    return eval(string)

def _is_fully_bounded(list_bounds):
    res = np.all([_is_a_strict_bound(b) for b in list_bounds])
    return res
    
def _is_a_strict_bound(bound):
    res = hasattr(bound, '__iter__') and (len(bound) == 2) and (bound[0] < bound[1])
    return res

def random_from_bound(bound):
    """ return a random value if a tuple is provided else 0 """
    if (isinstance(bound, tuple)):
        val = np.random.uniform(low = bound[0], high = bound[1])
    else:
        val = 0.0
    return val
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
    Om = np.random.uniform(0, 4 * np.pi / T_four, nb_h)
    four_dico = {'A': [1,0], 'B':[0,0], 'c0':3, 'Om':Om, 'phi':np.zeros(nb_h),
                 'A_bounds': None,'B_bounds': None,'c0_bounds': None,
                 'Om_bounds': None,'phi_bounds': None}
    ff1 = FourierFunc(**four_dico)

    # Test different functionalities
    ff1.plot_function(x)
    ff1.get_params()
    print(ff1.theta)
    repr(ff1)
    
    ff2 = ff1.clone()
    ff2.theta = np.array([1,0,0,0,2*np.pi, 4*np.pi, 0, 0,1])
    #set_params
    #set_theta
    ff2.plot_function(x)
    

# --------------------------------------------------------------------------- #
#    CRAB testing
# fullFun = Wrap(guess *(1 + scaling() ) )
# --------------------------------------------------------------------------- #
    T = 1
    x = np.arange(-1, 2, 0.01)
    xx =np.arange(-0.02, 1.02, 0.01)
    om_ref = 2 * np.pi / T
    Om = [2 * np.pi/T, 4 * np.pi/T]
    A = np.random.sample(2)
    B = np.random.sample(2)
    
    # Only function with free params
    dico_f2h = {'Om':Om, 'A': A, 'B': B, 'phi':[0,0], 'c0':0,'Om_bounds':False, 
    'A_bounds': (-1,1), 'B_bounds': (-1,1), 'phi_bounds':False, 'c0_bounds':False, 'name':'f2h'}
    dico_linear = {'w':1, 'bias':0, 'name':'guess'}
    dico_sin_scaling = {'Om':[om_ref/2], 'A': [0], 'B': [1], 'phi':[0],'c0':0 , 'name':'scaling'}
    dico_bound ={'bounds_min':0, 'bounds_max':1,'name':'bound'}
    dico_ow = {'input_min':[-100, 1], 'input_max':[0, 100], 'output_ow':[0,1], 'name':'ow'}
           
    
    f2h = FourierFunc(**dico_f2h)
    guess = LinearFunc(**dico_linear) 
    sin_scaling = FourierFunc(**dico_sin_scaling)
    bound = BoundWrap(**dico_bound)
    ow = OwriterYWrap(**dico_ow)
    better = ow *(bound* (guess * (1 + (sin_scaling * f2h))))
    
    better2 = better.clone()
    better2.theta
    better2.theta = np.array([1,1,1,1])    

    better.plot_function(x)    
    better2.plot_function(x)

    print(better2.get_params())
    print(better2.theta)
    

    one_name_func = better.all_names[-1]
    better.get_func_by_name(one_name_func)

# --------------------------------------------------------------------------- #
#    Logistic
# --------------------------------------------------------------------------- #
    

    # Only function with free params
    T = 1
    N_params = 2
    A = np.random.uniform(-1, 1, N_params)
    B = np.random.uniform(-1, 1, N_params)
    phi = np.zeros(N_params) 
    om_ref = 2 * np.pi / T 
    Om = np.arange(1, N_params +1) * om_ref
    
    dico_f2h = {'Om':Om, 'A': A, 'B': B, 'phi':phi, 'c0':0,'Om_bounds':False, 
    'A_bounds': (-1,1), 'B_bounds': (-1,1), 'phi_bounds':False, 'c0_bounds':False, 'name':'f2h'}
    dico_linear = {'w':1/T, 'bias':0, 'name':'guess'}
    dico_sin_scaling = {'Om':[om_ref/2], 'A': [0], 'B': [1], 'phi':[0],'c0':0 , 'name':'scaling'}
    dico_bound ={'bounds_min':0, 'bounds_max':1,'name':'bound'}
    dico_ow = {'input_min':[-100, 1], 'input_max':[0, 100], 'output_ow':[0,1], 'name':'ow'}
    dico_logis = {'L':3,'k':2/N_params,'x0':0}

    
    
    f2h = FourierFunc(**dico_f2h)
    guess = LinearFunc(**dico_linear)
    sin_scaling = FourierFunc(**dico_sin_scaling)
    sqrt_sin_scaling = sin_scaling ** 0.5
    bound = BoundWrap(**dico_bound)
    ow = OwriterYWrap(**dico_ow)
    logis = LogisticFunc(**dico_logis)
    #f2h.plot_function(xx)
    #(guess * (1 + sin_scaling**2 *Composition(list_func =[logis-1, f2h]))).plot_function(xx)
    

    better = ow *(bound* (guess * (1 + (sin_scaling**1.5 * Composition(list_func =[logis-1, f2h])))))
    better.plot_function(xx)
    guess.plot_function(xx)
    
    
    # FredParam
    T = 1
    N_params = 2
    A = np.random.uniform(-1, 1, N_params)
    B = np.random.uniform(-1, 1, N_params)
    phi = np.zeros(N_params) 
    om_ref = 2 * np.pi / T 
    Om = np.arange(1, N_params +1) * om_ref
    
    dico_f2h = {'Om':Om, 'A': A, 'B': B, 'phi':phi, 'c0':0,'Om_bounds':False, 
    'A_bounds': (-1,1), 'B_bounds': (-1,1), 'phi_bounds':False, 'c0_bounds':False, 'name':'f2h'}
    dico_linear = {'w':1/T, 'bias':0, 'name':'guess'}
    dico_sin_scaling = {'Om':[om_ref/2], 'A': [0], 'B': [1], 'phi':[0],'c0':0 , 'name':'scaling'}
    dico_bound ={'bounds_min':0, 'bounds_max':1,'name':'bound'}
    dico_ow = {'input_min':[-100, 1], 'input_max':[0, 100], 'output_ow':[0,1], 'name':'ow'}
    dico_logis = {'L':2,'k':4/N_params,'x0':0}

    f2h = FourierFunc(**dico_f2h)
    guess = LinearFunc(**dico_linear)
    sin_scaling = FourierFunc(**dico_sin_scaling) ** 0.5
    bound = BoundWrap(**dico_bound)
    ow = OwriterYWrap(**dico_ow)
    logis = LogisticFunc(**dico_logis)
    #(logis-1).plot_function(np.arange(-6,6))
    #f2h.plot_function(xx)
    #(guess * (1 + sin_scaling**2 *Composition(list_func =[logis-1, f2h]))).plot_function(xx)
    #Composition(list_func =[logis-1, f2h]).plot_function(xx)

    better = ow *(bound* (guess + ((sin_scaling**0.5) *0.5 * Composition(list_func =[logis-1, f2h]))))
    better.plot_function(xx)
    guess.plot_function(xx)
    
    
    #RBF
    T = 10
    tt = np.linspace(0, T, 1000)
    nb_P = 4
    a_scale = np.sqrt(8* np.log(2))
    b_scale = np.sqrt(8* np.log(4))
    sigma = T/(2 * b_scale + (nb_P-1) * a_scale)
    A = np.random.uniform(-1,1, nb_P)#np.repeat(1, nb_P)
    x0 = b_scale * sigma + np.arange(nb_P) * a_scale * sigma
    l = np.repeat(sigma, nb_P)
    
    RBF = GRBFFunc(l=l, x0=x0, A = A)
    RBF.plot_function(tt)
    
    #logisticmask
    x0 = 0.1
    k = 6 / x0
     
    #Log2
    log1 = (LogisticFunc(x0= (1-x0)*T , L = 1, k=-k/T))
    log1.plot_function(tt)


    # --------------------------------------------------------------------------- #
    #   PWL
    # --------------------------------------------------------------------------- #
    x = np.arange(0.1, 1.1,0.1)
    x_plot = np.arange(-0.1, 1.2,0.1)
    y = np.random.sample(len(x))
    f0 = 0  
    x0 = 0
    sf_dico = {'Tstep': x, 'F':y, 'F0':f0, 'T0':x0, 'F0_bounds':(-10, 10), 'Tstep_bounds':(-10, 10)}
    sf1 = PWL(**sf_dico)
    sf1.plot_function(x_plot)
    