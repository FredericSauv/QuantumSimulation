#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:18 2018

@author: fred
"""
import logging 
logger = logging.getLogger(__name__)
if(__name__ == '__main__'):
    import sys
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    from QuantumSimulation.Utility.Optim.pFunc_base import Composition, Product, Sum
    from QuantumSimulation.Utility.Misc.RandomGenerator import RandomGenerator
else:
    from .. import Helper as ut
    from . import pFunc_base as pf
    from .pFunc_base import Composition, Product, Sum # needed
    from ..Misc.RandomGenerator import RandomGenerator
    
from collections import OrderedDict
import numpy as np
from numpy import array, inf
import copy, pdb

class pFunc_factory():
    """Create on demand parametrized functions with complicated / ad-hoc patterns
        Enhance pFunc_base by providing for more flexibility, functionality 
        when building the function

        Two ways are provided:
        * by passing a dict: with at leadt an entry (key = 'name_func' ,value=<str>)
          other entries with the parameters of the function
          custom functions implemented can be found in _LIST_CUSTOM_FUNC

        * string

    EXAMPLE
    -------

    TODO
    ----
    REMOVE PARSING ZONE // USE DIRECTLY THE PARSER CLASS 
    BETTER DOC STRINGS

    """
    # Custom functions informations
    # <str>key: (<str>:infos, <class>:constructor, <list<list<str>>>: params mandatory, <list<str>>: params optional)



        


    def __init__(self, rdm_obj = None, context = None):
        """ 

        PARAMETER
        ---------
            rdm_gen: RandomState
            contest: dict 
                mapping variable name (key) to variable value 

        Attributes
        ----------
        self._LIST_CUSTOM_FUNC: list of the custom implemenetations provided 
            dict(<str> name_func, <obj> infos)


        """
        self.rdm_gen = rdm_obj
        self.context = context

        #         # With infos = (<str> explicit name, <method> constructor, )
        _LIST_CUSTOM_FUNC = {} 
        # Custom implementation
        _LIST_CUSTOM_FUNC['PWC'] = ('Piece Wise Constant', self._custom_PWC)
        _LIST_CUSTOM_FUNC['PWL'] = ('Piece Wise Linear', self._custom_PWL)
        _LIST_CUSTOM_FUNC['FourierFunc'] = ('Fourier basis', self._custom_FourierFunc)
        _LIST_CUSTOM_FUNC['OwriterYWrap']= ('wrapper: overwritting', self._custom_OW, ['ow'], [])
         

    @classmethod
    def help(cls, name_func = None):
        if(name_func is None):
            print ("list of available func is {}".format(cls._LIST_CUSTOM_FUNC.keys()))
        else:
            info = cls._LIST_CUSTOM_FUNC[name_func]
            print(info)
            print ("for name_func = {0},\n ".format(str(info[0])))
            print ("underlying func = {0}\n".format(str(info[1])))
            print ("init params are {0} \n".format(str(info[3])))
            print ("or at leats: {0} \n".format(str(info[2])))
               
    @property
    def rdm_gen(self):
        return self._rdm_gen
    
    @rdm_gen.setter
    def rdm_gen(self, rdm_obj):
        self._rdm_gen = RandomGenerator.init_random_generator(rdm_obj)

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, context):
        if(context is None):
            self._context = {}
        elif(ut.is_dico(context)):
            self._context = context
        else:
            SystemError('context should be a dictionnary')
            
    def update_context(self, more_context):
        if(ut.is_dico(more_context)):
            self._context.update(more_context)
        else:
            SystemError('context should be a dictionnary')

    #-------------------------------------------------------#
    # MAIN ENTRY POINT
    # create pFunc
    #-------------------------------------------------------#
    def create_pfunc(self, fun_obj, **extra_args)
        if(ut.is_str(fun_obj)):
            func = self.from_str(fun_obj, **extra_args)
        elif(ut.is_dico(fun_obj)):
            func = self.from_dict(fun_obj, **extra_args)
        else:
            raise SystemError("create_pfunc: type of fun_obj not recognized")
        return func


    #-------------------------------------------------------#
    # SHORTCUT MANAGEMENT
    # transform a shortcut string into a pFunc
    #-------------------------------------------------------#
    def from_str(self, shortcut, **extra_args):
        """

        Example
        -------

        """
        full_expr = Parser.extend_shortcut(shortcut)
        func = self.eval_string(full_expr, **extra_args)
        return func



    #-------------------------------------------------------#
    #                    EVALUATING ZONE
    #-------------------------------------------------------#
    def eval_string(self, string, context = None):
        """ eval a string in this environment 
        
        Example
        -----
        """
        if context is None:
            context = self.context
        return eval(string, None, locals().update(context))


    
    #-------------------------------------------------------#
    # FROM A DICTIONARY
    # Helps to build function from a dico
    #-------------------------------------------------------#
    def from_dict(self, dico_fun):
        """ built a pFunc_base based on a dict specifying
            + key = 'name_func' which (custom) funcs should be used 
            + key = 'paramsXXX' params necessary to generate the function
                     the necessary/optional params can be found in the code below 
                     or should be indicated in _LIST_CUSTOM_FUNC provided it is maintained
            + key = 'paramsXXX_bounds' (optional) by default they will be set to False
                    i.e. the params are frozen (cf. docstring of pFunc_base)

        Example
        -------
        Default (i.e. where no special implementation has beeen provided):


        Custom (i.e. some custom rules have been implemented):
        self.from_dict({'name_func':'PWL', 'nb_steps':5, 'F_bounds':(0,1)})
        """
        name_func = dico_fun['name_func']
        if(name_func in self._LIST_CUSTOM_FUNC)
            info_func = self._LIST_CUSTOM_FUNC.get(name_func)
            func = info_func[1](**dico_source)
        else:
            pf.from_dict(dico_fun)
            
            
        return constructor(**dico_source)

    
    def _custom_OW(self, dico_fun):
        """ 
        custom rules to build a OwriterYWrap 

        Rules
        -----
            * if provided use ow = [(min1,max1,ow1), (min2,max2,ow2), ..] 
            * if not default dehavior
        """
        constructor = pf.OwriterYWrap
        ow = dico_fun.get('ow')
        if(ow is not None):
            bounds = np.array(ow)
            dico_constructor = {'input_min':bounds[:,0], 'input_max':bounds[:,1], 'output_ow':bounds[:,2]}
            res = constructor(**dico_constructor)
        else:
            res = constructor(**dico_fun)
        return res

    def _custom_FourierFunc(self, dico_source, **extra_args):
        """ custom rules to build a PWL function 
        
        Rules
        -----
        * if Om (frequencies) provided use it else generate it 
          based on nb_H (number of harmonics), freq_type() 
          (option, False). len(T) = nb_steps + 1
        * if F (values at T) provided use them, else 0
        * if F0 or FLast provided enforce these values and make
          them fixed (i.e. not part of free parameters elements)
        """
        constructor = pf.FourierFunc
        Om = dico_source.get('Om')
        
        if(Om is None):
            T = dico_source['T']
            nb_H = dico_source['nb_H']
            freq_type = dico_source.get('freq_type')
            Om = self._gen_frequencies(T, nb_H, freq_type)
            if(isinstance(Om, tuple)):      
                dico_source.update({'Om_bounds':Om[1]})
                Om = Om[0]
        elif(ut.is_iter(Om)):
            nb_H = len(Om)
        else:
            Om = [Om]
            nb_H = 1

        phi = dico_source.get('phi', np.zeros(nb_H))
        A = dico_source.get('A', np.zeros(nb_H))
        B = dico_source.get('B', np.zeros(nb_H))
        c0 = dico_source.get('c0', 0)
        dico_constructor = {'A':A, 'B':B, 'c0':c0, 'phi':phi, 'Om':Om}
        self._add_bounds(dico_constructor, dico_source)
        return constructor(**dico_constructor)
    
    def _custom_StepFunc(self, dico_source, **extra_args):
        """ DEPRECIATED Use PWL instead
        custom rules to build a StepFunc """
        info_func = self._LIST_CUSTOM_FUNC['StepFunc']
        constructor = info_func[1]
        Tstep = dico_source.get('Tstep')
        if(Tstep is None):
            T = dico_source['T']
            nb_step = dico_source['nb_steps']
            dt = float(T/(nb_step -1))
            Tstep = np.r_[np.arange(0, T, dt), T]

        else:
            nb_step = len(Tstep)
        F = dico_source.get('F', np.zeros(nb_step))
        F0 = dico_source.get('F0', 0.0)
        dico_constructor = {'F':F, 'F0':F0, 'Tstep':Tstep}
        self._add_bounds(dico_constructor, dico_source)
        return constructor(**dico_constructor)
    
    def _custom_PWL(self, dico_source, **extra_args):
        """ custom rules to build a PWL function 
        
        Rules
        -----
        * if T (time anchors) provided use it else generate it 
          based on TLast, T0(option, 0), nb_steps, reandomized 
          (option, False). len(T) = nb_steps + 1
        * if F (values at T) provided use them, else 0
        * if F0 or FLast provided enforce these values and make
          them fixed (i.e. not part of free parameters elements)
        """
        constructor = pf.PWL
        T = dico_source.get('T')
        # get the time anchors
        if(T is None):
            T0 = dico_source.get('T0',0)
            TLast = dico_source['TLast']
            nb_step = dico_source['nb_steps']
            randomized = dico_source.get('randomized', False)
            if(randomized):
                T_intermediate = np.random.uniform(T0, TLast, nb_step-1)
                T_intermediate.sort()
            else:
                dt = float((TLast-T0)/(nb_step))    
                T_intermediate = np.arange(T0 + dt, TLast, dt)
            T = np.r_[T0, T_intermediate, TLast]
        else:
            nb_step = len(T)
        F = dico_source.get('F', np.zeros_like(T))
        F0 = dico_source.get('F0')
        FLast = dico_source.get('FLast')                
        if(F0 is not None):
            F[0]=F0
            F0_fixed = True
        else:
            F0_fixed = False
        if(FLast is not None):
            F[-1]=FLast
            FLast_fixed = True
        else:
            FLast_fixed = False
            
        dico_constructor = {'F':F,'T':T}
        self._add_bounds(dico_constructor, dico_source)
        constructed = constructor(**dico_constructor)
    
        if(F0_fixed or FLast_fixed):
            F_bounds = constructed._get_one_bound('F')
            if(F0_fixed):
                F_bounds[0] = False
            if(FLast_fixed):
                F_bounds[-1] = False
            constructed.set_params(F_bounds=F_bounds)
        return constructed

    def _custom_PWC(self, dico_source, **extra_args):
        """ custom rules to build a PWL function 
        
        Rules
        -----
        * if T (time anchors) provided use it else generate it 
          based on TLast, T0(option, 0), nb_steps, reandomized 
          (option, False). len(T) = nb_steps + 1
        * if F (values at T) provided use them, else 0
        * if F0 or FLast provided enforce these values and make
          them fixed (i.e. not part of free parameters elements)
        """
        constructor = pf.PWC
        T = dico_source.get('T')
        # get the time anchors
        if(T is None):
            T0 = dico_source.get('T0',0)
            TLast = dico_source['TLast']
            nb_step = dico_source['nb_steps']
            randomized = dico_source.get('randomized', False)
            if(randomized):
                T_intermediate = np.random.uniform(T0, TLast, nb_step-1)
                T_intermediate.sort()
            else:
                dt = float((TLast-T0)/(nb_step))    
                T_intermediate = np.arange(T0 + dt, TLast, dt)
            T = np.r_[T0, T_intermediate, TLast]
        else:
            nb_step = len(T)
        F = dico_source.get('F', np.zeros_like(T))
        F0 = dico_source.get('F0')
        FLast = dico_source.get('FLast')                
        if(F0 is not None):
            F[0]=F0
            F0_fixed = True
        else:
            F0_fixed = False
        if(FLast is not None):
            F[-1]=FLast
            FLast_fixed = True
        else:
            FLast_fixed = False
            
        dico_constructor = {'F':F,'T':T}
        self._add_bounds(dico_constructor, dico_source)
        constructed = constructor(**dico_constructor)
    
        if(F0_fixed or FLast_fixed):
            F_bounds = constructed._get_one_bound('F')
            if(F0_fixed):
                F_bounds[0] = False
            if(FLast_fixed):
                F_bounds[-1] = False
            constructed.set_params(F_bounds=F_bounds)
        return constructed

    def _add_bounds(self, dico_target, dico_source):
        """ look in dico_source for bounds to update dico_target"""
        dico_update = {}
        for k, v in dico_target.items():
            bounds_name = k + '_bounds'
            if((bounds_name not in dico_target) and (bounds_name in dico_source)): 
                dico_update[bounds_name] = dico_source[bounds_name]
        dico_target.update(dico_update)
 
    
    def _gen_frequencies(self, T, nb_freq = 1, freq_type = None):
        """Generate a set of frequencies based
        'freq_type' controls the way they are picked:
            * 'principal' or None om[l] = 2 * Pi * l /T  
            * 'CRAB' om[l] = 2 * Pi * l * (1 + eps[l]) /T with eps iid U[-0.5, 0.5]  
            * 'DCRAB' om[l] ~ U[0, w_max]
            * string encoding some random number gen
        """
        Om_ref = 2 * np.pi / T
        #dico_args = {'freq_type':freq_type}
        args_rdm = freq_type.split("_")
        rgen = self._rdm_gen #use of the random state

        if(args_rdm[0] in [None, 'principal']):
            Om = (1 + np.arange(nb_freq)) * Om_ref
        
        elif(args_rdm[0] == 'CRAB'):
            rdv_method = 'uniform_-0.5_0.5' 
            rdvgen = rgen.gen_rdmnb_from_string(rdv_method, nb_freq)
            Om = (1 + np.arange(nb_freq) + rdvgen) * Om_ref

        elif(args_rdm[0] == 'DCRAB'):
            if(len(args_rdm)>1):
                Nmax = int(args_rdm[1])
            else:
                Nmax = nb_freq
            wmax = Nmax * Om_ref
            rdv_method = ut.concat2String('uniform', 0, wmax)  
            Om = rgen.gen_rdmnb_from_string(rdv_method, nb_freq)
            Om = np.sort(Om)
            #dico_args['flag'] = 'DCRAB'
        
        # Omega is also a param with some specific boundaries
        elif(args_rdm[0] == 'CRAB_FREEOM'):
            om = (1 + np.arange(nb_freq)) * Om_ref
            om_bounds = [(0.5 + nb, 1.5 + nb) * Om_ref for nb in np.arange(nb_freq)]
            Om = (om, om_bounds)

        else:
            Om = rgen.gen_rdmnb_from_string(freq_type, nb_freq)
            Om = np.sort(Om)

        #dico_args['omegas'] = om
        return Om
        

class Parser():
    """ Parse a string expression into something which can be evaluated 
    by pFunc_zoo.pFuncFactory.eval_string() 

    NOTES
    -----
        *Everything is dealt with strings and the evaluation 
         (with potentially context) is dealt with somewhere else.
        * It is not state dependent thus everything is static

    """


    # keys: expression tag
    # values: tuple(<str> full expr, <tuple> params_0 infos, <tuple> params_1 infos, ...)
    # params_i infos (<str> name, <bool> is_default, <str> default_value)
    _NON_TERMINAL = OrderedDict()
    _NON_TERMINAL['+'] = ("Sum(list_func=[{0}}])", (False, None))
    _NON_TERMINAL['*'] = ("Product(list_func=[{0}}])", (False, None))
    _NON_TERMINAL['#'] = ("Composition(list_func=[{0}}])", (False, None))

    _TERMINAL = OrderedDict()
    _TERMINAL['ow'] = ("{'name_func':'OwriterYWrap', 'ow':[(-inf,{0},{1}),({2},inf,{3})]}",())
    _TERMINAL['bds'] = ("{'name_func':'BoundWrap', 'bounds_min':{0}, 'bounds_max':{1}}", ('min', True, 0), ('max', True, 1))
    _TERMINAL['linear'] = ("{'name_func':'LinearFunc', 'w':{0}, 'bias':{1}}", ('slope', True, 1), ('intercept', True, 0))
    _TERMINAL['constant'] = ("{'name_func':'ConstantFunc', 'c0':{0}}", ('constant', True, 1))
    _TERMINAL['pwc'] = ("{'name_func':'StepFunc','T':{0},'nb_steps':{1}},'F_bounds':{}}", ('final time', True, 1), 
        ('nb_steps', True, 1), ('value bounds', True, [0,1]))
    _TERMINAL['pwl'] = ("{'name_func':'PWL','TLast':{0},'nb_steps':{1},'F_bounds':{2}, 'T0':{3},'F0':{4},'FLast':{5}}", 
        ('final time', True, 1), ('number steps', True, 1),('bounds', True, [0,1]), ('value before', True, 0), ('value after', True, 1))
    
    #to incorporate
    #_LIST_SHORTCUT['rfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['sinpi'] = ("{'name_func':'FourierFunc','A':[0], 'B':[1],'Om':[np.pi/T]}", [])
    #_LIST_SHORTCUT['power'] = ("{'name_func':'PowerFunc','power':%s}", ['power'])            
    #_LIST_SHORTCUT['grbf'] = ("{'name_func':'GRBFFunc','A':%s, 'x0':%s,'l':%s,'A_bounds':%s}", ['A','x0', 'l', 'A_bounds'])
    #_LIST_SHORTCUT['rsinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['four'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'principal','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['sinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'principal','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['omfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['omsinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    #_LIST_SHORTCUT['logis'] = ("{'name_func':'LogisticFunc','L':2,'k':%s,'x0':0}", [])
    #_LIST_SHORTCUT['logisflex'] = ("{'name_func':'LogisticFunc','L':%s,'k':%s,'x0':%s}", [])

    @staticmethod
    def infos(shortcut = None):
        """ Should return infos about the different shorcuts used"""
        if(shortcut is None):
            print('list of operators:')
            print(Parser._NON_TERMINAL)
            print('list pFunc:')
            print(Parser._TERMINAL)
        else:
            infos = Parser._TERMINAL[shortcut]
            print('expression template:')
            print(infos[0])
            print('Parameters (infos, is_default, default_value)'):
            for i in infos[1:]:
                print(i)




    @staticmethod
    def parse(string, debug = False):
        """ Main function transform a string representing a function in a compressed form
        into a string which can be evaluated in pFuncZoo

        Two types of expressions: Terminal(functions) and non Terminal(operators on functions) 
        
        NOTE
        ----
        
        EXAMPLE
        -------
        >>> parse

        """
        if debug:
            pdb.set_trace()
        expr, params = _split(string)
        if(expr in Parser._NON_TERMINAL):
            params = [Parser._parse(p) for p in params]

        final = Parser._assemble(expr, params)
        return final

    @staticmethod
    def _split(string):
        """ split a string as an expr and a list of parameters. separator is "," 
        and sub_expr are based on "(" ")"
        
        NOTE
        ----
        Not computationally optimal (scan several times same part of a string,
        heavy use of append, etc..) still totally fine as string is never that long
        
        EXAMPLE
        -------
        >>> _split("#(a,b(c,d),e)")
        '#', ['a', 'b(c,d)', 'e']
        """
        index_first = string.find("(")
        expr = string[:index_first]
        nesting = 1
        params = []
        index_start, index_stop = 0, 0
        rest = string[index_first+1:]
        for c in rest:
            if(c == '('):
                nesting += 1
            elif(c == ')'):
                nesting -= 1
            elif((c == ',') and (nesting == 1)):
                params += rest[index_start:index_stop-1]
                index_start = index_stop + 1
            index_stop += 1

        if(nested != 1):
            raise SystemError("Parser: {0} too many parenthesis".format(nested-1))

        return expr, params

    @staticmethod
    def _assemble(expr, params):
        """ assemble an expr and its list of parameters"""
        if(expr not in Parser._TERMINAL):
            infos = Parser._TERMINAL[expr]
            expr_template = infos[0]
            param_def = infos[1:]
            formatting = []
            for n, p in len(param_def):
                if n < len(params):
                    formatting.insert(params[n])
                else:
                    if(p[1]): #is there a default value
                        formatting.insert(p[2])
                    else:
                        raise SystemError("Parser: should provide a value for {0}".format(p[0]))

            assembled = expr_template.format(*formatting)

        elif(expr not in Parser._TERMINAL):
            formatting = '(list_func = [' + ','.join(params) + '])'

        else:
            raise SystemError("Parser: {0} not recognized".format(expr))

        return assembled




#==============================================================================
#                     CUSTOM FUNCTIONS ZOOOO
# (non exhaustive) type of functions which can be built with pFunc_factory
### Fourier basis
# fourrier with only A, B as parameters    >>   
# sine basis                  >>   f(t) = sum B_l sin(om_l t) 
# sine basis plus a linear trend                  >>   f(t) = at + b + sum B_l sin(om_l t) 
#
##        
# chebyshev with fixed constant term           >>   f(t) = c0 + sum(1..N) C_i*T_i(t) where c0 is fixed
# chebyshev with fixed trend        >>   f(t) = c0 + sum(1..N) C_i*T_i(t) where c0 and C_1 are fixed
# chebyshev with fixed trend and only even coefficients
#==============================================================================    
if __name__ == '__main__':
    
    expr = '**(#a,*(#b,#c))'
    db = {'a':'func_a','b':'func_b', 'c':'func_c'}
    res = pFunc_factory.parse_compound(expr, db)
    
    
    dico_atom = {'ow':"{'name_func':'OwriterYWrap', 'ow':[(-100,0,0),(5,100,1)]}",
                'bd':"{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}",
                'pw':"{'name_func':'StepFunc','T':5,'nb_steps':10,'F_bounds':(0,1)}"}
    dico_expr = {'final':'**(#ow,**(#bd,#pw))'}
    res2 = pFunc_factory.parse(dico_atom, dico_expr)


    factory = pFunc_factory(None)
    res3 = factory.eval_string(res2['final'])





    # #-------------------------------------------------------#
    # #                    PARSING ZONE // TO BE REMOVED
    # # Dependencies in run_batch.py // gen_config customs
    # # To be linked to Models
    # #-------------------------------------------------------#
    # def parse_and_eval(self, dico_atom, dico_compounds):
    #     """ Parse and evaluate. This is a (state-dependent) method as
    #     the evaluation may depend on the state of the random generator 
    #     and the attribute 

    #     Warning: each functions created are independent..may be messy 
    #     Not used so far.. Think about it
    #     """
    #     dico_parsed = self.parse(dico_atom, dico_compounds)
    #     dico_evaluated = {k:self.eval_string(v) for k,v in dico_parsed}
    #     return dico_evaluated

    
    # @classmethod
    # def parse(cls, dico_atom, dico_compounds):
    #     """  parse dico of expressions (either atomic i.e. can be cretaed on their
    #     own or compounds i.e. rely on other expressions) and return a dico with
    #     a string which can be eval to a function"""
    #     parsed = {}
    #     for k, v in dico_atom.items():
    #         parsed_tmp = cls.parse_atom(v)
    #         parsed.update({k:parsed_tmp})
        
    #     for k,v in dico_compounds.items():
    #         parsed_tmp = cls.parse_compound(v, parsed)
    #         parsed.update({k:parsed_tmp})

    #     return parsed
    
    # @classmethod
    # def parse_atom(cls, atom):
    #     """ parse atomic expression simply append """
    #     if(not(ut.is_str(atom))):
    #         raise SystemError('prse_atom: atom arg should be a string')
    #     res = 'self.from_dict(' + atom + ')'
    #     return res
   
    # @classmethod
    # def parse_compound(cls, expr, db_atom):
    #     """ Parse compound expressions (based on some atomic expressions defined in db_atom)

    #     Compound expressions follow the syntax either 
    #         '#expr' if expr is an atomic expression
    #         '$(expr_0,..,expr_n)' where $ is an operator (#, *, **, +) acting on expr_0 to _n
        
    #     e.g. expr = '**(#a,*(#c,+(#d,#e)))' with db = {'a':xx,'c':yy,'d':qq,'e':zz}
    #     """
    #     if(not(ut.is_str(expr))):
    #         raise SystemError('prse_atom: atom arg should be a string')
        
    #     op, list_sub_expr = cls._split_op_args(expr)
    #     if(op == '#'):
    #        res = db_atom[list_sub_expr]
        
    #     elif(op in cls._PARSING_DICO_OP):
    #         parsed_sub = [cls.parse_compound(sub, db_atom) for sub in list_sub_expr]
    #         res = cls._apply_operator(op, parsed_sub)
    #     else:
    #         raise SystemError('operator {0} not recognized'.format(op))
            
    #     return res
            
    # @classmethod
    # def _split_op_args(cls, expr):
    #     """ 'op(expr1, expr)' >> op='**', list_sub_expr ='[expr1,expr2]'
    #     Special case when operator is # 
    #     '#expr1 >> op='#', list_sub_expr = expr1
    #     """
    #     if(expr[0] == "#"):
    #        op = '#'
    #        list_sub_expr = expr[1:]
    #     else:
    #         index = expr.find('(')
    #         op = expr[:index]
    #         list_sub_expr = cls._split_expressions(expr[index+1:-1])
    #     return op, list_sub_expr

    # @classmethod
    # def _split_expressions(cls, multi):
    #     """ split a string with ',' as a delimiter only if not in a nested exp
    #     ression i.e. all the open) are closed
    #     e.g. '#(f),**(#(g), #(h))' >> ['#(f)', '**(#(g), #(h))']
    #     """
    #     counter_nested = 0
    #     list_expr = []
    #     expr_tmp = ''
    #     for char in multi:
    #         if(char == ','):
    #             if counter_nested == 0:
    #                 list_expr.append(expr_tmp)
    #                 expr_tmp = ''
    #             else:
    #                 expr_tmp += (char)
    #         else:
    #             expr_tmp += (char)
    #             if(char == '('):
    #                 counter_nested += 1 
    #             elif(char == ')'):
    #                 counter_nested -= 1
                    
    #     if(len(expr_tmp) > 0):
    #         list_expr.append(expr_tmp)
    #     return list_expr
                

    # @classmethod
    # def _apply_operator(cls, op, list_parsed):
    #     beg = cls._PARSING_DICO_OP[op] + '(list_func = ['
    #     res = beg + ','.join(list_parsed) + '])'
    #     return res
    #     