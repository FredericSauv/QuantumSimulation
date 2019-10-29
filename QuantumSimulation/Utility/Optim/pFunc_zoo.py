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
import copy
import pdb

class pFunc_factory():
    """Create on demand parametrized function with complicated / ad-hoc patterns
        Enhance pFunc_base by providing for more flexibility/functionality 
        when building the function

        e.g.1 when building a FourierFunc, doesn't have to provide A and B (which should be 
        provided when using the constructor FourierFunc) - if not found they will be taken as np.zeros
        e.g.2 can provide a list of fixed parameters removing the need to provide their bounds
        e.g.3 some randomization cabilities are provided for the choice of frequencies
    


    EXAMPLE
    -------

    """
    # FIRST WAY: From a dictionary of arguments
    # Custom functions and arguments needed to init them
    # <str>key: (<str>:infos, <class>:constructor, <list<list<str>>>: params mandatory, <list<str>>: params optional)
    _LIST_CUSTOM_FUNC = {} 
    _LIST_CUSTOM_FUNC['StepFunc'] = ('StepFunction', pf.StepFunc, [['Tstep'],['T', 'nb_steps']], ['F', 'F0'])
    _LIST_CUSTOM_FUNC['PWL'] = ('StepFunction', pf.PWL, [['Tstep'],['T', 'nb_steps']], ['F', 'F0', 'T0', 'TLast'])
    _LIST_CUSTOM_FUNC['InterpQuad'] = ('IntegratorQuadratic', pf.InterpQuad, [['Tstep'],['T', 'nb_steps']], ['F', 'F0', 'T0', 'TLast','FLast'])
    _LIST_CUSTOM_FUNC['InterpCub'] = ('IntegratorCub', pf.InterpCub, [['Tstep'],['T', 'nb_steps']], ['F', 'F0', 'T0', 'TLast','FLast'])    
    _LIST_CUSTOM_FUNC['FourierFunc'] = ('Fourier basis', pf.FourierFunc, [['Om'], ['T', 'nb_H']],['c0', 'phi', 'A', 'B', 'freq_type'])
    _LIST_CUSTOM_FUNC['ConstantFunc'] = ('Constant function', pf.ConstantFunc, [['c0']],[])
    _LIST_CUSTOM_FUNC['OwriterYWrap']= ('wrapper: overwritting', pf.OwriterYWrap, ['ow'], [])
    _LIST_CUSTOM_FUNC['BoundWrap']= ('wrapper: boundaries', pf.BoundWrap, ['bounds'], [])
    _LIST_CUSTOM_FUNC['LinearFunc'] = ('Linear function', pf.LinearFunc, pf.LinearFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['ExpRampFunc'] = ('Exponential Ramp', pf.ExpRampFunc,pf.ExpRampFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['ChebyshevFunc']= ('Chebyshev basis', pf.ChebyshevFunc, pf.ChebyshevFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['PowerFunc']= ('PowerFunc', pf.PowerFunc, pf.PowerFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['LogisticFunc']= ('LogisticFunc', pf.LogisticFunc, pf.LogisticFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['GRBFFunc']= ('GRBFFunc', pf.GRBFFunc, pf.GRBFFunc._LIST_PARAMETERS, [])


        
    # SECOND WAY with shorcuts
    # Shortcut strings and arguments needed to create them
    # <str>key: (<str>:expression, <list:<str>>:list of parameters)
    _LIST_SHORTCUT = OrderedDict()
    _LIST_SHORTCUT['ow'] = ("{'name_func':'OwriterYWrap', 'ow':[(-inf,0,0),(T,inf,1)]}",[])
    _LIST_SHORTCUT['bds'] = ("{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}", [])
    _LIST_SHORTCUT['linear_r'] = ("{'name_func':'LinearFunc', 'bias':1, 'w':-1/T}", [])
    _LIST_SHORTCUT['linear'] = ("{'name_func':'LinearFunc', 'bias':0, 'w':1/T}", [])
    _LIST_SHORTCUT['constant'] = ("{'name_func':'ConstantFunc', 'c0':[%s]}", ['c0'])
    _LIST_SHORTCUT['sinpi'] = ("{'name_func':'FourierFunc','A':[0], 'B':[1],'Om':[np.pi/T]}", [])
    _LIST_SHORTCUT['power'] = ("{'name_func':'PowerFunc','power':%s}", ['power'])            
    _LIST_SHORTCUT['grbf'] = ("{'name_func':'GRBFFunc','A':%s, 'x0':%s,'l':%s,'A_bounds':%s}", ['A','x0', 'l', 'A_bounds'])
    _LIST_SHORTCUT['rfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    _LIST_SHORTCUT['rsinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    _LIST_SHORTCUT['four'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'principal','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    _LIST_SHORTCUT['sinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'principal','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    _LIST_SHORTCUT['omfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','A_bounds':%s,'B_bounds':%s,'nb_H':%s}", ['A_bounds', 'B_bounds', 'nb_H'])
    _LIST_SHORTCUT['omsinfour'] = ("{'name_func':'FourierFunc','T':T,'freq_type':'CRAB_FREEOM','B_bounds':%s,'nb_H':%s}", ['B_bounds', 'nb_H'])
    _LIST_SHORTCUT['pwc'] = ("{'name_func':'StepFunc','T':T,'F_bounds':%s,'nb_steps':%s}", ['F_bounds', 'nb_steps'])
    _LIST_SHORTCUT['pwl'] = ("{'name_func':'PWL','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}", ['F_bounds', 'nb_steps'])
    _LIST_SHORTCUT['intcub'] = ("{'name_func':'InterpCub','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}", ['F_bounds', 'nb_steps'])
    _LIST_SHORTCUT['intquad'] = ("{'name_func':'InterpQuad','TLast':T,'T0':0,'F0':0,'FLast':1,'F_bounds':%s,'nb_steps':%s}", ['F_bounds', 'nb_steps'])
    _LIST_SHORTCUT['pwlr'] = ("{'name_func':'PWL','TLast':T,'T0':0,'F0':1,'FLast':0,'F_bounds':%s,'nb_steps':%s}", ['F_bounds', 'nb_steps'])
    _LIST_SHORTCUT['logis'] = ("{'name_func':'LogisticFunc','L':2,'k':%s,'x0':0}", [])
    _LIST_SHORTCUT['logisflex'] = ("{'name_func':'LogisticFunc','L':%s,'k':%s,'x0':%s}", [])

    _SHORTCUT_SYNTAX = {}

    ## Mapping symbols operators
    _PARSING_DICO_OP = {'**':'Composition', '*':'Product', '+':'Sum', '#':''}  

    def __init__(self, rdm_obj = None, context = None):
        """ 

        PARAMETER
        ---------
            rdm_gen: RandomState
            contest: dict 
                mapping variable name (key) to variable value 

        """
        self.rdm_gen = rdm_obj
        self.context = context

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



    #-------------------------------------------------------#
    # SHORTCUT MANAGEMENT
    # transform a shortcut string into a pFuncBase
    #-------------------------------------------------------#
    def from_shortcut(self, shortcut, context):
        pass

    @staticmethod
    def get_shortctut_atom_expr(shortcut):
        """ Match the expression return """
        res_expr = None
        res_rest = []
        len_res_found= np.inf
        for k, v in pFunc_factory._LIST_SHORTCUT.items():
            length = len(k)
            try:
                if(shortcut[:length] == k and length < len_res_found):
                    len_res_found = length
                    res_expr = v[0]
                    res_rest = shortcut[length:]
            except:
                pass
        return res_expr, res_rest



    @classmethod
    def extend_shortcut(cls, shortcut):
        """ 
        TO: THINK ABOU//FINISH
        ad-hoc processing to make description of the controller not too long 
        (1) retrieve all the keys starting with ctl_
        (1a) ctl_final is the expression of the controler
        (1b) otherss are the definition of bricks involved in ctl_final
        They are parsed by pFunc_parser
        
        e.g. dico = {'ctl_a':xxx, 'ctl_b':yyy, 'ctl_c':zzz, 'ctl_final':"*(#a, +(#b, #c))"}
        #TODO: NEED TO BE REFACTORED // PUT SOMEWHERE ELSE
        """        
        expr, extra = type(cls).get_shortctut_expr(shortcut)
        logger.info('use of shortcut')

#        if('ctl_shortcut' in dico):
#            
#            shortcut = dico['ctl_shortcut']
#            
#
#
#            
#            if(shortcut[:11] == 'owbds01_pwc'):
#                nb_params = int(shortcut[11:])
#                dico_atom = {'ow':ow,'bd':bds,'pwc':pwc %('(0,1)',nb_params)}
#                dico_expr = {'final':'**(#ow,**(#bd,#pwc))'}
#
#            elif(shortcut[:12] == 'owbds01r_pwc'):
#                nb_params = int(shortcut[12:])
#                dico_atom = {'ow':ow_r,'bd':bds,'pwc':pwc %('(0,1)',nb_params)}
#                dico_expr = {'final':'**(#ow,**(#bd,#pwc))'}
#                        
#            elif(shortcut[:11] == 'owbds01_pwl'):
#                nb_params = int(shortcut[11:])
#                dico_atom = {'ow':ow,'bd':bds,'pwl':pwl %('(0,1)',nb_params+1)}
#                dico_expr = {'final':'**(#ow,**(#bd,#pwl))'}
#
#            elif(shortcut[:13] == 'owbds01r_pwlr'):
#                nb_params = int(shortcut[13:])
#                dico_atom = {'ow':ow_r,'bd':bds,'pwlr':pwlr %('(0,1)',nb_params+1)}
#                dico_expr = {'final':'**(#ow,**(#bd,#pwlr))'}
#
#            ### RDMIZED FREQ
#            elif(shortcut[:13] == 'owbds01_1crab'):
#                # Custom Crab parametrization f(t) = g(t) * (1 + alpha(t)* erf((four series)))
#                # slightly different from the normal one (cf. before)
#                # additional erf function (logistic function such that the four 
#                # series is bounded) alpha(t) is sine ** 1.5
#                nb_params = int(shortcut[13:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                k = 4 /nb_params
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                             'ctm':mone,'logis': logis%(str(k)),
#                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,**(+(#logis,#ctm),#rfour))))))'}
#             
#            elif(shortcut[:13] == 'owbds01_2crab'):
#                # alpha(t) is sine ** 0.5
#                nb_params = int(shortcut[13:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                k = 4 /nb_params
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                             'powscale':sqrt, 'ctm':mone,'logis': logis%(str(k)),
#                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(**(#powscale,#scale),**(+(#logis,#ctm),#rfour))))))'}
#                        
#            elif(shortcut[:13] == 'owbds01_3crab'):
#                # Custom Crab parametrization f(t) = g(t) * (1 + alpha(t)* erf((four series)))
#                # slightly different from the normal one (cf. before)
#                # additional erf function (logistic function such that the four 
#                # series is bounded) alpha(t) is sine ** 1.5
#                nb_params = int(shortcut[13:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                k = 4 /nb_params
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': linear_r, 'ct': one,
#                             'ctm':mone,'logis': logis%(str(k)),
#                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,+(#guess,*(#scale,**(+(#logis,#ctm),#rfour)))))'}
#
#            elif(shortcut[:13] == 'owbds01_4crab'):
#                # AANOTHER Custom Crab parametrization f(t) = g(t) + erf((sin four series)))
#                nb_params = int(shortcut[13:])
#                k = 4 /nb_params
#                x0 = '0.1*T'
#                k2 = '60/T'          
#                L = '1'
#                
#                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone, 'mask':logisflex%(L,k2,x0),
#                             'logis': logis%(str(k)), 'sinfour':rsinfour%('(-1,1)', nb_params)}
#                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,*(#mask,**(+(#logis,#ctm),#sinfour)))))'}
#
#
#            elif(shortcut[:12] == 'owbds01_crab'):
#                # Crab parametrization f(t) = g(t) * (1+alpha(t)*(four series))
#                # with g(t) a linear guess, alpha(t) a sine s.t alpha(0) = alpha(T) = 0
#                # and the four series used randomized frequencies
#                nb_params = int(shortcut[12:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                            'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#rfour)))))'}
#                        
#                        
#            ##########  WITH FREE OMEGAS  ##########
#            elif(shortcut[:15] == 'owbds01_Om4crab'):
#                # AANOTHER Custom Crab parametrization f(t) = g(t) + erf((sin four series)))
#                nb_params = int(shortcut[15:])
#                k = 4 /nb_params
#                x0 = '0.1*T'
#                k2 = '60/T'          
#                L = '1'
#                if(nb_params % 2 != 0):
#                    SystemError('nb_params = {} while it should be 2n'.format(nb_params))
#                nbH = int(nb_params/2)
#                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone, 'mask':logisflex%(L,k2,x0),
#                             'logis': logis%(str(k)), 'sinfour':omsinfour%('(-1,1)', nb_params-1)}
#                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,*(#mask,**(+(#logis,#ctm),#sinfour)))))'}
#
#
#            elif(shortcut[:14] == 'owbds01_Omcrab'):
#                # f(t) = g(t) * (1+alpha(t)*(four series))
#                nb_params = int(shortcut[14:])
#                if(nb_params % 3 != 0):
#                    SystemError('nb_params = {} while it should be 3n'.format(nb_params))
#                nbH = int(nb_params/3)
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                            'rfour':omfour%('(-1,1)', '(-1,1)', str(nbH))}
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#rfour)))))'}           
#            
#
#            ### NOMORERANDOMIZATION BUT GRBF INSTEAD
#            elif(shortcut[:13] == 'owbds01_1grbf'):
#                #pdb.set_trace()
#                # (t) = g(t) * (1 + alpha(t)* RBF)
#                nb_params = int(shortcut[13:])
#                #RBF
#                a_scale = np.sqrt(8* np.log(2))
#                b_scale = np.sqrt(8* np.log(4))
#                cc = (2 * b_scale + (nb_params - 1) * a_scale)
#                sigma_str = 'T/' + str(cc)
#                sigma = [sigma_str for _ in range(nb_params)]
#                l = '[' + ",".join(sigma) + "]"
#                A = str([0.0 for _ in range(nb_params)]) #np.repeat(1, nb_P)
#                x0_list = [str(b_scale) +'*'+ sigma_str + "+" + str(a_scale) + "*" + sigma_str + "*" + str(p) for p in np.arange(nb_params)]  
#                x0 = "[" + ",".join(x0_list)+"]"
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                             'ctm':mone, 'grbf':grbf%(A, x0, l, (-1,1))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#grbf)))))'}
#
#
#            ### W/O RANDOMIZED FREQ    
#            elif(shortcut[:14] == 'owbds01_crfour'):
#                # Crab parametrization w/o randomized freq
#                nb_params = int(shortcut[14:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                            'four':four%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(#scale,#four)))))'}
#             
#            elif(shortcut[:15] == 'owbds01_Ccrfour'):
#                # Custom Crab parametrization w/o randomized freq
#                nb_params = int(shortcut[15:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                k = 4 /nb_params
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                             'powscale':pow15, 'ctm':mone,'logis': logis%(str(k)),
#                             'four':four%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,*(#guess,+(#ct,*(**(#powscale,#scale),**(+(#logis,#ctm),#four))))))'}
#                        
#            
#            elif(shortcut[:14] == 'owbds01_trevfour'):
#                #trend and fourier (sine part only)
#                nb_params = int(shortcut[14:])
#                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'sinfour':sinfour%('(-1,1)', nb_params)}
#                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,#sinfour)))'}
#                        
#                        
#            elif(shortcut[:13] == 'owbds01_trsin'):
#                #f(t) = g(t) + erf((sin four series)))
#                nb_params = int(shortcut[13:])
#                k = 4 /nb_params
#                x0 = '0.1*T'
#                k2 = '60/T'          
#                L = '1'
#                dico_atom = {'ow':ow,'bd':bds,'trend':linear, 'ctm':mone,
#                             'logis': logis%(str(k)), 'sinfour':sinfour%('(-1,1)', nb_params)}
#                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,**(+(#logis,#ctm),#sinfour))))'}
#            
#            
#            # LINEAR RAMP
#            elif(shortcut[:14] == 'owbds01_linear'):
#                dico_atom = {'ow':ow,'bd':bds,'lin':linear}
#                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
#            
#            #LINEAR INVERTED
#            elif(shortcut[:16] == 'owbds01r_linearr'):
#                dico_atom = {'ow':ow_r,'bd':bds,'lin':linear_r}
#                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
#            
#            elif(shortcut[:14] == 'owbds01_wrfour'):
#                #wrapped fourier
#                dico_atom = {'ow':ow,'bd':bds,'lin':linear}
#                dico_expr = {'final':'**(#ow,**(#bd,#lin))'}
#                        
#            elif(shortcut[:13] == 'owbds01_cfred'):
#                # Custom parametrization f(t) = g(t)  + alpha(t)* erf((four series)))
#                # slightly different from the normal one (cf. before)
#                # additional erf function (logistic function such that the four 
#                # series is bounded) alpha(t) = sine ** 0.5
#                nb_params = int(shortcut[13:])
#                if(ut.is_odd(nb_params)):
#                    SystemError('nb_params = {} while it should be even'.format(nb_params))
#                k = 4 /nb_params
#                dico_atom = {'ow':ow,'bd':bds,'guess':linear, 'scale': sinpi, 'ct': one,
#                             'powscale':sqrt, 'ctm':mone,'logis': logis%(str(k)), 'half':half,
#                             'rfour':rfour%('(-1,1)', '(-1,1)', str(int(nb_params/2)))}
#                
#                dico_expr = {'final':'**(#ow,**(#bd,+(#guess,*(*(#half,**(#powscale,#scale)),**(+(#logis,#ctm),#rfour)))))'}
#                        
#            
#            else:
#                 raise SystemError('implement more shotcuts here')
#            
#            dico_processed['control_obj'] = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
#        
#        else:
#            dico_atom = {}
#            dico_expr = {}
#            list_keys_to_remove = []
#            for k, v in dico_processed.items():
#                bits = k.split('_')
#                if(bits[0] == 'ctl'):
#                    list_keys_to_remove.append(k)
#                    if(bits[1] == 'final'):
#                        dico_expr.update({bits[1]:v})
#                    else:
#                        dico_atom.update({bits[1]:v})
#                    
#    
#            for k in list_keys_to_remove:
#                del dico_processed[k]
#            if('final' in dico_expr):
#                dico_processed['control_obj'] = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
#        return dico_processed

    #-------------------------------------------------------#
    #                    EVALUATING ZONE
    #-------------------------------------------------------#
    def eval_string(self, string, context = None):
        """ eval a string in this environment (string generated 
        from pFunc_parser.parse)"""
        if context is None:
            context = self.context
        return eval(string, None, locals().update(context))

      
    #-------------------------------------------------------#
    #                    PARSING ZONE
    #-------------------------------------------------------#
    def parse_and_eval(self, dico_atom, dico_compounds):
        """ Parse and evaluate. This is a (state-dependent) method as
        the evaluation may depend on the state of the random generator 
        and the attribute 

        Warning: each functions created are independent..may be messy 
        Not used so far.. Think about it
        """
        dico_parsed = self.parse(dico_atom, dico_compounds)
        dico_evaluated = {k:self.eval_string(v) for k,v in dico_parsed}
        return dico_evaluated

    
    @classmethod
    def parse(cls, dico_atom, dico_compounds):
        """  parse dico of expressions (either atomic i.e. can be cretaed on their
        own or compounds i.e. rely on other expressions) and return a dico with
        a string which can be eval to a function"""
        parsed = {}
        for k, v in dico_atom.items():
            parsed_tmp = cls.parse_atom(v)
            parsed.update({k:parsed_tmp})
        
        for k,v in dico_compounds.items():
            parsed_tmp = cls.parse_compound(v, parsed)
            parsed.update({k:parsed_tmp})

        return parsed
    
    @classmethod
    def parse_atom(cls, atom):
        """ parse atomic expression simply append """
        if(not(ut.is_str(atom))):
            raise SystemError('prse_atom: atom arg should be a string')
        res = 'self.build_atom_func(' + atom + ')'
        return res
   
    @classmethod
    def parse_compound(cls, expr, db_atom):
        """ Parse compound expressions (based on some atomic expressions defined in db_atom)

        Compound expressions follow the syntax either 
            '#expr' if expr is an atomic expression
            '$(expr_0,..,expr_n)' where $ is an operator (#, *, **, +) acting on expr_0 to _n
        
        e.g. expr = '**(#a,*(#c,+(#d,#e)))' with db = {'a':xx,'c':yy,'d':qq,'e':zz}
        """
        if(not(ut.is_str(expr))):
            raise SystemError('prse_atom: atom arg should be a string')
        
        op, list_sub_expr = cls._split_op_args(expr)
        if(op == '#'):
           res = db_atom[list_sub_expr]
        
        elif(op in cls._PARSING_DICO_OP):
            parsed_sub = [cls.parse_compound(sub, db_atom) for sub in list_sub_expr]
            res = cls._apply_operator(op, parsed_sub)
        else:
            raise SystemError('operator {0} not recognized'.format(op))
            
        return res
            
    @classmethod
    def _split_op_args(cls, expr):
        """ 'op(expr1, expr)' >> op='**', list_sub_expr ='[expr1,expr2]'
        Special case when operator is # 
        '#expr1 >> op='#', list_sub_expr = expr1
        """
        if(expr[0] == "#"):
           op = '#'
           list_sub_expr = expr[1:]
        else:
            index = expr.find('(')
            op = expr[:index]
            list_sub_expr = cls._split_expressions(expr[index+1:-1])
        return op, list_sub_expr

    @classmethod
    def _split_expressions(cls, multi):
        """ split a string with ',' as a delimiter only if not in a nested exp
        ression i.e. all the open) are closed
        e.g. '#(f),**(#(g), #(h))' >> ['#(f)', '**(#(g), #(h))']
        """
        counter_nested = 0
        list_expr = []
        expr_tmp = ''
        for char in multi:
            if(char == ','):
                if counter_nested == 0:
                    list_expr.append(expr_tmp)
                    expr_tmp = ''
                else:
                    expr_tmp += (char)
            else:
                expr_tmp += (char)
                if(char == '('):
                    counter_nested += 1 
                elif(char == ')'):
                    counter_nested -= 1
                    
        if(len(expr_tmp) > 0):
            list_expr.append(expr_tmp)
        return list_expr
                

    @classmethod
    def _apply_operator(cls, op, list_parsed):
        beg = cls._PARSING_DICO_OP[op] + '(list_func = ['
        res = beg + ','.join(list_parsed) + '])'
        return res
        


    
    #-------------------------------------------------------#
    # FROM A DICTIONARY
    # Helps to build function from a dico
    #-------------------------------------------------------#
    def build_atom_func(self, dico_fun):
        """ built a pFunc_base based on a dico specifying
            + key = 'name_func' which func we are expecting 
            + key = 'paramsXXX' params necessary to gen the function
                     the necessary/optional params can be found in the code below 
                     or should be indicated in _LIST_CUSTOM_FUNC provided it is maintained
            + key = 'paramsXXX_bounds' (optional) by default they will be set to False
                    i.e. the params are frozen (cf. docstring of pFunc_base)
        """
        name_func = dico_fun['name_func']
        if(name_func == 'StepFunc'):
            func = self._build_custom_StepFunc(dico_fun)
        elif(name_func in ['PWL','InterpQuad', 'InterpCub']):
            func = self._build_custom_PWL(name_func, dico_fun)
        elif(name_func == 'FourierFunc'):
            func = self._build_custom_FourierFunc(dico_fun)
        elif(name_func == 'BoundWrap'):
            func = self._build_custom_BoundWrap(dico_fun)
        elif(name_func == 'OwriterYWrap'):
            func = self._build_custom_OwriterYWrap(dico_fun)
        else:
            func = self._build_custom_Default(dico_fun)
        return func

    def _build_custom_BoundWrap(self, dico_fun):
        """ can provides the bounds as {'bounds':[a,b]}"""
        info_func = self._LIST_CUSTOM_FUNC['BoundWrap']
        constructor = info_func[1]
        bounds = dico_fun.get('bounds')
        if(bounds is not None):
            dico_constructor = {'bounds_min':bounds[0], 'b_build_control_from_stringounds_max':bounds[1]}
            res = constructor(**dico_constructor)
        else:
            res = self._build_custom_Default(dico_fun)
        return res
    
    def _build_custom_OwriterYWrap(self, dico_fun):
        """ can provide all the needed infos as #
        bounds=[(min1,max1,ow1), (min2,max2,ow2), ..] """
        info_func = self._LIST_CUSTOM_FUNC['OwriterYWrap']
        constructor = info_func[1]
        ow = dico_fun.get('ow')
        if(ow is not None):
            bounds = np.array(ow)
            dico_constructor = {'input_min':bounds[:,0], 'input_max':bounds[:,1], 'output_ow':bounds[:,2]}
            res = constructor(**dico_constructor)
        else:
            res = self._build_custom_Default(dico_fun)
        return res

    def _build_custom_FourierFunc(self, dico_source, **extra_args):
        """ custom rules to build a FourierFunc """
        info_func = self._LIST_CUSTOM_FUNC['FourierFunc']
        constructor = info_func[1]
        Om = dico_source.get('Om')
        use_bounds = False
        
        if(Om is None):
            T = dico_source['T']
            nb_H = dico_source['nb_H']
            freq_type = dico_source.get('freq_type')
            Om = self._gen_frequencies(T, nb_H, freq_type)
            if(isinstance(Om, tuple)):
                use_bounds = True        
                Om_bounds = Om[1] 
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
        if(use_bounds):
            dico_constructor['Om_bounds'] = Om_bounds
        self._add_bounds(dico_constructor, dico_source)
        return constructor(**dico_constructor)
    
    def _build_custom_Default(self, dico_source, **extra_args):
        """ by Default fetch the relevant constructor and pass dico_source
        as it is"""
        name_func = dico_source.get('name_func')
        info_func = self._LIST_CUSTOM_FUNC[name_func]
        constructor = info_func[1]
        return constructor(**dico_source)


    def _build_custom_StepFunc(self, dico_source, **extra_args):
        """ custom rules to build a StepFunc """
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
    
    def _build_custom_PWL(self, name_integrator, dico_source, **extra_args):
        """ custom rules to build either a PWL/integrators 
        with the possibility to pass a T0, TLast, F0, FLast
        TODO: Integgrate pwl (they could all rely on scipy interpolator)
        """
        info_func = self._LIST_CUSTOM_FUNC[name_integrator]
        constructor = info_func[1]
        T = dico_source.get('T')
        if(T is None):
            T0 = dico_source.get('T0',0)
            TLast = dico_source['TLast']
            nb_step = dico_source['nb_steps']
            dt = float((TLast-T0)/(nb_step))
            T = np.r_[np.arange(T0, TLast, dt), TLast]
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
        """ look (in dico_source) for potentially missing bounds (in dico_target)
        and add them (to dico_target)"""
        dico_update = {}
        for k, v in dico_target.items():
            bounds_name = k + '_bounds'
            if((bounds_name not in dico_target) and (bounds_name in dico_source)): 
                dico_update[bounds_name] = dico_source[bounds_name]
        dico_target.update(dico_update)
 
    
    def _gen_frequencies(self, T, nb_freq = 1, freq_type = None):
        """Generate (potentially randomized) frequencies based on 'freq_type'
        'principal' or None om[l] = 2 * Pi * l /T  
        'CRAB' om[l] = 2 * Pi * l * (1 + eps[l]) /T with eps iid U[-0.5, 0.5]  
        'DCRAB' om[l] ~ U[0, w_max]
        others 
        """
        Om_ref = 2 * np.pi / T
        #dico_args = {'freq_type':freq_type}
        args_rdm = freq_type.split("_")
        rgen = self._rdm_gen #Use this rdamgen (provided or created at init of the factory)

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
    ## Try parsing dicos
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
    
    
    # Build from shortcut string
    # func_factory = pFunc_factory(rdm_obj=None, context={'T':5})
    # built_control = func_factory.eval_string('owbds01_pwl15')