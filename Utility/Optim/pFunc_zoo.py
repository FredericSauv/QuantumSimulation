if(__name__ == '__main__'):
    import sys
    sys.path.insert(0, '../')
    import Helper as ut
    import pFunc_base as pf
    from RandomGenerator import RandomGenerator as rg
else:
    from .. import Helper as ut
    from . import pFunc_base as pf
    from .RandomGenerator import RandomGenerator as rg
    
import numpy as np
import pdb

class pFunc_factory():
    """ Create on demand parametrized function with complicated / ad-hoc patterns
        Enhance pFunc_base by providing for more flexibility/functionality 
        when building the function

        e.g.1 when building a FourierFunc, doesn't have to provide A and B (which should be 
        provided when using the constructor FourierFunc) - if not found they will be taken as np.zeros
        e.g.2 can provide a list of fixed parameters removing the need to provide their bounds
        e.g.3 some randomization cabilities are provided for the choice of frequencies 
    """

    # <str>key: (<str>:infos, <class>:constructor, <list<list<str>>>: params mandatory, <list<str>>: params optional)
    _LIST_CUSTOM_FUNC = {} 
    _LIST_CUSTOM_FUNC['StepFunc'] = ('StepFunction', pf.StepFunc, [['Tstep'],['T', 'nb_steps']], ['F', 'F0'])
    _LIST_CUSTOM_FUNC['FourierFunc'] = ('Fourier basis', pf.FourierFunc, [['Om'], ['T', 'nb_H']],['c0', 'phi', 'A', 'B', 'freq_type'])
    _LIST_CUSTOM_FUNC['ConstantFunc'] = ('Constant function', pf.ConstantFunc, [['c0']])
    _LIST_CUSTOM_FUNC['LinearFunc'] = ('Linear function', pf.LinearFunc, pf.LinearFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['ExpRampFunc'] = ('Exponential Ramp', pf.ExpRampFunc,pf.ExpRampFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['ChebyshevFunc']= ('Chebyshev basis', pf.ChebyshevFunc, pf.ChebyshevFunc._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['OwriterYWrap']= ('wrapper: overwritting', pf.OwriterYWrap, pf.OwriterYWrap._LIST_PARAMETERS, [])
    _LIST_CUSTOM_FUNC['BoundWrap']= ('wrapper: boundaries', pf.BoundWrap, pf.BoundWrap._LIST_PARAMETERS, [])

    @classmethod
    def help(cls, name_func = None):
        if(name_func is None):
            print ("list of available func is {}".format(cls._LIST_CUSTOM_FUNC.keys()))
        else:
            info = cls._LIST_CUSTOM_FUNC[name_func]
            print ("for name_func = {}, underlying func = {}, combination of mandatory params is"
                " ".format(name_func, str(info[1]), info[2]))

    @classmethod
    def build_custom_func(cls, fun_object, **extra_args):
        """ generate composed functions based on complex expr
        '({dico_func1} + {dico_func2}) . {dico_func3})
        TODO: probably need to deal with noise at some point e.g. 'noise_om'
        """
        #pdb.set_trace()
        if(ut.is_str(fun_object)):
            try:
                func = eval(fun_object)
            except:
                try:
                    bf = 'cls.build_atom_custom_func('
                    af = ",".join([k+'='+str(v) for k, v in extra_args.items()]) + ')'
                    st = ut.parse_enclose_with_counter(fun_object, before = bf, after = af)
                    func = eval(st)
                except:
                    func = pf.pFunc_base.build_pfunc(fun_object)
        else:
            try:
                func = cls.build_atom_custom_func(cls, fun_object, **extra_args)
            except:
                func = pf.pFunc_base.build_from_various(fun_object)
        return func


    @classmethod
    def build_atom_custom_func(cls, dico_fun, rdm_object = None):
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
            func = cls._build_custom_StepFunc(dico_fun)
        elif(name_func == 'FourierFunc'):
            func = cls._build_custom_FourierFunc(dico_fun, rdm_object = rdm_object)
        else:
            func = cls._build_custom_Default(dico_fun, name_func = name_func)
        return func


    @classmethod
    def _build_custom_FourierFunc(cls, dico_source, **extra_args):
        """ custom rules to build a FourierFunc """
        pdb.set_trace()
        info_func = cls._LIST_CUSTOM_FUNC['FourierFunc']
        constructor = info_func[1]
        Om = dico_source.get('Om')
        rdm_object = extra_args.get('rdm_object')
        
        if(Om is None):
            T = dico_source['T']
            nb_H = dico_source['nb_H']
            freq_type = dico_source.get('freq_type')
            Om = cls._gen_frequencies(T, nb_H, freq_type, rdm_object)
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
        pdb.set_trace()
        cls._add_bounds(dico_constructor, dico_source)
        return constructor(**dico_constructor)
    
    @classmethod
    def _build_custom_Default(cls, dico_source, **extra_args):
        """ by Default fetch the relevant constructor and pass it dico_source
        as it is"""
        name_func = extra_args.get('name_func')
        info_func = cls._LIST_CUSTOM_FUNC[name_func]
        constructor = info_func[1]
        return constructor(**dico_source)

    @classmethod  
    def _build_custom_StepFunc(cls, dico_source, **extra_args):
        """ custom rules to build a StepFunc """
        info_func = cls._LIST_CUSTOM_FUNC['StepFunc']
        constructor = info_func[1]
        Tstep = dico_source.get('Tstep')
        if(Tstep is None):
            T = dico_source['T']
            nb_step = dico_source['nb_step']
            dt = float(T/(nb_step -1))
            Tstep = np.c_[np.arange(0, T, dt), T]

        else:
            nb_step = len(Tstep)
        F = dico_source.get('F', np.zeros(nb_step))
        F0 = dico_source.get('F0', 0)
        
        dico_constructor = {'F':F, 'F0':F0, 'Tstep':Tstep}
        cls._add_bounds(dico_constructor, dico_source)
        return constructor(**dico_constructor)


    @classmethod  
    def _add_bounds(self, dico_target, dico_source):
        """ look (in dico_source) for potentially missing bounds (in dico_target)
        and add them (to dico_target)"""
        dico_update = {}
        for k, v in dico_target.items():
            bounds_name = k + '_bounds'
            if((bounds_name not in dico_target) and (bounds_name in dico_source)): 
                dico_update[bounds_name] = dico_source[bounds_name]
        dico_target.update(dico_update)
 
    @classmethod
    def _gen_frequencies(cls, T, nb_freq = 1, freq_type = None, random_obj = None):
        """Generate (potentially randomized) frequencies based on 'freq_type'
        'principal' or None om[l] = 2 * Pi * l /T  
        'CRAB' om[l] = 2 * Pi * l * (1 + eps[l]) /T with eps iid U[-0.5, 0.5]  
        'DCRAB' om[l] ~ U[0, w_max]
        others 
        """
        Om_ref = 2 * np.pi / T
        #dico_args = {'freq_type':freq_type}
        args_rdm = freq_type.split("_")
        rgen = rg.init_random_generator(random_obj)

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
    x = np.arange(-1, 2, 0.01)
    xx =np.arange(0, 1, 0.01)
    A = np.random.sample(2)
    B = np.random.sample(2)
    
    # Only function with free params
    s_f2h = "{'name_func':'FourierFunc', 'T':1, 'nb_H':2, 'A_bounds': None, 'B_bounds': None, 'freq_type':'CRAB'}"
    s_linear = "{'name_func':'LinearFunc', 'w':1, 'bias':0}"
    s_sin_scaling = "{'name_func':'FourierFunc', 'Om':[np.pi], 'A': [0], 'B': [1]}"
    s_bound = "{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}"
    s_ow = "{'name_func':'OwriterYWrap', 'input_min':[-100, 1], 'input_max':[0, 100], 'output_ow':[0,1]}"

    str_func = '{0} * ({1} * ({2} * (1+({3} * {4}))))'.format(s_ow, s_bound, s_linear, s_sin_scaling, s_f2h)
    crab_func = pFunc_factory.build_custom_func(str_func)
    crab_func.theta
    
