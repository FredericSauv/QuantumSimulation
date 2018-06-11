if(__name__ == '__main__'):
    import sys
    sys.path.insert(0, '../')
    import Helper as ut
    import pFunc_base as pf
    import RandomGenerator as rg
else:
    from .. import Helper as ut
    from . import pFunc_base as pf
    from . import RandomGenerator as rg
    
import numpy as np
from numpy import array, zeros
import numpy.polynomial.chebyshev as cheb
import matplotlib.pylab as plt

class pFunc_factory():
    """ Create on demand parametrized function with complicated / ad-hoc patterns
    Enhance pFunc_base by providing for more flexibility/functionality when building the function:

        e.g.1 when building a FourierFunc, doesn't have to provide A and B (which should be 
        provided when using the constructor FourierFunc) - if not found they will be taken as np.zeros
        e.g.2 can provide a list of fixed parameters removing the need to provide their bounds
        e.g.3 some randomization cabilities are provided for the choice of frequencies 
    """

    # <str>key: (<str>:infos, <class>:constructor, <list<list<str>>>: params mandatory, <list<str>>: params optional)
    _LIST_CUSTOM_FUNC = {} 
    ## custom rules
    _LIST_CUSTOM_FUNC['StepFunc'] = ('StepFunction', pf.StepFunc, [['Tstep'],['T', 'nb_steps']], ['F', 'F0'])
    _LIST_CUSTOM_FUNC['FourierFunc'] = ('Fourier basis', pf.FourierFunc, [['Om'], ['T', 'nb_H']],['c0', 'phi', 'A', 'B', 'freq_type'])
    ## untouched
    _LIST_CUSTOM_FUNC['ConstantFunc'] = ('Constant Function', pf.ConstantFunc, [['c0']])
    _LIST_CUSTOM_FUNC['ExpRampFunc'] = ('Exponential Ramp', pf.ExpRampFunc, [])
    _LIST_CUSTOM_FUNC['ChebyshevFunc']= ('Chebyshev basis', pf.ChebyshevFunc, [])
    ## wrappers
    _LIST_CUSTOM_FUNC['OWriter']= ('wrapper: overwritting', pf.OWriter, [],pf.OWriter._LIST_PARAMETERS)
    _LIST_CUSTOM_FUNC['Bounder']= ('wrapper: boundaries', pf.Bounder, [], pf.Bounder._LIST_PARAMETERS)
    _LIST_CUSTOM_FUNC['LTransformer']= ('wrapper: linear transfo', pf.LTransformer, [], pf.LTransformer._LIST_PARAMETERS)
    _LIST_CUSTOM_FUNC['Enrober']= ('wrapper: bound + linear transfo + owritting', pf.Enrober, [], pf.Enrober._LIST_PARAMETERS)    

    @classmethod
    def help(cls, name_func = None):
        if(name_func is None):
            print ("list of available func is {}".format(cls._LIST_CUSTOM_FUNC.keys()))
        else:
            info = cls._LIST_CUSTOM_FUNC[name_func]
            print ("for name_func = {}, underlying func = {}, combination of mandatory params is"
                " ".format(name_func, str(info[1]), info[2]))

    @classmethod
    def build_function(cls, fun_object, dico_out = False, rdm_object = None):
        """ generate composed functions based on complex expr
        ({dico_func1} + {dico_func2}) . {dico_func3}

        TODO: probably need to deal with noise at some point e.g. 'noise_om'
        """
        raise NotImplementedError()
    
    @classmethod
    def gen_atom_func_from_custom_dico(cls, dico_fun, dico_out = False, rdm_object = None):
        """ built a pFunc_base based on a dico specifying
            + key = 'name_func' which func we are expecting 
            + key = 'fixed_params' list of fixed parameters (if True they are all)
            + key = 'paramsXXX' params necessary to gen the function
                     the necessary/optional params can be found in the code below 
                     or should be indicated in _LIST_CUSTOM_FUNC provided it is maintained
        """
        if(dico_fun is None):
            constructor = pf.ConstantFunc
            dico_constructor = {'c0':1, 'c0_bounds':False, 'name_func':'cConstant_F'}

        else:
            name_func = dico_fun['name_func']
            fixed_params = dico_fun['fixed_params']
            info_func = self._LIST_CUSTOM_FUNC[name_func]
            constructor = info_func[1]
            #TODO: check here that there are (one of) the set of required parameters
            # in dico_fun (based on _LIST_CUSTOM_FUNC)

            # Functions for which custom implementation is provided
            if(name_func == 'StepFunc'):
                Tstep = dico_fun.get('Tstep')
                if(Tstep is None):
                    T = dico_fun['T']
                    nb_step = dico_fun['nb_step']
                    dt = float(T/(nb_step -1))
                    Tstep = np.c_[np.arange(0, T, dt), T]

                else:
                    nb_step = len(Tstep)
                F = dico_fun.get('F', np.zeros(nb_step))
                F0 = dico_fun.get('F0', 0)
                
                dico_constructor = {'name_func': name_func, 'F':F, 'F0':F0, 'Tstep':Tstep}

            # fourier basis
            elif(name_func == 'FourierFunc'):
                #either 
                Om = dico_fun.get('Om')
                
                if(Om is None):
                    T = dico_fun['T']
                    nb_H = dico_fun['nb_H']
                    freq_type = dico_fun.get('freq_type')
                    Om = cls._gen_frequencies(T, nbH, args, rdm_object)
                elif(ut.is_iter(Om)):
                    nb_H = len(Om)
                else:
                    Om = [Om]
                    nb_H = 1

                phi = args.get('phi', np.zeros(nb_H))
                A = args.get('A', np.zeros(nb_H))
                B = args.get('B', np.zeros(nb_H))
                c0 = args.get('c0', 0)
                dico_constructor = {'A':A, 'B':B, 'c0':c0, 'phi':phi, 'Om':om}
 

            # No particular implementation for other type of functions 
            # i.e. all the parameters of the function should be in dico_fun
            # still they benificy from the fixing mechanism
            else:
                dico_constructor = copy(dico_fun)

        #### Part about fixing parameters or not
        for k, v in dico_constructor.items():
            if k in fixed_params:
                bounds = False
            else:
                bounds = None
            dico_constructor[k + '_bounds'] = bounds

        func = constructor(**dico_constructor)
        
        if dico_out:
            dico_constructor['name_func'] = name_func
            return func, dico_constructor
        else:
            return func
        
 
    @classmethod
    def _gen_frequencies(cls, T, nb_freq = 1, freq_type = None, random_obj = None):
        """Generate (potentially randomized) frequencies based on 'freq_type'
        'principal' or None om[l] = 2 * Pi * l /T  
        'CRAB' om[l] = 2 * Pi * l * (1 + eps[l]) /T with eps iid U[-0.5, 0.5]  
        'DCRAB' om[l] ~ U[0, w_max]
        others 
        """
        om_ref = 2 * np.pi / T
        #dico_args = {'freq_type':freq_type}
        args_rdm = freq_type.split("_")
        rgen = rg.init_random_generator(random_obj)

        if(args_rdm[0] in [None, 'principal']):
            om = (1 + np.arange(nb_freq)) * om_ref
        
        elif(args_rdm[0] == 'CRAB'):
            rdv_method = 'uniform_-0.5_0.5' 
            rdvgen = rgen.gen_rdmnb_from_string(rdv_method, nb_freq)
            om = (1 + np.arange(nb_freq) + rdvgen()) * om_ref

        elif(args_rdm[0] == 'DCRAB'):
            if(len(args_rdm)>1):
                Nmax = int(args_rdm[1])
            else:
                Nmax = nb_freq
            wmax = Nmax * om_ref
            rdv_method = ut.concat2String('uniform', 0, wmax)  
            om = rgen.gen_rdmnb_from_string(rdv_method, nb_freq)
            om = np.sort(om)
            #dico_args['flag'] = 'DCRAB'

        else:
            om = rgen.gen_rdmnb_from_string(freq_type, nb_freq)
            om = np.sort(om)

        #dico_args['omegas'] = om
        return om
        


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
if __main__
