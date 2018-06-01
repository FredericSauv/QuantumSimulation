#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import sys 
import numpy as np
import matplotlib.pylab as plt
import pdb
import importlib as ilib

if(__name__ == '__main__'):
    sys.path.append("../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.Utility.Optim import RandomGenerator as rdm
    from QuantumSimulation.Utility.Optim import ParametrizedFunctionFactory as pfunc
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import RandomGenerator as rdm
    from ..Utility.Optim import ParametrizedFunctionFactory as pfunc


class Models:
    """Base class for (quantum) models i.e. 
    Attributes:

    
    Methods:
        - SetupTime
        - SetupRandom:    
        - UpdateStateInit / UpdateStateTarget
        - ComputeFOM
    """
    LIST_ARGS = {'T':'<float> Horizon of the simul', 'dt':'<float> time step', 
                 'flagInterTime':'<bool> allows the use of intermediate times', 
                 'state_init':'<string/np.array> Initial state'}
    
    LIST_ARGS_OPT = {'seed': '<int/randomstate/None> Use to generate/pass the random generator',
                     'noise':'<dict<string>> to build noise in the model ', 
                     'state_target': '<string/np.array> Target state'}
    
    def __init__(self, **args_model):
        """ 
        """
        self._ss = None
        self.H = None
        self.state_init = None
        self.state_tgt = None
        self.state_t = None
        self.pop_t = None
        self.time_array = None
        self.t_simul = None
        self._rdmgen = None
        self._fom_func = None
        
        self.SetupRandom(**args_model)
        self.SetupTime(**args_model)

     
    def SetupTime(self, **args_model):
        """ Generate time attributes of the model.
        """
        self.T = args_model['T']
        self.dt = args_model['dt']
        self._flag_intermediate = args_model['flagInterTime']
        
        self.t_array = np.concatenate((np.arange(0, self.T, self.dt), [self.T]))
        if(self._flag_intermediate):
            self.t_simul = self.t_array
        else:
            self.t_simul = self.T
        
        
    def SetupRandom(self, **args_model):
        """ Generate the random generator of the model
        noise = {'fom':'uniform_0_0.05'}
        """
        seed = args_model.get('seed')
        noise = args_model.get('noise')
        
        if(seed is not None):
            self._rdmgen = rdm.RandomGenerator.init_random_generator(seed)
        else:
            self._rdmgen = rdm.RandomGenerator()
            
        if(noise is not None):
            self._noise_func = {k:self._rdmgen.gen_rdmfunc_from_string(v) for 
                                k, v in self._random_dico.items() if k != 'seed'}
        else:
            self._noise_func = None
    
     
    ### LOGIC FOR COMPUTING THE FOM
    def ComputeFOM(self, st, fom):
        """Compute a potentially composed FOM (or list of FOM)
            i.e. fom = ['lst:f2t:neg:0.3, last:f2t2']
        """        
        if(isinstance(fom, list)):
            res=list([])
            for f in fom:
                components = ut.splitString(f)
                tmp = np.sum([self.ComputeFOMAtom(st, c) for c in components])
                res.append(tmp)
        else:
            components = ut.splitString(fom)
            res = np.sum([self.ComputeFOMAtom(st, c) for c in components])
        return res       


    def ComputeFOMAtom(self, st, fom):        
        """Compute a fom by composing functions
            e.g. ComputeFOMAtom(state, 'lst:f2t:neg:0.3') >> 0.3 * neg(f2t(lst))
            i.e. 0.3*(1 - f2t(st[-1])) i.e. inverse of the fidelity 
            computed on the last state.
        """                
        f2apply = [self._fom_func.get(k, lambda x: float(k)*x) for k in ut.splitString(fom, ":")]
        res = ut.compoFunctions(f2apply, st, order = 0)
        return res 


    
    
    ###TO BE IMPLEMENTED IN THE CHILD CLASSES
    def GetState(self, state):
        """ Return a state based on some string/array/other format
        """
        raise NotImplementedError()      
            
    def Simulate(self, **args):
        """ Main entry point to simulate the system: evolves the init state and 
        performs other actions (typically computing a figure of Merit or other 
        statistic)
        """
        raise NotImplementedError()            

        
    def Evolution(self, **args):
        """ Evolve the init_state according to the relevant method and store if
            required
        Output:
            state_t, 
        """
        raise NotImplementedError()      



class ControlledModels(Models):
    """ Base class for controlled models i.e. simulations depending of a
    (some) time-dependent control function(s)
    
    Attributes:
        - controlFun

    
    Methods:
        - SetupControlFun
        - __call__() 
        - 
    """
    LIST_ARGS = Models.LIST_ARGS
    LIST_ARGS['controlFun'] = "<callable/dict{'guess':XXX, 'control': YYY, 'overall'}>"
    
    # Available type of control functions
    # TODO: move to pFunc_zoo class
    DIC_NAME2FUN = {}
    DIC_NAME2FUN['fourFixedCt'] = pfunc.cFourierAmplitudesFixedCt
    DIC_NAME2FUN['sineFixedTrend'] = pfunc.cSineFixedTrend
    DIC_NAME2FUN['sineTrendFixedCt'] = pfunc.cSineTrendFixedCt
    DIC_NAME2FUN['sine'] = pfunc.cSine
    DIC_NAME2FUN['chebyFixedCt'] = pfunc.cChebyshevFixedCt
    DIC_NAME2FUN['chebyFixedTrend'] = pfunc.cChebyshevFixedTrend
    DIC_NAME2FUN['cheby'] = pfunc.cChebyshev    
    DIC_NAME2FUN['chebyOdd'] = pfunc.cChebyshevOdd
    DIC_NAME2FUN['step'] = pfunc.cStepFunc
    DIC_NAME2FUN['squareexp'] = pfunc.SquareExponential
    # DIC_NAME2FUN['sumsquareexp'] = pfunc.SumSquareExponential


    def __init__(self, controlFun, **args_model):
        """Store params of the toymodel
        """
        self.controlFun = None
        Models.__init__(self, **args_model)
        self.SetupControlFunction(controlFun)


    def SetupControlFunction(self, controlFun):
        """ Set up the control functions and 
        if callable (or list of callable are provided) store them directly
        if dico {'control':XXX, 'guess':YYY, 'overall':ZZZ} (or list) is provided
        generate the relevant control functions
        
        >> SetUpControlFunction(lambda x: x**2)
        >> SetUpControlFunction(        
        """
        if(ut.is_list(controlFun)):
            if(ut.is_callable(controlFun[0])):
                self.controlFun = controlFun
                self.info_control = None
            elif(ut.is_dico(controlFun[0])):
                tmp = [self.gen_control_functions(cf) for cf in controlFun]
                self.controlFun = [t[0] for t in tmp]
                self.info_control = [t[1] for t in tmp]
            else:
                raise NotImplementedError()
        else:
            if(ut.is_callable(controlFun)):
                self.controlFun = controlFun
                self.info_control = None
            elif(ut.is_dico(controlFun)):
                self.controlFun, self.info_control = self.gen_control_functions(controlFun)
            else:
                raise NotImplementedError()

    def __call__(self, params, **args_call):
        """ Model(params) >> res 
        """
        self.update_parameters(params)
        res = self.simulate()
        return res


    @property
    def _nb_control_func(self):
        if(self.controlFun is None):
            nb = 0
        elif(ut.is_iter(self.controlFun)):
            nb = len(self.controlFun)
        else:
            nb = 1
        return nb
    
# ---------------------------
# GENERATION OF THE CUSTOM CONTROL FUNCTIONS
# SHOULD MOVE TO pFunc_zoo at one point
# ---------------------------
    def gen_control_functions(self, params):
        """ generate custom parametrized control function(s)
        """
        dicoGuess = params.get('guess', None)
        dicoControl = params.get('control', None)
        dicoOverall = params.get('overall', None)
        T = params['T']
        
        if isinstance(dicoControl, list):
            nb_fun = len(dicoControl)
            control_func = []
            control_info = []
            
            if(not(isinstance(dicoGuess, list))):
                dicoGuess = [dicoGuess for _ in range(nb_fun)]
            
            if(not(isinstance(dicoOverall, list))):
                dicoOverall = [dicoOverall for _ in range(nb_fun)]
            
            for i in range(len(dicoControl)):
                ctl_tmp = dicoControl[i]
                guess_tmp = dicoGuess[i]
                overall_tmp = dicoOverall[i]
                fun_tmp, info_tmp = self.gen_1control_function(guess_tmp, ctl_tmp, overall_tmp, T =T)
                control_func.append(fun_tmp)
                control_info.append(info_tmp)
            
            control_info= ut.concat_dico(control_info, global_name=None, suffix = 'f')
        
        else:
            control_func, control_info = self.gen_1control_function(dicoGuess, dicoControl, dicoOverall, T =T)

        return control_func, control_info


    def gen_1control_function(self, p_guess, p_control, p_overall, **args):
        """ Generate one custom parametrized control function
        f(t) = overall[f_guess(t) * f_param(t)]
        where overall implement potential boundaries/rescaling
        """
        T = args['T']

        # Guess
        if(p_guess is not None):
            p_guess['T'] = T
        f_guess, info_guess = self._gen_param_func(p_guess)
        f_guess.FixParams()

        # Param function 
        p_control['T'] = T
        flag = p_control.get('flag')
        if(flag == 'D-CRAB'):
            f_param, info_control = self._gen_param_func(p_control, split = True)
        else:
            f_param, info_control = self._gen_param_func(p_control, split = False)
        
        # overall
        if (p_overall is not None):
            bds = p_overall.get('bounds')
            ct = p_overall.get('constraints')
            ct_type = p_overall.get('constraints_type')
            ow_X = p_overall.get('ow_X')
            ow_Y = p_overall.get('ow_Y')
            f_control = pfunc.Times([f_param, f_guess], ct, ct_type, bds, ow_X, ow_Y)
        else:
            f_control = f_param

        return f_control, info_control
    
            
    def _gen_param_func(self, args, split = False):
        """ generate parametrized functions:
        """
        if (args in [None, {}]):
            res_func = pfunc.ConstantFunc(1)
            res_dico_args = {'type':'constant', 'c0':1}

        else:
            T = args['T']
            ctts = args.get('constraints')
            ctts_type = args.get('constraints_type')
            bdries = args.get('hard_wall')
            name_function = args.get('func')
            res_dico_args = {'func': name_function, 'constraints':ctts, 
                             'constraints_type':ctts_type, 'T':T, 'hard_wall': bdries, 'indexFunToUpdate':0}
            
            if(name_function == 'constant'):
                 c0 = args.get('c0')
                 res_func = pfunc.ConstantFunc(c0)
                 res_dico_args = {'type':'constant', 'c0':c0}
                 
            # Linear function
            elif(name_function == 'linear'):
                if((ctts is not None) and (len(ctts) == 2)):
                    x0, y0 = ctts[0]
                    x1, y1 = ctts[1]
                    a = (y1 - y0)/(x1 - x0)
                    b = y0 - a*x0
                else:
                    a = args.get('a')
                    b = args.get('b')
                dico_params = {'a':a, 'b':b}
                res_dico_args = ut.add_dico(res_dico_args, dico_params)
                res_func = pfunc.LinearFunc(dico_params)                    
            
            # Exponential function
            elif(name_function == 'exp'):
                raise NotImplementedError()
            
            elif(name_function == 'expramp'):
                a = args['ampl']
                T = args['T']
                l = args['l']
                dico_params = {'ampl':a, 'T':T, 'l':l}
                res_func = pfunc.cFixedExpRamp(dico_params)
            
            # Fourier series type parametrization (needs a number of parameter
            # or harmonics)
            elif(name_function[:4] in ['four', 'sine']):
                function_class = self.DIC_NAME2FUN[name_function]
                nbH = args.get('nbH')
                if(nbH is None):
                    nbP = args.get('nbP')
                    nbH = function_class.GetNbH(nbP)
                else:
                    nbP = None
                    
                om, dico_freq = self.__GenFrequencies(T, nbH, args)
                phi = args.get('phi', 0)
                a = args.get('a', 0)
                b = args.get('b', 0)
                c0 = args.get('c0', 0)
                dico_params = {'nbP':nbP, 'c0':c0, 'phi':phi, 'om':om, 'a':a, 'b':b, 'nbH':nbH}
                # Workaround for DCRAB optim
                if(split):
                    dico_params['c0'] = 0
                    fs = function_class(dico_params, None, None, None)                  
                    ct = pfunc.ConstantFunc(args.get('c0', 0))
                    res_func = pfunc.Plus([fs, ct], ctts, ctts_type, bdries)
                    res_dico_args['indexFunToUpdate'] = [0,0]
                else:
                    res_func = function_class(dico_params, ctts, ctts_type, bdries)
                    
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params, dico_freq)
                    
                                    
            # Chebyshev type parametrization
            elif(name_function[:5] == 'cheby'):
                function_class = self.DIC_NAME2FUN[name_function]
                nbP = args['nbP']
                domain = args.get('domain', [0, T])
                c0 = args.get('c0', 0)
                dico_params = {'c0':c0, 'nbP': nbP, 'domain':domain}
                a = args.get('a')
                b = args.get('b')
                if(a is not None):
                    dico_params['a'] = a
                if(a is not None):
                    dico_params['c0'] = b

                if(split):
                    dico_params['c0'] = 0
                    ch = function_class(dico_params, None, None, None)
                    ct = pfunc.ConstantFunc(c0)
                    res_func = pfunc.Plus([ch, ct], ctts, ctts_type, bdries) 
                    res_dico_args['indexFunToUpdate'] = [0,0]
                else:       
                    res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)
            
            #Step function
            elif(name_function[:4] == 'step'):
                nbP = args['nbP']
                function_class = self.DIC_NAME2FUN[name_function]
                step_dt = T / nbP
                step_T = np.arange(0, T, step_dt)
                dico_params = {'Tstep':step_T}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)
                
            # sqquare exponential    
            elif(name_function[:9] == 'squareexp'):  
                function_class = self.DIC_NAME2FUN[name_function]
                dico_params = {'sigma':1.0, 'mu':T/2, 'l':T/4}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)

                raise NotImplementedError()
            res_dico_args['nb_effective_params'] = res_func._nbTotalParams
        return res_func, res_dico_args
    
    def update_control_parameters(self, params, index_control = None,  **args):
        """ Update the parameters of the control function(s)
        index_control
        if several control functions update them sequentially according to their
        numbers of free parameters: _nbTotalParameters
        """
        #pdb.set_trace()
        if(index_control is None):
            if(self._nb_control_func > 1):
                i_p = 0
                for index, func in enumerate(self._nb_control_func):
                    #What is we want to update part of the function via 
                    nb_eff = func._nbTotalParams                    
                    func.UpdateParams(params[i_p : i_p+nb_eff], **args)
                    i_p += nb_eff
            else:
                self.controlFun.UpdateParams(params, **args)
        else:
            self.controlFun[index_control].UpdateParams(params, **args)

    def __GenFrequencies(self, T, nb_freq = 1, params = None):
        """Generate (potentially randomized) frequencies
        + CRAB:
        + DCRAB: Uniformly drawn in
        + None: gen frequencies k 2Pi/T
        """
        om_ref = 2 * np.pi / T
        name_rdm = params.get('rdm_freq')
        dico_args = {'rdm_freq':name_rdm, 'flag_rdm':False}

        if(name_rdm not in [None, False]):
            args_rdm = ut.splitString(name_rdm)
            if(args_rdm[0] == 'CRAB'):
                dico_args['flag_rdm'] = True
                if(len(args_rdm) == 1):
                    rdv_method = 'uniform_-0.5_0.5'  
                elif(len(args_rdm) == 2):
                    val = str(args_rdm[1])
                    rdv_method = ut.concat2String('uniform', -val, val)
                elif(len(args_rdm) == 3):
                    distrib = val = str(args_rdm[1])
                    val = str(args_rdm[2])
                    rdv_method = ut.concat2String(distrib, -val, val)
                elif(len(args_rdm) == 4):
                    rdv_method = ut.concat2String(args_rdm[1], args_rdm[2], args_rdm[3])                        
                rdvgen = rdm.gen_rdmnb_from_string(rdv_method, nb_freq)
                om = (1 + np.arange(nb_freq) + rdvgen()) * om_ref
                    
            elif(args_rdm[0] == 'DCRAB'):
                dico_args['flag_rdm']=True
                if(len(args_rdm)>1):
                    Nmax = int(args_rdm[1])
                else:
                    Nmax = nb_freq
                wmax = Nmax * om_ref
                rdv_method = ut.concat2String('uniform', 0, wmax)  
                rdvgen = rdm.gen_rdmnb_from_string(rdv_method, nb_freq)
                om = rdvgen()                            
                dico_args['flag'] = 'DCRAB'
            else:
                raise NotImplementedError()
        else:
            om = (1 + np.arange(nb_freq)) * om_ref

        dico_args['omegas'] = om
        return om, dico_args

    def GetFluence(self, time = None, index = None):
        """ Fluence of the control  Functions
        """
        if (time is None):
            time = self.t_array
        
        if self._nb_control_func > 1:
            fl = [self._fluence(f, time) for f in self.controlFun]
        else:
            fl = self._fluence(self.controlFun, time)
        
        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res
    
    def GetSmoothness(self, time = None, time_step = None, index = None):
        """ Smoothness of the control functions
        """
        if (time is None):
            time = self.t_array

        if ut.is_iter(self.controlFun):
            fl = [self._smoothnes(f, time) for f in self.controlFun]
        else:
            fl = self._smoothnes(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res
    
    
    def _fluence(self, func, range_t, normalized = True):
        """ Compute fluence of the control field
        """
        time_step = np.array(range_t[1:]) - np.array(range_t[:-1])
        val_square = np.array([func(t)**2 for t in range_t[:-1]])
        res = np.sum(np.array(val_square * time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res

    def _smoothnes(self, func, range_t, normalized = True):
        """ (ad-hoc) Measure of smoothness for a function (func) evaluated at times
        (range_t)
        res = sum_t (func(t) - func(t-1))**2
        """
        time_step = np.diff(range_t)
        diff_val_square = np.square(np.diff(func(range_t)))
        res = np.sum(np.array(diff_val_square / time_step))
        if(normalized):
            res = res/(range_t[-1] - range_t[0])
        return res
      
     
    
    
class QuspinModels(ControlledModels):
    """ Models based on the QuSpin package.
    
    """
    def __init__(self, controlFun, **args_model):
        """Store params of the toymodel
        """
        self._fom_func = None
        ControlledModels.__init__(self,controlFun, **args_model)
        self.SetupBasis(**args_model)
        self.SetupH(**args_model)
        self.SetupFOM()
        self.UpdateStateInit(args_model['state_init'])
        self.UpdateStateTarget(args_model.get('state_tgt'))

    
    def UpdateStateInit(self, init = None):
        self.state_init = self.GetState(init)

    def UpdateStateTarget(self, tgt = None):
        self.state_tgt = self.GetState(tgt)    
        
    def SetupBasis(self, **args_model):
        """ build the basis (self._ss) and potentially other basis
        """
        raise NotImplementedError()
        
    def SetupH(self, **args_model):
        """ build the Hamiltonian governing the dynamics of the system (self.H)
        and potentially other Hamiltonians
        """
        raise NotImplementedError()
        
    def SetupFOM(self):
        """ Set up functions accessible to compose Figure OF Merit (fOm)
        used in getFOM method
        """
        #main figure
        self._fom_func = {'max': np.max, 'min': np.min, 'avg': np.average, 'sqrt':np.sqrt}
        self._fom_func['last'] = QuspinModels.h_last
        self._fom_func['neg'] = linear
        self._fom_func['f2t'] =  (lambda x: QuspinModels.h_fid(x, self.state_tgt))
        self._fom_func['f2t2'] =  (lambda x: QuspinModels.h_fid2(x, self.state_tgt))
        self._fom_func['fluence'] =  (lambda x: self.GetFluence(self.t_array))
        self._fom_func['fluenceNorm'] =  (lambda x: self.GetFluence(self.t_array) / 2)
        self._fom_func['smooth'] = (lambda x: self.GetSmoothness(self.t_array))
        
        
        #To add randomness
        self._fom_func['rdmplus'] = (lambda x: x + self._noise_func['fom']())
        self._fom_func['rdmtime'] = (lambda x: x * (1 + self._noise_func['fom']()))
        
        # measurement projection on the target_state 
        self._fom_func['proj5'] = (lambda x: QuspinModels.h_n_measures(x, nb =5, measur_basis = self.state_tgt))
        self._fom_func['proj10'] = (lambda x: QuspinModels.h_n_measures(x, nb =10, measur_basis = self.state_tgt))
        self._fom_func['proj100'] = (lambda x: QuspinModels.h_n_measures(x, nb =100, measur_basis = self.state_tgt))
        self._fom_func['proj1000'] = (lambda x: QuspinModels.h_n_measures(x, nb =1000, measur_basis = self.state_tgt))

    def GetState(self, state = None):
        """ Generate quantum states either from string or array/list of num values 
        """
        basis = self._ss
        if(state is None):
            state_res = None

        elif(isinstance(state, str)):
            #pdb.set_trace()
            if(state == 'GS_i'):
                # GS at t=0
                _, state_res = self.H.eigsh(time = 0.0, k=1, which='SA',maxiter=1E10)
            elif(state == 'GS_f'):
                #GS at t = T
                _, state_res = self.H.eigsh(time = self.T, k=1, which='SA',maxiter=1E10) 
            elif(state == 'uniform'):
                #GS at t = T
                state_res = np.random.uniform(-1, 1, size=basis.Ns)
                state_res = state_res / QuspinModels.h_norm(state_res)
            else:
                i_res = basis.index(state)
                state_res = np.zeros(basis.Ns, dtype=np.float64)
                state_res[i_res] =1.0
        else:
            assert(len(state) == basis.Ns), 'wrong size'
            state_res = np.array(state)

        return np.squeeze(state_res)



    ### Helper functions
    @staticmethod
    def h_ip(V1, V2):
        """ compute an inner product between V1 and V2
        """
        return np.dot(np.conj(np.squeeze(V1)), np.squeeze(V2))
    
    @staticmethod
    def h_norm(V1):
        """ compute an inner product between V1 and V2
        """
        return np.sqrt(np.abs(QuspinModels.h_ip(V1, V1)))
    
    @staticmethod
    def h_fid2(V1, V2):
        return np.square(np.abs(QuspinModels.h_ip(V1, V2)))
    
    @staticmethod
    def h_fid(V1, V2):
        return np.abs(QuspinModels.h_ip(V1, V2))
    
    @staticmethod
    def h_state2proba(ket1):
        """ Gen proba distrib from a quantum state
        """
        return np.square(np.abs(ket1))
    
    @staticmethod
    def h_last(V):
        """ return last element
        """
        return V[:, -1]
    
    @staticmethod              
    def h_variance(O, V):
        """ 
        """
        OV = O.dot(V)
        VOOV = np.asscalar(O.matrix_ele(V, OV))
        VOV2 = O.expt_value(V) ** 2
        var = VOOV -VOV2
        assert (np.imag(var) < 1e-8), 'Imaginary part not neglectible.. pb'
        return np.abs(var)
    
    @staticmethod
    def h_variance2(O, V, **args):
        """ Another flavor based on quant_fluct
        """
        res = -O.quant_fluct(V, **args)
        return res
    
    @staticmethod
    def n_measures(ket1, nb = 1, measur_basis = None, num_precis = 1e-6):        
        """Projective measurement in a basis (by default the basis in which the ket 
        is represented) 
        Returns an array with each entry corresponding to the frequency of the 
        measurement based on nb measurements
        
        !!CONVENTION!! measur_basis = N x D (with D dim of the Hilbert space and N the
        number of elements provided)
        
        # TODO: only work for one ket (as opposed to array of kets over time)
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
            assert(len(measur_basis) == dim_ket), "problem with the dimension"
            measur_basis = [measur_basis] # just reshape it
            single_value = True
        else:
            assert(measur_basis.shape[1] == dim_ket)
            
        index_measure = np.arange(len(measur_basis))
        proj = [QuspinModels.h_ip(basis, ket1) for basis in measur_basis]
        proba = QuspinModels.h_probaFromState(proj)
        assert(np.sum(proba)<=(1.0 + num_precis)), "not a valid proba distrib" #should it be relaxed 
        proba_cum = np.concatenate(([0],np.cumsum(proba)))
        samples = np.random.sample(nb)
        observations = [np.sum((samples > proba_cum[i]) * (samples <= proba_cum[i+1])) for i in index_measure]
        frequencies = np.array(observations)/nb
        if(single_value):
            assert(len(frequencies) == 1), "pb"
            frequencies = frequencies[0]
        return frequencies    
    
    
    
    def get_lowest_energies(self, time, nb = 2, H = None):
        """ Get the n lowest energies of the Hamiltonian at time t
        """
        if(H is None):
            H = self.H
            
        gif(ut.is_iter(time)):
            res = [self.get_lowest_energies(t, nb, H) for t in time]
        else:
            res, _ = H.eigsh(time = time, k=nb, which='SA',maxiter=1E10)
        
        return res    
    
    def project_to_instant_evect(self, time = None, ham = None, state_t = None, nb_ev = 5):
        """ Project a state (or a state_t) on the nb_ev first Eigen vectors of ham(time)
        return the poputation of these EigenVectors
        """
        #pdb.set_trace()
        if(time is None):
            time = self.t_simul
        if(ham is None):
            ham = self.H
        if(state_t is None):
            state_t = self.state_t #Need to have been stored
        
        if(ut.is_iter(time)):
            assert (state_t.shape[1] == len(time)), "length of time doesn't match length os state_t"
            pop_ev = [self.project_to_instant_evect(t, ham, np.squeeze(state_t[:,n]), nb_ev) for n, t in enumerate(time)]
        else:
            en, ev = self.H.eigsh(time = time, k=nb_ev, which='SA',maxiter=1E10)
            pop_ev = np.square(np.abs(QuspinModels.h_ip(state_t, ev)))
        return np.array(pop_ev)
    
    def plot_pop_adiab(self, **args_pop_adiab):
        """ Plot pop adiab where each population_t is dispatched on one of the 
        three subplot
        #TODO: better plots
        """
        if(hasattr(self,'pop_adiab')):
            pop_adiab = self.pop_adiab
            t_simul = self.t_simul
            energies = self.energies
            cf = self.get_controlFun_t(t_simul)
            nb_levels = pop_adiab.shape[1]    
            f, axarr = plt.subplots(2,2, sharex=True)
            
            axarr[0,0].plot(t_simul, cf, label = 'f(t)')
            for i in range(nb_levels):
                pop_tmp = pop_adiab[:, i]
                max_tmp = np.max(pop_tmp)
                if(max_tmp > 0.1):
                    axarr[0,1].plot(t_simul, pop_tmp, label = str(i))
                    axarr[1,0].plot(t_simul, energies[:, i], label = str(i))
                elif(max_tmp > 0.01):
                    axarr[1,1].plot(t_simul, pop_tmp, label = str(i))
                    axarr[1,0].plot(t_simul, energies[:, i], label = str(i))
            
            axarr[0,1].legend()
            axarr[1,1].legend()
            axarr[1,0].legend()
            axarr[0,0].legend()

                        
        else:
            _ = self.EvolutionPopAdiab(**args_pop_adiab)
            self.plot_pop_adiab()
        
    def get_controlFun_t(self, t_array = None):
        if(t_array is None):
            t_array = self.t_array
            
        if(self._nb_control_func == 0):
            res = None
        elif(self._nb_control_func == 1):
            res = np.array([self.controlFun(t) for t in t_array])
        else:
            res = np.array([[cf(t) for cf in self.controlFun] for t in t_array])
        return res
        
def linear(val, a = -1, b = 1):
    """ f(x) = b + ax, by def f(x) = 1 - x
    """
    return (b + a * val)

