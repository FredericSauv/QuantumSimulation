#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:35:46 2017

@author: frederic
"""
import numpy as np
import matplotlib.pylab as plt
import pdb
import importlib as ilib
from ..Utility.Optim import RandomGenerator as rdm


ilib.reload(rdm)

class Models:
    """Abstract class for models 

    """
    def __init__(self, noise = None):
        """ 
        """
        self._ss = None
        self.state_init = None
        self.state_tgt = None
        self.state_t = None
        self.pop_t = None
        self.time_array = None

        if(noise is not None):
            # noise = {'seed': 122888, 'Ex': 'uniform_5,5', 'Ey':'uniform'}
            self._noise = noise
            self._rdmgen = rdm.RandomGenerator(noise.get('seed'))
            self._noise_func = {k:self._rdmgen.gen_rdmfunc_from_string(v) for 
                                k, v in self._noise.items() if k != 'seed' }
        else:
            self._rdmgen = rdm.RandomGenerator()

            
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



# --------------------------------------------------------------------------- #
#   Updates
# --------------------------------------------------------------------------- #
    def UpdateStateInit(self, init = None):
        self.state_init = self.GetState(init)

    def UpdateStateTarget(self, tgt = None):
        self.state_tgt = self.GetState(tgt)

    def randomize(self):
        """ Randomize what has to be randomized
        """
        raise NotImplementedError()


# --------------------------------------------------------------------------- #
#   Extras: get the methods from the underlying statespace
# --------------------------------------------------------------------------- #
    def GetState(self, state = None):
        """ Get some specific state from the State Space 
        """
        if(state is None):
            res = None

        elif(isinstance(state, str)):
            res = self._ss.GetStateByName(state)

        else:
            res = np.array(state)
        return res



class ControlledModels(Models):
    """ Abstract class for a controlled model i.e. simulations depend
    on control function(s)
    """

    # available type of control functions 
    DIC_NAME2FUN = {}
    DIC_NAME2FUN['fourFixedCt'] = pfunc.cFourierAmplitudesFixedCt
    DIC_NAME2FUN['sineFixedTrend'] = pfunc.cSineFixedTrend
    DIC_NAME2FUN['sineTrendFixedCt'] = pfunc.cSineTrendFixedCt
    DIC_NAME2FUN['sine'] = pfunc.cSine
    DIC_NAME2FUN['chebyFixedCt'] = pfunc.cChebyshevFixedCt
    DIC_NAME2FUN['chebyFixedTrend'] = pfunc.cChebyshevFixedTrend
    DIC_NAME2FUN['cheby'] = pfunc.cChebyshev    
    DIC_NAME2FUN['chebyOdd'] = pfunc.cChebyshevOdd
    DIC_NAME2FUN['step'] = pfunc.StepFunc
    DIC_NAME2FUN['squareexp'] = pfunc.SquareExponential
    # DIC_NAME2FUN['sumsquareexp'] = pfunc.SumSquareExponential

    def __init__(self, **args_model):
        """Store params of the toymodel
        """
        self.controlFun = None
        Models.__init__(self, **args_model)

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
# CUSTOM CONTROL FUNCTION
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
        if(self._name_algo == 'D-CRAB'):
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
    

    def update_control_parameters(self, params, index_control = None,  **args):
        """ Update the parameters of the control function(s)
        index_control
        if several control functions update them sequentially according to their
        numbers of free parameters: _nbTotalParameters
        """
        if(index_control is None):
            if(self._nb_control_func > 1):
                i_p = 0
                for index, func in enumerate(self._nb_control_func):
                    nb_eff = func._nbTotalParams                    
                    func.UpdateParams(params[i_p : i_p+nb_eff], **args)
                    i_params += nb_eff
            else:
                self.controlFun.UpdateParams(params, **args)
        else:
            self.controlFun[index_control].UpdateParams(params, **args)


            
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
                    dico_params = ConstraintsToLinearCoeff(ctts, dico = True)
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
            
            # Fourier series type parametrization
            elif(name_function[:4] in ['four', 'sine']):
                function_class = self.DIC_NAME2FUN[name_function]
                nbH = function_class.GetNbH(self._nb_params)
                om, dico_freq = self.__GenFrequencies(T, nbH, args)
                phi = args.get('phi', 0)
                a = args.get('a', 0)
                b = args.get('b', 0)
                c0 = args.get('c0', 0)
                dico_params = {'nbP':self._nb_params, 'c0':c0, 'phi':phi, 'om':om, 'a':a, 'b':b}
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
                domain = args.get('domain', [0, T])
                c0 = args.get('c0', 0)
                dico_params = {'c0':c0, 'nbP': self._nb_params, 'domain':domain}
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
                function_class = self.DIC_NAME2FUN[name_function]
                step_dt = T / self._nb_params
                step_T = np.arange(0, T, step_dt)
                dico_params = {'Tstep':step_T}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)
                
            elif(name_function[:9] == 'squareexp'):  
                function_class = self.DIC_NAME2FUN[name_function]
                dico_params = {'sigma':1.0, 'mu':T/2, 'l':T/4}
                res_func = function_class(dico_params, ctts, ctts_type, bdries)
                res_dico_args = ut.merge_N_dico(1, res_dico_args, dico_params)

            else:
                raise NotImplementedError()
            res_dico_args['nb_effective_params'] = res_func._nbTotalParams
        return res_func, res_dico_args
    

    def __GenFrequencies(self, T, nb_freq = 1, params = None):
        """Generate (potentially randomized) frequencies
        
        + CRAB:
        + DCRAB: Uniformly drawn in
        + None: gen frequencies k 2Pi/T
        """
        om_ref = 2 * np.pi / T
        name_rdm = params.get('name_rdm')
        seed_rdm = params.get('seed_rdm')
        dico_args = {}
        dico_args['name_rdm'] = name_rdm
        dico_args['seed_rdm'] = seed_rdm
        
        if(name_rdm is not None):
            args_rdm = ut.splitString(name_rdm)
            if(args_rdm[0] == 'CRAB'):
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
                rdvgen = ut.getRandomGenerator(method = rdv_method, nb = nb_freq, seed = seed_rdm)
                om = (1 + np.arange(nb_freq) + rdvgen()) * om_ref
                    
            elif(args_rdm[0] == 'DCRAB'):
                if(len(args_rdm)>1):
                    Nmax = int(args_rdm[1])
                else:
                    Nmax = nb_freq
                wmax = Nmax * om_ref
                rdv_method = ut.concat2String('uniform', 0, wmax)  
                rdvgen = ut.getRandomGenerator(method = rdv_method, nb = nb_freq, seed = seed_rdm)
                om = rdvgen()                            
                dico_args['flag'] = 'DCRAB'
        else:
            om = (1 + np.arange(nb_freq)) * om_ref

        dico_args['omegas'] = om
        return om, dico_args

    def GetFluence(self, time = None, index = None):
        """
        """
        if (time is None):
            time = self.t_array
        
        if ut.is_iter(self.controlFun):
            fl = [getFluence(f, time) for f in self.controlFun]
        else:
            fl = getFluence(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])

        return res
    
    def GetSmoothness(self, time = None, time_step = None, index = None):
        """
            
        """
        if (time is None):
            time = self.t_array

        if ut.is_iter(self.controlFun):
            fl = [getSmoothnes(f, time) for f in self.controlFun]
        else:
            fl = getSmoothnes(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res
    
    def GetSmoothness2(self, time = None, time_step = None, index = None):
        """
            
        """
        if (time is None):
            time = self.t_array

        if ut.is_iter(self.controlFun):
            fl = [getSmoothnes(f, time) for f in self.controlFun]
        else:
            fl = getSmoothnes(self.controlFun, time)

        if (index is None):
            res = np.sum(fl)
        else:
            res = np.sum(fl[index])
        return res   