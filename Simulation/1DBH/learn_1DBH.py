#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
#
#============================================================================== 
import sys
import pdb
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import pFunc_base as pf
    from QuantumSimulation.Utility.Optim import pFunc_zoo as pfz
    from QuantumSimulation.Utility.Optim import Learner as learn
    from QuantumSimulation.Utility.Optim import Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import pFunc_base as pf
    from ..Utility.Optim import pFunc_zoo as pfz
    from ..Utility.Optim import Learner as learn
    from . import ModelBase as mod
    from .ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import Batch 


class learner(Batch.Batch):
    """
    Should cope with:
        + management of randomGen and mp
        + testing
        + dispatching i.e. more flexibility in creating the controler
        + 
    """
    def run_one_procedure(self, config):
        """ 
        """
        self.random_gen = RandomGenerator.init_random_generator('_RDM_SEED')
        self.mp = MPCapability.init_mp(config.get('_MP_FLAG'))
        dico_update ={'rdm_obj': self.random_gen, 'mp_obj':self.mp}
        
        model_dico = config['model_dico']
        model_dico.update(dico_update)
        model = bh1d.BH1D(**model_dico)
        
        optim_dico = config['optim_dico']
        optim_dico.update(dico_update)
        optim = learn.learner_Opt(model = model, **optim_dico)        
        res = optim()
        
        testing_dico = config.get('testing_dico')
        if(testing_dico is not None):
            testing_dico.update(dico_update)
            model_test = bh1d.BH1D(**testing_dico)
            optim_params = res['params']
            res_test = model_test(optim_params)
            res['test_fom'] = res_test
            res['test_params'] = optim_params

        return res

    @classmethod
    def _dispatch_configs(cls, dico):
        """ AdHoc dispatch rules for the creation of the controller """
        model_dico = dico['model_dico']
        testing_dico = dico.get('testing_dico')
        
        if('_flag_dispatch' in model_dico):
            dico['model_dico'] = cls._process_controler(model_dico)
        
        if((testing_dico is not None) and ('_flag_dispatch' in model_dico)):
            dico['testing_dico'] = cls._process_controler(testing_dico)
    
        return dico

    @classmethod
    def _process_controler(cls, dico):
        """ ad-hoc processing to make description of the controller not too compact 
        final_func= oa_ow ** oa_bds ** (guess* guess2 * (main_ct+(main_fun*main_fun_2))
        where each piece is provided as a dico which can be parsed by pFunc_zoo
        build_atom_custom_func
        """
        pdb.set_trace()
        dico.pop('_flag_dispatch')
        list_bits = [('main_fun2', pf.Product), ('main_ct', pf.Sum),
                     ('main_guess', pf.Product), ('main_guess2', pf.Product),
                     ('oa_bds', pf.Composition), ('oa_ow', pf.Composition)]
        
        main_fun = dico.pop('main_fun')
        res_f = pfz.build_atom_custom_func(main_fun)
        for key, constr in list_bits:
            str_constr = dico.pop(key, None)
            if(key is not None):
                func_tmp = pfz.build_atom_custom_func(str_constr)
                res_f = constr(listfunc = [func_tmp, res_f])

        dico['control_obj']=repr(res_f)
            
        

#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    run_optim = True # examples of optims which run
    create_configs = False # example of generating and storing configs from a meta-config
    create_configs_dispatch=True# example of generating and storing configs from a meta-config and extra-rules
    run_configs = False # run a config
    
    if(run_optim):    
        import copy
        # Create a model
        fom = ['f2t2:neg_fluence:0.0001_smooth:0.01']
        fom_test = ['f2t2:neg_fluence:0.0001_smooth:0.01', 'f2t2', 'fluence', 'smooth', 'varN']
        T=5
        
        s_pwc = "{'name_func':'StepFunc', 'T':5, 'nb_steps':15, 'F_bounds':(0,1)}"
        s_bound = "{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}"
        s_ow = "{'name_func':'OwriterYWrap', 'input_min':[-100, 5], 'input_max':[0, 100], 'output_ow':[0,1]}"
        str_func = '{0} * ({1} * {2})'.format(s_ow, s_bound, s_pwc)
        controler = pfz.pFunc_factory.build_custom_func(str_func)
        controler = pf.pFunc_List([controler])
        dico_simul = {'control_obj':controler, 'L':6, 'Nb':6, 'mu':0, 'T':T, 'dt':0.01, 
                    'flag_intermediate':False, 'setup':'1', 'state_init':'GS_i', 
                    'state_tgt':'GS_f', 'fom':fom, 'fom_print':True, 'track_learning': True}
        
        model = bh1d.BH1D(**dico_simul)


        #DE
        optim_args = {'algo': 'DE'}
        optim = learn.learner_Opt(model = model, **optim_args)
        resDE = optim()
        print(resDE)
        
        #BO
        optim_args = {'algo': 'BO'}
        optim = learn.learner_Opt(model = model, **optim_args)
        resBO = optim()
        resBO['last_func'] = model.control_fun
        print(resBO)
        
        #NM
        optim_args = {'algo': 'NM', 'init_obj': 'uniform_-1_1'}
        optim = learn.learner_Opt(model = model, **optim_args)
        resNM = optim()
        print(resNM)
        
        ## Recreate testing
        dico_test = copy.copy(dico_simul)
        dico_test['fom']=fom_test
        dico_test['track_learning'] = False
        model_test = bh1d.BH1D(**dico_test)
        optim_params = resBO['params']
        res_test = model_test(optim_params)

        
    
    if(create_configs):
        """ take a meta config, generate all the configs and store them """
        learn.parse_and_save_meta_config('test_meta_config.txt', output_folder = 'test_gen_configs')

    if(create_configs_dispatch):
        """ take a meta config, generate all the configs and store them """
        learn.parse_and_save_meta_config('test_meta_config.txt', output_folder = 'test_gen_configs')
    
    if(run_configs):
        batch = learner('test_gen_configs/res0.txt')
        batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
        
        
        
        