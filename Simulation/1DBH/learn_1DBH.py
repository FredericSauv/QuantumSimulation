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
import copy
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import pFunc_zoo as pfz
    from QuantumSimulation.Utility.Optim import Learner as learn
    from QuantumSimulation.Utility.Optim import Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 
    
else:
    from ..Utility import Helper as ut
    from ..Utility.Optim import pFunc_zoo as pfz
    from ..Utility.Optim import Learner as learn
    from .ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 

class learner1DBH(Batch.Batch):
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
        control = model_dico['control_obj']
        if(ut.is_str(control)):
            pdb.set_trace()
            context = copy.copy(model_dico)
            factory.eval_string_from_parse(control, rdm_obj = self.random_gen, 
                                           context = context)
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
        pdb.set_trace
        model_dico = dico['model_dico']
        testing_dico = dico.get('testing_dico')
        
        if((testing_dico is not None) and ('_flag_dispatch' in model_dico)):
            dico['testing_dico'] = cls._process_controler(testing_dico)
    
        return dico

    @classmethod
    def _process_controler(cls, dico):
        """ ad-hoc processing to make description of the controller not too long 
        final_fun c = oa1 ** oa2 ** (guess* guess2 * (mainCt + (main1 * main2))
        where each piece is provided as a dico which can be parsed by pFunc_zoo
        build_atom_custom_func
        """
        pdb.set_trace()
        dico.pop('_flag_dispatch')
        dico_atom = {}
        dico_expr = {}
        for k, v in dico.items():
            bits = k.split('_')
            if(bits[0] == 'ctl'):
                if(bits[1] == 'final'):
                    dico_expr.update({bits[1]:v})
                else:
                    dico_atom.update({bits[1]:v})

        dico['control_obj']=pfz.pFunc_parser.parse(dico_atom, dico_expr)
            
        

#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    run_optim = False # examples of optims which run
    create_configs = False # example of generating and storing configs from a meta-config
    create_configs_dispatch=False# example of generating and storing configs from a meta-config and extra-rules
    run_configs = False # run a config
    
    if(run_optim):    
        
        # Create a model
        fom = ['f2t2:neg_fluence:0.0001_smooth:0.01']
        T=5
        
        factory = pfz.pFunc_factory()
        
        pwc = factory.build_atom_custom_func({'name_func':'StepFunc', 'T':T, 'nb_steps':15, 'F_bounds':(0,1)})
        bounding = factory.build_atom_custom_func({'name_func':'BoundWrap', 'bounds':[0,1]})
        owing = factory.build_atom_custom_func({'name_func':'OwriterYWrap', 'ow':[(-100,0,1), (T,T+100,1)]})
        controler = owing * (bounding * pwc)
       
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
        fom_test = ['f2t2:neg_fluence:0.0001_smooth:0.01', 'f2t2', 'fluence', 'smooth', 'varN']
        dico_test = copy.copy(dico_simul)
        dico_test['fom']=fom_test
        dico_test['track_learning'] = False
        model_test = bh1d.BH1D(**dico_test)
        optim_params = resBO['params']
        res_test = model_test(optim_params)

        
    
    if(create_configs):
        """ take a meta config, generate all the configs and store them """
        learner1DBH.parse_and_save_meta_config('test_meta_config.txt', output_folder = 'test_gen_configs')

    if(create_configs_dispatch):
        """ take a meta config, generate all the configs and store them  TESTED """
        learner1DBH.parse_and_save_meta_config('test_meta_config_dispatch.txt',
                     output_folder = 'test_gen_configs', dispatch = True)
    
    if(run_configs):
        batch = learner1DBH('test_gen_configs/res0.txt')
        batch.run_procedures(saveFreq = 1, splitRes = True, printInfo = False)
        
        
        
        test ={'control_obj':controler,'L':6,'Nb':6,'mu':0,'T':5,'dt':0.01,'flag_intermediate':False,'setup':'1','state_init':'GS_i','state_tgt':'GS_f','fom':['f2t2:neg_fluence:0.0001_smooth:0.01'],'fom_print':True,'track_learning':True}
        testing_dico = {'control_obj':controler,'L':6,'Nb':6,'mu':0,'T':5,'dt':0.01,'flag_intermediate':False,'setup':'1','state_init':'GS_i','state_tgt':'GS_f','fom':['f2t2:neg_fluence:0.0001_smooth:0.01','f2t2','fluence','smooth','varN'],'fom_print':True,'track_learning':True}