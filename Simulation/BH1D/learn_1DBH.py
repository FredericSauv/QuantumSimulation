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
import numpy as np
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.Utility import Helper as ut
    from QuantumSimulation.ToyModels import BH1D as bh1d
    from QuantumSimulation.Utility.Optim import pFunc_zoo, pFunc_base, Learner, Batch 
    from QuantumSimulation.Utility.Optim.RandomGenerator import RandomGenerator 
    from QuantumSimulation.Utility.Optim.MP import MPCapability 
    
else:
    from ...Utility import Helper as ut
    from ...Utility.Optim import pFunc_zoo, pFunc_base, Batch, Learner
    from ...Utility.Optim.RandomGenerator import RandomGenerator 
    from ...Utility.Optim.MP import MPCapability 
    from ...ToyModels import BH1D as bh1d

class learner1DBH(Batch.Batch):
    """
    Should cope with:
        + management of randomGen and mp
        + testing
        + dispatching i.e. more flexibility in creating the controler
    TODO: noise in the test through seed_noise = [(), ..., ()]
    """
    def run_one_procedure(self, config):
        """ 
        """
        self.random_gen = RandomGenerator.init_random_generator(config['_RDM_SEED'])
        self.mp = MPCapability.init_mp(config.get(config['_MP_FLAG']))
        dico_update ={'rdm_obj': self.random_gen, 'mp_obj':self.mp}
        
        model_dico = config['model_dico']
        track_learning = model_dico.pop('track_learning', False)
        model_dico.update(dico_update)
        self._build_control(model_dico)
        model = bh1d.BH1D(**model_dico)
        
        optim_dico = config['optim_dico']
        optim_dico.update(dico_update)
        optim = Learner.learner_Opt(model = model, **optim_dico)        
        res = optim(track_learning = track_learning)
        control_fun = model.control_fun.clone()
        control_fun.theta = res['params']
        res['func'] = repr(control_fun)
        del model_dico['rdm_obj']
        del model_dico['mp_obj']
        del optim_dico['rdm_obj']
        del optim_dico['mp_obj']
        
        testing_dico = config.get('testing_dico')
        if(testing_dico is not None):
            testing_dico.update(dico_update)
            self._build_control(testing_dico)
            model_test = bh1d.BH1D(**testing_dico)
            optim_params = res['params']
            res_test = model_test(optim_params, trunc_res = False)
            res['test_fom'] = res_test
            res['test_params'] = optim_params
            del testing_dico['rdm_obj']
            del testing_dico['mp_obj']
        return res
    
    
    def _build_control(self, model_dico):
        """ if the control_object is a string evaluate it if not do nothing"""
        control = model_dico['control_obj']
        if(ut.is_str(control)):
            model_dico['control_obj'] = learner1DBH._build_control_from_string(control,
                      self.random_gen, model_dico)

            
    @classmethod
    def _build_control_from_string(cls, control, random_gen=None, context_dico = None):
        if(context_dico is not None):
            context = {k:v for k, v in context_dico.items() if k not in 
                       ['control_obj', 'random_obj', 'mp_obj']}
        else:
            context = None
            
        if(random_gen is None):
            random_gen = RandomGenerator()
        func_factory = pFunc_zoo.pFunc_factory(random_gen, context)
        built_control = func_factory.eval_string(control)
        return built_control
    
    @classmethod
    def _processing_meta_configs(cls, dico):
        """ AdHoc processing rules when dealing with meta configs: 
        helps to create the controler    
            """
        model_dico = dico['model_dico']
        dico['model_dico'] = cls._process_controler(model_dico)
        testing_dico = dico.get('testing_dico')
        if((testing_dico is not None) ):
            dico['testing_dico'] = cls._process_controler(testing_dico)
    
        return dico

    @classmethod
    def _process_controler(cls, dico):
        """ ad-hoc processing to make description of the controller not too long 
        (1) retrieve all the keys starting with ctl_
        (1a) ctl_final is the expression of the controler
        (1b) otherss are the definition of bricks involved in ctl_final
        They are parsed by pFunc_parser
        
        e.g. dico = {'ctl_a':xxx, 'ctl_b':yyy, 'ctl_c':zzz, 'ctl_final':"*(#a, +(#b, #c))"}
        """
        dico_atom = {}
        dico_expr = {}
        dico_processed = copy.copy(dico)
        
        if('ctl_shortcut' in dico):
            print('use of shortcut')
            shortcut = dico['ctl_shortcut']
            
            if(shortcut[:11] == 'owbds01_pwc'):
                nb_params = int(shortcut[11:])
                dico_atom = {'ow':"{'name_func':'OwriterYWrap', 'ow':[(-100,0,0),(T,100+T,1)]}",
                            'bd':"{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}",
                            'pw':"{'name_func':'StepFunc','T':T,'F_bounds':(0,1),'nb_steps':"+ str(nb_params)+"}"}
                dico_expr = {'final':'**(#ow,**(#bd,#pw))'}
             
            elif(shortcut[:12] == 'owbds01_crab'):
                nb_params = int(shortcut[12:])
                if(ut.is_odd(nb_params)):
                    SystemError('nb_params = {} while it should be even'.format(nb_params))
                
                dico_atom = {'ow':"{'name_func':'OwriterYWrap', 'ow':[(-100,0,0),(T,100+T,1)]}",
                            'bd':"{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}",
                            'guess':"{'name_func':'LinearFunc', 'bias':0, 'w':1/T}",
                            'rfour':"{'name_func':'FourierFunc','T':T,freq_type:'CRAB','A_bounds':(-1,1),'B_bounds':(-1,1),'nb_H':"+ str(int(nb_params/2))+"}",
                            'scale':"{'name_func':'FourierFunc','A':[0], 'B':[1],Om:[np.pi/T]}",
                            'ct':"{'name_func':'ConstantFunc', 'c0':[1]}"                            
                            }
                dico_expr = {'final':'**(#ow,**(#bd,+(#ct,*(#scale,#rfour))))'}
            
            elif(shortcut[:14] == 'owbds01_trfour'):
                nb_params = int(shortcut[14:])
                dico_atom = {'ow':"{'name_func':'OwriterYWrap', 'ow':[(-100,0,0),(T,100+T,1)]}",
                            'bd':"{'name_func':'BoundWrap', 'bounds_min':0, 'bounds_max':1}",
                            'trend':"{'name_func':'LinearFunc', 'bias':0, 'w':1/T}",
                            'sinfour':"{'name_func':'FourierFunc','T':T,freq_type:'principal','B_bounds':(-1,1),'nb_H':"+ str(nb_params)+"}"}
                dico_expr = {'final':'**(#ow,**(#bd,+(#trend,#sinfour)))'}
            
            else:
                 raise SystemError('implement more shotcuts here')
            
            control_obj = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
        
        else:
            list_keys_to_remove = []
            for k, v in dico_processed.items():
                bits = k.split('_')
                if(bits[0] == 'ctl'):
                    list_keys_to_remove.append(k)
                    if(bits[1] == 'final'):
                        dico_expr.update({bits[1]:v})
                    else:
                        dico_atom.update({bits[1]:v})
                    
    
            for k in list_keys_to_remove:
                del dico_processed[k]

            control_obj = pFunc_zoo.pFunc_factory.parse(dico_atom, dico_expr)['final']
        dico_processed['control_obj'] = control_obj
        return dico_processed
        
    @classmethod
    def extract_res(cls, name):
        res = ut.eval_from_file('TestBatch/res0.txt', evfunc = pFunc_base.eval_with_pFunc)
        return res

#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    # Testing has been spin off to Test/BH1D
    pass


        