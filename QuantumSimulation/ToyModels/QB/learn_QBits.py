#==============================================================================
#                   ToyModelOPTIM CLASS 
# 
# Implementation of the optimization abstract class (from Utility.Optim) 
# for the toyModel (simulated via the ToyModels.ControlledSpin class)
#  
# Everything has been factored and moved to Batch BaseClass maybe no need to 
# maintain a separate file
#============================================================================== 
import sys
if(__name__ == '__main__'):
    sys.path.append("../../../")
    from QuantumSimulation.ToyModels.QB import QBits as qb
    from QuantumSimulation.Utility.Optim import Batch 
    
else:
    from ...Utility.Optim import Batch
    from . import QBits as qb

class learnerQB(Batch.BatchParametrizedControler):
    """
    Should cope with:
        + management of randomGen and mp
        + testing
        + dispatching i.e. more flexibility in creating the controler
    TODO: noise in the test through seed_noise = [(), ..., ()]
    """
    UNDERLYING_MODEL_CONSTRUCTOR = qb.Qubits


#==============================================================================
#                   TESTING
#============================================================================== 
if(__name__ == '__main__'):
    # Testing has been spin off to Test/BH1D
    pass


        