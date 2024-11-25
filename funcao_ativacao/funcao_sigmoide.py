from .funcao_ativacao import ActivationFunctionInterface
import numpy as np

class SigmoidFunction(ActivationFunctionInterface):
    def ativar(self, x):
        return (1-(1+np.exp(x)))
    
