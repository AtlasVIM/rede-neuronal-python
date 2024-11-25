from .funcao_ativacao import ActivationFunctionInterface
import numpy as np

class TanhFunction(ActivationFunctionInterface):
    def ativar(self, x):
        return np.tanh(x)


    def derivar(self, x):
        return (self.ativar(x + 1e-8) - self.ativar(x)) / 1e-8
