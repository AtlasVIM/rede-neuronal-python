from .funcao_ativacao import ActivationFunctionInterface
import numpy as np


class ReLUFunction(ActivationFunctionInterface):
    def ativar(self, x):
        return max(0.0, x)

