
import numpy as np
class Neuronio:
    """
    Initializing method
    :param d: Number of entries in the neuron
    """
    def __init__(self,d, phi):
        self.phi = phi
        self.d = d
        self.w = np.random.uniform(size=d ,low = -1, high= 1)
        self.b = np.random.uniform(low = -1,high= 1)
        self.h = 0
        self.y = 0
        self.y_deriv = 0
        self.deltaw = np.repeat(0,d)
        self.deltab = 0
    
    def propagar(self, x):
        self.h = self.w @ x + self.b
        self.y = self.phi.ativar(self.h)
        self.y_deriv = self.phi.derivar(self.h)
        return self.y
    
    def adaptar(self, delta, y_anterior, alpha, beta):
        movimento_pesos = beta * self.deltaw
        self.deltaw = (-alpha) * self.y_deriv * delta * y_anterior + movimento_pesos
        self.w = self.w + self.deltaw
        movimento_pendores = beta * self.deltab
        self.deltab = -alpha * self.y_deriv * delta + movimento_pendores
        self.b = self.b + self.deltab


    def guardar(self, ficheiro):
        ficheiro.write(f"Weights: {self.w.tolist()}\n")
        ficheiro.write(f"Bias: {self.b}\n")