import numpy as np
from neuronio import Neuronio
class DenseLayer:


    def __init__(self, de, ds, phi):
        """
        Initializing method of the Dense Layer
        :param de: Number of entry connections in the layer
        :param ds: Number of exit connections in the layer
        :param phi: Network's Activation Function
        :param neurons: Neurons within this layer, they are instanced within this function
        """

        self.de = de
        self.ds = ds
        self.phi = phi
        self.neuronios = []
        for _ in range(self.ds):
            self.neuronios.append(Neuronio(self.de, self.phi))

    @property
    def y(self):
        """
        The y property, meaning the exit values of the layer. This property is an array of all the y values given by the neurons within this layer
        """
        return np.array([n.y for n in self.neuronios])
       
    def propagar(self, x):
        y = np.array([neuronio.propagar(x) for neuronio in self.neuronios])
        return y
    
    def adaptar(self, delta, y_anterior, alpha, beta):
       for j in range(0, self.ds):
           self.neuronios[j].adaptar(delta[j], y_anterior, alpha, beta)

           