import numpy as np
class CamadaEntrada:
    def __init__(self, ds):
        self.ds = ds
        self.y = np.repeat(0, ds)
        
    def propagar(self, x):
            self.y = x
            return self.y