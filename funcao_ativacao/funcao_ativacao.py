class ActivationFunctionInterface:
    
    
    def ativar(self, x):
        pass


    def derivar(self, x):
        return (self.ativar(x + 1e-12) - self.ativar(x)) / 1e-12
