from funcao_ativacao.funcao_relu import ReLUFunction
from testes.teste_quadrado import teste_quadrado
from funcao_ativacao.funcao_sigmoide import SigmoidFunction
from funcao_ativacao.funcao_tanh import TanhFunction
from rede_neuronal import NeuralNetwork
import matplotlib.pyplot as plt

# FUNÇÕES DE ATIVAÇÃO
tanh = TanhFunction()
phi = tanh


forma = [784, 28, 2, 1]
nn = NeuralNetwork(forma, phi)
epocas = 400
erro_max = 0.05
alpha = 0.01
beta = 0.3


#RESOLUÇÃO DO PROBLEMA
teste_quadrado(nn,epocas,erro_max,alpha,beta, 50, 0.8)

# GRÁFICO DO ERRO
x,y = zip(*nn.err_epoca)
plt.plot(x,y)
plt.show()



