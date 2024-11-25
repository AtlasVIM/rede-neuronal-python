import numpy as np
def test_XOR(nn,epocas,erro_max,alpha,beta):


    X = [np.array([-1,-1]), np.array([-1,1]), np.array([1,-1]), np.array([1,1])]
    Y = [np.array([-1]), np.array([1]), np.array([1]), np.array([-1])]

    print("Testing XOR Gate using Neural Network")
    nn.treinar(X,Y,epocas,erro_max,alpha, beta)
    resultado_previsao = nn.prever(X)
    print(X)
    for i in range(len(resultado_previsao)):
        print("RESULTADO DE", X[i], " ",resultado_previsao[i])



