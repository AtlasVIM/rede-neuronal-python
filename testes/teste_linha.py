import numpy as np
def teste_linha(nn, epocas, erro_max,alpha, beta):
    X = [
        np.array([0,0,0,0,0,0,0,0,0]),
        np.array([1,0,0,1,0,0,1,0,0]),
        np.array([0,1,0,0,1,0,0,1,0]),
        np.array([0,0,1,0,0,1,0,0,1]),
        np.array([1,1,1,0,0,0,0,0,0]),
        np.array([0,0,0,1,1,1,0,0,0]),
        np.array([0,0,0,0,0,0,1,1,1]),
    ]
    Y = [
        np.array([-1]),
        np.array([1]),
        np.array([1]),
        np.array([1]),
        np.array([-1]),
        np.array([-1]),
        np.array([-1]),
    ]

    X_teste = [
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0]),
        np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]),
        np.array([0, 0, 1, 0, 0, 0, 0, 0, 1]),
        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 1, 0, 1, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 0]),
        np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]),
    ]

    nn.treinar(X,Y,epocas,erro_max,alpha, beta)
    resultado_previsao = nn.prever(X_teste)
    for i in range(len(resultado_previsao)):
        print(f'RESULTADO DE {X_teste[i]} {resultado_previsao[i]} ' +
              f' {"LINHA" if resultado_previsao[i] > 0.5 else "NÃO É LINHA"}')
