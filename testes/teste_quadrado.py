from datasets.formas_dataset import FormasDataset

def teste_quadrado(nn,epocas,erro_max,alpha,beta, amostra, racio_de_teste=0.9):
    print('---------- INICIO DO PROBLEMA DO QUADRADO. ---------')
    print('Foram geradas três tipos de imagens, quadrados, triângulos e uma dispersão aleatoria de pixeis.')
    print('A rede é treinada sobre um dataset com várias destas imagens e é testada sobre se a imagem dada é um quadrado ou não.')
    dataset_obj = FormasDataset()
    X_treino, y_treino, X_teste, y_teste = dataset_obj.criar_teste(amostra, 28, racio_de_teste)
    nn.treinar(X_treino, y_treino,epocas,erro_max,alpha,beta)
    resultado_previsao = nn.prever(X_teste)
    for y, i in zip(y_teste,resultado_previsao):
        print('EXPECTED OUTPUT: ',y, 'NETWORK OUTPUT', i, ' <É UM QUADRADO>' if i > 0 else ' <NÃO É UM QUADRADO>',  ' - ACCURACY:', (1-abs(y[0]-i[0]))*100, '%')
