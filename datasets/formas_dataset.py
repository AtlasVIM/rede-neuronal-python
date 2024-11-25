from random import random, randrange, shuffle

import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

class FormasDataset:

    """
    Função que gera imagens 28x28 de um quadrado
    :param img: A imagem a devolver:
    """
    def gerar_quadrado(self,pix=28):
        img = np.zeros((pix, pix))
        ini = pix // (randrange(4,10))
        fim = pix - ini
        img[ini:fim, ini:fim] = 1
        return img

    def gerar_forma(self,pix=28):
        img = np.random.rand(pix, pix)
        return (img > 0.5).astype(float)


    def gerar_triangulo(self, pix=28):
        img = np.zeros((pix, pix))
        rand = randrange(4,8)
        r = np.array([pix // rand, 3 * pix // rand, 3 * pix // rand])
        c = np.array([pix // 2, pix // rand, 3 * pix // rand])
        rr, cc = polygon(r, c)
        img[rr, cc] = 1
        return img


    def criar_dataset(self,amostra=1000, pix=28):
        X = []
        y = []

        for _ in range(amostra):
            rand = np.random.randint(1,3)
            match rand:
                case 1:
                    X.append(self.gerar_quadrado(28))
                    y.append([1])
                case 2:
                    X.append(self.gerar_forma(28))
                    y.append([-1])
                case 3:
                    X.append(self.gerar_triangulo(28))
                    y.append([-1])

        X = np.array(X)
        y = np.array(y)
        X_achatado = X.reshape(X.shape[0], -1)  # Flatten images
        X_achatado = X_achatado / 255.0  # Normalize pixel values (0 to 1)
        return X_achatado, y

    def criar_teste(self, amostra, pix=28, racio=0.9):
        X_treino = []
        y_treino = []
        X_teste = []
        y_teste = []
        X, y = self.criar_dataset(amostra,pix)

        for i in range(len(X)):

            if i < len(X) * racio:
                X_treino.append(X[i])
                y_treino.append(y[i])
            else:
                X_teste.append(X[i])
                y_teste.append(y[i])

        X_treino = np.array(X_treino)
        X_teste = np.array(X_teste)
        y_treino = np.array(y_treino)
        y_teste = np.array(y_teste)


        return X_treino,y_treino, X_teste, y_teste

    #def test(self):
       # X_train, y_train = self.criar_dataset(1000,28)
       # plt.imshow(X_train[1], cmap='gray')
       # plt.show()


#testdata = FormasDataset()
#triangulo = testdata.gerar_triangulo()
#plt.imshow(triangulo, cmap='gray')
#plt.show()