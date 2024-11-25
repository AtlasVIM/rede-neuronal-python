from camada_entrada import CamadaEntrada
from camada_densa import DenseLayer
import matplotlib.pyplot as plt
import numpy as np



class NeuralNetwork:
    def __init__(self, forma, phi):
        self.err_epoca = []
        self.N = len(forma)
        self.camadas = []
        self.camadas.append(CamadaEntrada(forma[0]))

        for n in range(1, self.N):
            detemp = forma[n - 1]
            dstemp = forma[n]
            layer = DenseLayer(detemp, dstemp, phi)
            self.camadas.append(layer)


    def delta_saida(self, yn, y):
        return [yn[k] - y[k] for k in range(len(y))]

    def prever(self, X):
        print("--- PREVISÃO ---")
        Y =[self.propagar(x) for x in X]
        return Y


    def propagar(self, x):
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y

    def retropropagar(self, delta_err_saida, alpha, beta):
        delta = delta_err_saida

        for n in range(self.N - 1, 0,-1):
            y_anterior = self.camadas[n - 1].y
            dn_anterior = self.camadas[n - 1].ds
            dn = self.camadas[n].ds
            neuronios = self.camadas[n].neuronios
            delta_anterior = [sum(neuronios[j].w[i]*delta[j]*neuronios[j].y_deriv for j in range(dn)) for i in range(dn_anterior)]
            self.camadas[n].adaptar(delta, y_anterior, alpha, beta)
            delta = delta_anterior;
    def adaptar(self, x, y, alpha, beta):
        yn = self.propagar(x)
        delta_err_saida = self.delta_saida(yn, y)
        self.retropropagar(delta_err_saida, alpha, beta)
        K = len(delta_err_saida)
        err_med = 1/K * (sum([delta_err_saida[k] for k in range(K)])**2)

        return err_med

    def treinar(self, X, Y, epocas, err_max, alpha, beta):
        for n in range(epocas):
            print("--- EPOCA ", n, " ---")
            err = 0
            for x,y in zip(X,Y):
                err_x = self.adaptar(x,y,alpha,beta)
                err = max(err, err_x)

            self.err_epoca.append((n, err))
            if err <= err_max:
                print(err)
                break
            print (err)
            
    def guardar_pesos(self, ficheiro):
        with open(ficheiro, 'w') as f:
            for i in range(1, len(self.camadas)):
                f.write(f"Layer {i+1}:\n")
                for j in range(len(self.camadas[i].neuronios)):
                    self.camadas[i].neuronios[j].guardar(f)
                f.write("\n")

    def carregar_pesos(self, ficheiro):
        with open(ficheiro, 'r') as f:
            lines = f.readlines()

        cam_num = 0
        neu_num = 0
        for line in lines:
            if "Camada" in line:
                layer_num = int(line.strip().split()[1]) - 1
                neuron_num = 0
            elif "Pesos" in line:
                weights = eval(line.split(":", 1)[1].strip())  # Convert string to list
            elif "Bias" in line:
                bias = float(line.split(":", 1)[1].strip())
                # Load weights and bias into the neuron
                self.camadas[cam_num].neurons[neuron_num].load(weights, bias)
                neuron_num += 1
