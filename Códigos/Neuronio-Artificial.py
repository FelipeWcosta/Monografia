import numpy as np
class perceptron(object):
    """
        eta: Coeficiente de treinamento
        i: Número de iterações
        n_Xj: Número de entradas

        Wj: Vetor de pesos iniciado com zeros
    """
    def __init__(self, n_Xj, eta = 0.01, i = 100):
        self.eta = eta
        self.i = i
        self.Wj = np.zeros(n_Xj + 1)

    def predict(self, inputs):
        z = np.dot(inputs, self.Wj[1:]) + self.Wj[0]
        if z >= 0:
            sigma = 1
        else:
            sigma = 0
        return sigma
    
    """
        Xj: Entradas de treinamento
        y: Saídas esperadas
    """

    def fit(self, Xj, y):
        for _ in range(self.i):
            for inputs, labels in zip(Xj, y):
                predicao = self.predict(inputs)
                self.Wj += self.eta*(labels - predicao)*inputs
                self.Wj += self.eta*(labels - predicao)

if __name__ == "__main__":

    #Entradas para treinamento
    Xj = []
    Xj.append(np.array([0, 0]))
    Xj.append(np.array([0, 1]))
    Xj.append(np.array([1, 0]))
    Xj.append(np.array([1, 1]))
    print(Xj)

    #Saídas esperadas
    y = np.array([1, 0, 0, 1])

    #Neurônio artificial
    na = perceptron(0.01, 50, 2)