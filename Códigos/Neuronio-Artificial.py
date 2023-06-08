import numpy as np

class Perceptron(object):
  """
    n_Xj: Número de entradas
    i: Número de iterações
    eta: Coeficiente de apredizagem

    Wj: Vetor de pesos
  """
  
  def __init__(self, n_Xj, i, eta):
    self.i = i
    self.eta = eta
    self.Wj = np.zeros(n_Xj + 1)
    
  def predict(self, input):
    z = np.dot(self.Wj[1:], input) + self.Wj[0]
    if z >= 0:
      return 1
    else:
      return 0
  
  def train(self, Xj, y):
    for _ in range(self.i):
      for inputs,label in zip(Xj, y):
        prediction = self.predict(inputs)
        self.Wj[1:] += self.eta*(label - prediction)*inputs
        self.Wj[0] += self.eta*(label - prediction)

if __name__ == "__main__":
    #Entradas
    Xj = []
    Xj.append(np.array([1,1]))
    Xj.append(np.array([1,0]))
    Xj.append(np.array([0,1]))
    Xj.append(np.array([0,0]))

    #Saídas
    y = np.array([1,1,1,0])

    or_percep = Perceptron(n_Xj = 2, i=200, eta=0.1)
    or_percep.train(Xj, y)
    
    #Teste nº 1
    inputs = np.array([1,1])
    print(or_percep.predict(inputs))

    #Teste nº 2
    inputs = np.array([0,1])
    print(or_percep.predict(inputs))

    #Pesos
    print(or_percep.Wj)