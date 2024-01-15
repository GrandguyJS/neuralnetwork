import numpy as np

# Training set for XOR

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

class NeuralNetwork():
    def __init__(self):
        self.weights = []
        self.biases = []
        
    def AddLayers(self, layers):
        if type(layers) != list:
            raise TypeError
        elif len(layers) < 3:
            raise TypeError
        self.layers = layers
        return True

    def loadWeights(self, inputWeights = None):
        if inputWeights:
            raise NotImplementedError
        else:
            for i,v in enumerate(self.layers[:-1]):
                # Weights
                self.weights.append(np.random.randn(self.layers[i+1], v) * 0.1)
                # Biases
                self.biases.append(np.zeros((self.layers[i+1], 1)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forwardprop(self, X):

        weights, biases = self.weights, self.biases

        cache = [X.T]
        for i in range(len(weights)):
            cache.append(np.dot(weights[i], cache[-1]) + biases[i])
            cache.append(self.sigmoid(cache[-1]))

        return cache[-1], cache





nn1 = NeuralNetwork()
nn1.AddLayers([2, 3, 1])
nn1.loadWeights()
result, cache = nn1.forwardprop(np.array([[0], [0]]))

print(result)