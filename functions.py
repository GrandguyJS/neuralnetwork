import numpy as np

# Training set for XOR

training_data = np.array([[1, 1]])
Y = np.array([[0], [1], [1], [0]])

class NeuralNetwork():
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        if type(layers) != list:
            raise TypeError
        elif len(layers) < 2:
            raise TypeError
        self.layers = layers

    def loadWeights(self, inputWeights = None):
        if inputWeights:
            self.weights = inputWeights
        else:
            for i,v in enumerate(self.layers[:-1]):
                # Weights
                self.weights.append(np.random.randn(self.layers[i+1], v) * 0.1)
                # Biases
                self.biases.append(np.zeros((self.layers[i+1], 1)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forwardprop(self, input):

        weights, biases = self.weights, self.biases
        
        cache = [input.T]

        for i in range(len(weights)):
            cache.append(np.dot(weights[i], cache[-1]) + biases[i])

            cache.append(self.sigmoid(cache[-1]))

        return cache[-1], cache





nn1 = NeuralNetwork([2,3,1])

nn1.loadWeights()
result, cache = nn1.forwardprop(training_data)

print(cache)