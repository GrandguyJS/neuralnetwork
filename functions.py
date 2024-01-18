import numpy as np

# Training set for XOR



class NeuralNetwork():
    def __init__(self, layers): # When initializing the NeuralNetwork object, you create all layers, it can have as many layers, as you want it to,

        # Check if layers is a valid list with at least 2 layers

        if type(layers) != list:
            raise TypeError
        elif len(layers) < 2:
            raise TypeError

        # Set the layers and set run = False, as we have not yet run the input

        self.layers = layers
        self.run = False

    def loadWeights(self, inputWeights = None, biases = None):  # Function to load the weights of the neural network. Either you specify your own weights and biases or they get created randomly.
        if inputWeights:
            if not biases:  # If we do have weights inputed but not biases, we will simply generate them randomly
                self.biases = []
                for i in range(0, self.layers[-1]):
                    self.biases.append(np.zeros((self.layers[i+1], 1)))     # This generates biases randomly, in an array size of (next layer size, 1)
            else:
                self.weights = inputWeights     # If we do have both weights and biases we just set them in the object
                self.biases = biases

        else:   # If we don't have weights inputed we will generate weights and biases randomly, we won't take inputed biases, as it doesn't make any sense to have biases before the weights
            self.weights = []          
            self.biases = []
            for i,v in enumerate(self.layers[:-1]):     # In this loop, we go to each layer of the Neural Network (layer sizes) up to the second to last, because we don't want weights after that
                # Weights
                self.weights.append(np.random.randn(self.layers[i+1], v) * 0.1)     # We create weights randomly, in an array with the dimensions of the next layer, and the current one
                # Biases
                self.biases.append(np.zeros((self.layers[i+1], 1)))     # We create biases (initially only as zeros) in an array with the dimensions of the next layer and 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))     # Defining the Sigmoid function so that we only get values between 0 and 1

    def forwardprop(self, X, solution = None):   # Defining the actual function that calculates the ouput based on input, weights and biases

        weights, biases = self.weights, self.biases     # We firstly initialize the weights and biases
        
        self.inputValue = X  # Set the input of the object

        cache = [X.T]   # We initialize the list cache, that will store all neuron values and their 'sigmoided' value + the end result

        for i in range(len(weights)):
            cache.append(np.dot(weights[i], cache[-1]) + biases[i])     # Basically, each iteration we matrix multiply the last value of the list with the n-th weights and add the biases

            cache.append(self.sigmoid(cache[-1])) # Then we get the sigmoid of that and append that to cache. So the top of the array always is the neurons of the last layer, we have computed

        self.run = True     # We set run to True so the object knows that
        self.cache = cache  # We also set the cache of the object
        
        if solution is not None:    # Check if we also have the expected output passed in the function. If yes, calculate the error and output that too
            self.solution = solution
            error = self.loss_calculation(solution)    # Get the error

            return cache[-1][0], cache, error     # We return the result (cache[-1]) and the cache itself that will later be used for back-propagation and the error
        else:
            return cache[-1][0], cache     # We return the result (cache[-1]) and the cache itself that will later be used for back-propagation. The error will have to get calculated

    def loss_calculation(self, y):   # With this function we can calculate the error in respect to the true value we expected

        if not self.run:    # We check if we have run the neural network
            raise TypeError

        result = self.cache[-1] # Assign the result

        # Some maths to calculate the loss

        m = y.shape[0]

        loss = -(1/m) * np.sum(y*np.log(result) + (1-y)*np.log(1-result))
        
        # Set and return the loss
        self.loss = loss
        return loss

    def backwards_prop(self):
        weights, biases, cache, X, Y, output = self.weights, self.biases, self.cache, self.inputValue, self.solution, self.cache[-1]

        raise NotImplementedError

# Basic training data set with XOR. I know that the Neural network will just memorize it, but its a simple example and all I want now is to see if the back-propagation and learning is working

X = [[0,0], [0,1], [1, 0], [1,1]]
Y = [[0], [1], [1], [0]]

n = 3   # First n trainign sets
X_train = X[0:n]
Y_train_results = Y[0:n]

X_test = X[n:]
Y_test_results = Y[n:]

# Object nn1

nn1 = NeuralNetwork([2,3,1])    # Initialize the Neural Network Layers

nn1.loadWeights()   # Load the weights
result, cache, error = nn1.forwardprop(np.array(X_train), np.array(Y_train_results))  # Get the result, cache and error

nn1.backwards_prop()





