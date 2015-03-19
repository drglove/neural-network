# Nick Lange
# Mar. 19th, 2015

import numpy as np
from itertools import izip_longest

# Based on example from http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php

class Network:
    def __init__(self, *layers):
        """Initialize the network with the dimensions specified in layers
        e.g. 4,6,3,2 yiels a network with 4 input neurons, two hidden layers
        with 6 and 3 neurons each, and 2 output neurons.
        Each layer is given a bias neuron.
        Weights are initialized randomly in (-1,1)
        """
        # Keep a list of matrices as weights
        # w^k_ij = weight from ith neuron in layer k-1 to jth neuron in layer k
        # k = 0,...,numlayers-1
        self.weights = []
        # Build the weights as a matrix for each pair of layers
        for n_prev, n_curr in pairwise(layers):
            # Weights go from (-1,1)
            # Add an extra weight for the bias
            self.weights.append(2*np.random.rand(n_prev + 1, n_curr) - 1)

    def activation(self, x):
        return tanh(x)

    def activation_deriv(self, x):
        return tanh_deriv(x)

    def train(self, inputs, outputs, learn_rate=0.1, epochs=1000):
        """Learn from the input data and its corresponding output.
        Repeat the learning process epoch times by repeatedly drawing a sample
        at random from the inputs and then updating the weights.
        """
        for _ in xrange(epochs):
            # Pick sample at random
            sample_index = np.random.randint(inputs.shape[0])
            sample = inputs[sample_index]
            activations = self.feedforward(sample)
            
            # Build up a list of the deltas for back-propagation
            # How poor did we do?
            error = outputs[sample_index] - activations[-1]
            delta = error*self.activation_deriv(activations[-1])
            deltas = [delta]
            for layer in xrange(len(self.weights) - 1, 0, -1):
                delta = deltas[-1].dot(self.weights[layer].T)*self.activation_deriv(activations[layer])
                # Remove the delta for the bias node, since it doesn't propagate back
                delta = np.delete(delta, -1)
                deltas.append(delta)

            # Reverse the deltas since we built them backwards
            deltas.reverse()

            # Do the back-propagation
            for layer in xrange(len(self.weights)):
                result = np.atleast_2d(activations[layer])
                delta = np.atleast_2d(deltas[layer])
                self.weights[layer] += learn_rate*result.T.dot(delta)

    def feedforward(self, sample):
        # Keep a list of the outputs at each layer, which for the input layer is trivial
        activations = [sample]
            
        # Feed forward
        for layer in xrange(len(self.weights)):
            # Append a 1 for the input to the bias weight
            activations[layer] = np.append(activations[layer], 1)
            z = self.weights[layer].T.dot(activations[layer])
            result = self.activation(z)
            activations.append(result)

        return activations

    def predict(self, inputs):
        """Return the ANN's prediction for the given input.
        """
        # Get the last feedforward result
        results = self.feedforward(inputs)
        return results[-1]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return np.exp(x) / np.square(1 + np.exp(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.square(np.tanh(x))

# See https://docs.python.org/2/library/itertools.html
from itertools import tee, izip
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)
