import numpy as np

class Network:
    def __init__(self):
        self.neurons = []
    def connect(self, neuron_from, neuron_to):
        neuron_from.set_output(neuron_to)
        neuron_to.add_input(neuron_from)
        if neuron_from not in self.neurons:
            self.neurons.append(neuron_from)
        if neuron_to not in self.neurons:
            self.neurons.append(neuron_to)

class Neuron:
    def __init__(self, input_value=None):
        self.weighted_inputs = {}
        self.input_value = input_value
        self.output_neuron = None
    def output(self):
        if self.weighted_inputs:
            y = np.sum([neuron.output()*weight for neuron, weight in self.weighted_inputs.iteritems()])
        else:
            y = self.input_value
        return self.activate(y)
    def activate(self, x):
        if not x:
            print self.weighted_inputs
        return sigmoid(x)
    def add_input(self, neuron, weight=None):
        if not weight:
            weight = np.random.rand()
        self.weighted_inputs[neuron] = weight
    def set_output(self, neuron):
        self.output_neuron = neuron

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
