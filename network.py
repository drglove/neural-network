import numpy as np

class Network:
    def __init__(self, learn_rate=1):
        self.layers = {'input' : None, 'output' : None, 'hidden' : []}
        self.learn_rate = learn_rate
    def connect(self, layer_from, layer_to, from_is_input=False, to_is_output=False):
        layer_from.connect(layer_to)
        if from_is_input:
            self.layers['input'] = layer_from
        else:
            self.layers['hidden'].append(layer_from)

        if to_is_output:
            self.layers['output'] = layer_to
        else:
            self.layers['hidden'].append(layer_to)
    def set_input(self, input_values):
        if self.layers['input']:
            self.layers['input'].set_input(input_values)
        else:
            print 'No input layer set on the network!'
    def output(self):
        return self.layers['output'].output()
    def train(self, input_values, truth_values):
        self.set_input(input_values)
        # TODO: Calculate change in weights from backpropagation
        # Do the output layer first and work backwards
        o = np.array(self.layers['output'].output())
        t = np.array(truth_values)
        d = (o - t) * o * (1 - o)
        x = self.layers['output'].prev_layer.output()
        for neuron in self.layers['output'].prev_layer.neurons:
            # do things
        # Need dE/dw_(ij) so need xi

class NetworkLayer:
    def __init__(self, n_neurons=1):
        self.neurons = [Neuron() for _ in xrange(n_neurons)]
        self.next_layer = None
    def connect(self, layer_to):
        self.next_layer = layer_to
        layer_to.prev_layer = self
        for neuron_from in self.neurons:
            for neuron_to in layer_to.neurons:
                neuron_from.set_output(neuron_to)
                neuron_to.add_input(neuron_from)
    def set_input(self, input_values):
        for i in xrange(len(self.neurons)):
            self.neurons[i].set_input(input_values[i])
    def output(self):
        return [neuron.output() for neuron in self.neurons]

class Neuron:
    def __init__(self, input_value=None):
        self.weighted_inputs = {}
        self.input_value = input_value
        self.output_neuron = None
        self.bias = 0.1
    def output(self):
        if self.input_value:
            y = self.input_value
        else:
            y = np.sum([neuron.output()*weight for neuron, weight in self.weighted_inputs.iteritems()]) + self.bias
        return self.activate(y)
    def activate(self, x):
        return sigmoid(x)
    def add_input(self, neuron, weight=None):
        if not weight:
            weight = np.random.rand()
        self.weighted_inputs[neuron] = weight
    def set_input(self, input_value):
        self.input_value = input_value
    def set_output(self, neuron):
        self.output_neuron = neuron

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
