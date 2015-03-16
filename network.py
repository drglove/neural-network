import numpy as np
from itertools import izip_longest

np.random.seed(10)

class Network:
    def __init__(self, dim=[1, 1]):
        self.layers = {'input' : None, 'output' : None, 'hidden' : []}
        # Create our layers
        self.layers['input'] = NetworkLayer(dim[0])
        for hidden_layer_neurons in dim[1:-1]:
            self.layers['hidden'].append(NetworkLayer(hidden_layer_neurons))
        self.layers['output'] = NetworkLayer(dim[-1])
        
        #Connect the layers
        if self.layers['hidden']:
            self.layers['input'].connect(self.layers['hidden'][0])
            #Connect pairs of hidden layers
            for hidden_from, hidden_to in zip(self.layers['hidden'][:-1], self.layers['hidden'][1:]):
                hidden_from.connect(hidden_to)
            self.layers['hidden'][-1].connect(self.layers['output'])
        else:
            self.layers['input'].connect(self.layers['output'])

    def connect(self, layer_from, layer_to, weights=[[]], from_is_input=False, to_is_output=False):
        layer_from.connect(layer_to, weights)
        if from_is_input and to_is_output:
            self.layers['input'] = layer_from
            self.layers['output'] = layer_to
        elif from_is_input and not to_is_output:
            self.layers['input'] = layer_from
            self.layers['hidden'].append(layer_to)
        elif not from_is_input and not to_is_output:
            self.layers['hidden'].append(layer_to)
            self.layers['hidden'].append(layer_from)
        elif not from_is_input and to_is_output:
            self.layers['hidden'].append(layer_from)
            self.layers['output'] = layer_to
    def set_input(self, input_values):
        if self.layers['input']:
            self.layers['input'].set_input(input_values)
        else:
            print 'No input layer set on the network!'
    def output(self):
        return self.layers['output'].output()
    def train(self, input_values, truth_values, learn_rate=1, epochs=100):
        self.set_input(input_values)
        # Calculate change in weights from backpropagation
        # Do the output layer first and work backwards
        t = np.array(truth_values)
        for _ in xrange(epochs):
            layer = self.layers['output']
            while layer.prev_layer and self.layers['input'] != layer:
                o = np.array(layer.output())
                weights_out = np.array([n.weighted_outputs.values() for n in layer.neurons])
                deriv = o * (1 - o)
                if layer is self.layers['output']:
                    d = (o - t) * deriv
                else:
                    d = weights_out.dot(d) * deriv
                for neuron_to, delta_w in zip(layer.neurons, d):
                    for neuron_from in neuron_to.weighted_inputs.keys():
                        x = neuron_from.output()
                        neuron_from.update_weight(neuron_to, learn_rate*delta_w*x, incremental=True)
                layer = layer.prev_layer

class NetworkLayer:
    def __init__(self, n_neurons=1):
        self.neurons = [Neuron() for _ in xrange(n_neurons)]
        self.next_layer = None
        self.prev_layer = None
    def connect(self, layer_to, weights=[]):
        self.next_layer = layer_to
        layer_to.prev_layer = self
        for neuron_to, input_weights in izip_longest(layer_to.neurons, weights, fillvalue=[]):
            for neuron_from, weight in izip_longest(self.neurons, input_weights):
                neuron_from.add_output(neuron_to, weight)
    def set_input(self, input_values):
        for i in xrange(len(self.neurons)):
            self.neurons[i].set_input(input_values[i])
    def output(self):
        return [neuron.output() for neuron in self.neurons]

class Neuron:
    def __init__(self, name=None, input_value=None):
        if not name:
            self.name = Neuron.get_next_name()
        self.weighted_inputs = {}
        self.weighted_outputs = {}
        self.input_value = input_value
        self.bias = 0.0
    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)
    name = 'a'
    @staticmethod
    def get_next_name():
        curr = Neuron.name
        Neuron.name = chr(ord(Neuron.name) + 1)
        return curr
    def output(self):
        if self.input_value:
            y = self.input_value
        else:
            y = np.sum([neuron.output()*weight for neuron, weight in self.weighted_inputs.iteritems()]) + self.bias
        return self.activate(y)
    def activate(self, x):
        return 2*sigmoid(x) - 1
    def add_input(self, neuron, weight=None):
        if not weight:
            weight = 2*np.random.rand() - 1
        self.weighted_inputs[neuron] = weight
    def add_output(self, neuron, weight=None):
        if not weight:
            weight = 2*np.random.rand() - 1
        self.weighted_outputs[neuron] = weight
        neuron.add_input(self, weight)
    def update_weight(self, neuron_to, weight, incremental=False):
        if not incremental:
            self.weighted_outputs[neuron_to] = weight
            neuron_to.weighted_inputs[self] = weight
        else:
            self.weighted_outputs[neuron_to] += weight
            neuron_to.weighted_inputs[self] += weight
    def set_input(self, input_value):
        self.input_value = input_value

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
