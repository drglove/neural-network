from network import *
# example from 3.1 of below
# https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
net = Network()
layer_in = NetworkLayer(2)
layer_hidden = NetworkLayer(2)
layer_out = NetworkLayer(1)
net.connect(layer_in, layer_hidden, weights=[[0.1, 0.8], [0.4, 0.6]], from_is_input=True)
net.connect(layer_hidden, layer_out, weights=[[0.3, 0.9]], to_is_output=True)
net.train([0.35, 0.9], [0.5])
print net.output()
