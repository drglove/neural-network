from network import *
net = Network()
layer_in = NetworkLayer(3)
layer_out = NetworkLayer(2)
layer_hidden = NetworkLayer(6)
net.connect(layer_in, layer_hidden, from_is_input=True)
net.connect(layer_hidden, layer_out, to_is_output=True)
net.train([1, 2, 555], [3, -1])
print net.output()
