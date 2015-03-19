import network as net
import numpy as np

# Learn the XOR function
# 0 0 = 0
# 0 1 = 1
# 1 0 = 1
# 1 1 = 0
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# 1 input layer with 2 neurons + 1 bias
# 1 hidden layer with 2 neurons + 1 bias
# 1 output layer with 1 neuron
nn = net.Network(2,2,1)

# Learn by drawing 10000 samples randomly from our inputs
# Learning rate if too large will fail miserably, but there seems to
# be other local minima
nn.train(x,y,learn_rate=0.1,epochs=10000)

# Show our output
for sample, answer in zip(x,y):
    print (sample, nn.predict(sample))
