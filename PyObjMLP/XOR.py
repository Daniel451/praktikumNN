

# imports
import NeuralNetworkClass as NNC


# data and net config
input = [[0,0],[0,1],[1,0],[1,1]]
hidden = [3]
outputLength = 1 
expectedOutput = [ [0], [1], [1], [0] ]

# create one net
net = NNC.NeuralNetwork(input, hidden, outputLength, expectedOutput)

# train the net
net.teach(1000, 0.2)

# calculate output
net.calculate([1,1])
net.calculate([0,1])
net.calculate([1,0])
net.calculate([0,0])
