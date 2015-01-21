

# imports
import NeuralNetworkClass as NNC


# data and net config
input = [[0,0],[0,1],[1,0],[1,1]]
hidden = [30]
outputLength = 1 
expectedOutput = [ [0], [1], [1], [0] ]
recurrentCount = 1


# create one net
net = NNC.NeuralNetworkElman(input, hidden, outputLength, expectedOutput, recurrentCount)

# train the net
net.teach(5000, 0.2)

# calculate output
net.calculate([1,1])
net.calculate([0,1])
net.calculate([1,0])
net.calculate([0,0])
