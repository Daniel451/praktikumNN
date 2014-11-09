


# imports
import NeuralNetworkClass as NNC


input = [[0,0],[0,1],[1,0],[1,1]]
hidden = [2,3,2]
outputLength = 1 
expectedOutput = [ [0], [1], [1], [0] ]


net = NNC.NeuralNetwork(input, hidden, outputLength, expectedOutput)
net.printNetWeights()
