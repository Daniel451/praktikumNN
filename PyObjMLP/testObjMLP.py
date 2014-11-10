


# imports
import NeuralNetworkClass as NNC


input = [[0,0],[0,1],[1,0],[1,1]]
hidden = [2]
outputLength = 1 
expectedOutput = [ [0], [1], [1], [0] ]


net = NNC.NeuralNetwork(input, hidden, outputLength, expectedOutput)

net.printNetWeights()

net.teach(10000, 0.2)

net.printNetWeights()

net.calculate([1,1])
