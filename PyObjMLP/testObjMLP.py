


# imports
import NeuralNetworkClass as NNC

from speedtest import Speedtest as sp


mysp = sp()


input = [[0,0],[0,1],[1,0],[1,1]]
hidden = [3]
outputLength = 1 
expectedOutput = [ [0], [1], [1], [0] ]


net = NNC.NeuralNetwork(input, hidden, outputLength, expectedOutput)

net.printNetWeights()

mysp.record("start")
net.teach(1000, 0.2)
mysp.record("ende")

net.printNetWeights()

net.calculate([1,1])
net.calculate([0,1])
net.calculate([1,0])
net.calculate([0,0])

mysp.printRecords()
