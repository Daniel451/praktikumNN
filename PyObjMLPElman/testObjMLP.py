


# imports
import NeuralNetworkClass as NNC

from speedtest import Speedtest as sp
from world import world_digits as wd
from collections import OrderedDict

mysp = sp()
mywd = wd()


input = list(mywd.sensor())

hidden = [30]
outputLength = 7

Output = OrderedDict()

for i in range(0,36):
    Output[i] = [ int(item) for item in list(bin(i)[2:].rjust(7,"0")) ]

expectedOutput = [ val for key,val in Output.items() ]


net = NNC.NeuralNetwork(input, hidden, outputLength, expectedOutput)

mysp.record("start")
net.teach(20000, 0.2)
mysp.record("ende")

net.printNetWeights()

net.calculate(list(input[0]), Output[0]) # guess zero
net.calculate(list(input[1]), Output[1]) # guess one
net.calculate(list(input[2]), Output[2]) # guess two
net.calculate(list(input[3]), Output[3]) # guess three
net.calculate(list(input[10]), Output[10]) # guess a
net.calculate(list(input[17]), Output[17]) # guess h
net.calculate(list(input[18]), Output[18]) # guess i

mysp.printRecords()
