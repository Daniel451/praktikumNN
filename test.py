__author__ = 'daniel'


from PyObjMLP import NeuralNetworkClass as NNC


a = NNC.NeuralNetwork([1,2,3], [2,3,2,6,7,88])
a.printNetWeights()
