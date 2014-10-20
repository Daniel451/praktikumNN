__author__ = 'daniel'


import numpy


class Neuron:


    def __init__(self, layerLength):
        # create a numpy array for weights
        # random.uniform uniformly distributes floats
        # low <= randomFloat < high
        low = 0.1
        high = 2
        self.weights = numpy.random.uniform(low, high, layerLength)