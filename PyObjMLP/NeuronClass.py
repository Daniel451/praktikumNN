__author__ = 'daniel'


import numpy


class Neuron:
    """
    class which simulates a single neuron
    """

    def __init__(self, layerLength, bias, name="no-name"):
        """

        :param layerLength: Integer -> Length of the parent layer
        :param bias: Bias of the neuron
        :param name: Name of the Neuron
        :return:
        """


        # create a numpy array for weights
        # random.uniform uniformly distributes floats
        # low <= randomFloat < high
        low = 0.1
        high = 2
        self.weights = numpy.random.uniform(low, high, layerLength)

        # set bias
        self.bias = numpy.array([bias])

        self.name = str(name)


    def retWeights(self):
        return self.weights


    def retBias(self):
        return self.bias


    def retWeightsAndBias(self):
        return numpy.concatenate((self.weights, self.bias))

    def getName(self):
        return self.name