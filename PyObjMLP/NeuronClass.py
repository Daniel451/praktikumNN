__author__ = 'daniel'


import numpy


class Neuron:
    """
    class which simulates a single neuron
    """

    def __init__(self, parentLayerLength, bias, label="no-name"):
        """

        :param parentLayerLength: Integer -> Length of the parent layer (needed for dimension of weights)
        :param bias: Bias of the neuron
        :param label: Name of the Neuron
        :return:
        """


        # create a numpy array for weights
        # random.uniform uniformly distributes floats
        # low <= randomFloat < high
        low = 0.1
        high = 2
        self.weights = numpy.random.uniform(low, high, parentLayerLength)

        # set bias
        self.bias = numpy.array([bias])

        self.label = str(label)


    def getWeights(self):
        return self.weights


    def getBias(self):
        """ returns the bias """
        return self.bias


    def getWeightsAndBias(self):
        """ returns Weights and Bias as one concatenated numpy array """
        return numpy.concatenate((self.weights, self.bias))


    def getLabel(self):
        return self.label
