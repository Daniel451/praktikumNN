__author__ = 'daniel'


import numpy
import NeuronClass


class Layer:
    """
    Layer-class, used to model one layer of neurons
    """

    def __init__(self, neuronCount, name="no-name"):
        """

        :param neuronCount: Integer -> Amount of neurons in this layer
        :param name: String -> Name of the Layer
        :return:
        """

        self.name = str(name)
        self.dataset = []
        self.layerLength = neuronCount
        self.defaultBias = 1

        for i in range (0, self.layerLength):
            self.dataset.append(NeuronClass.Neuron(self.layerLength, self.defaultBias, i))


    def getNeurons(self):
        return self.dataset


    def getName(self):
        return self.name
