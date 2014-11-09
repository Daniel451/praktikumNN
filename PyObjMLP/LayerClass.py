__author__ = 'daniel'


import numpy
import NeuronClass


class Layer:
    """
    Layer-class, used to model one layer of neurons
    """

    def __init__(self, neuronCount, parentLayerLength, label="no-name"):
        """

        :param neuronCount: Integer -> Amount of neurons in this layer
        :param parentLayerLength: Length (amount of neurons/inputs) of the parent layer
        :param label: String -> Name of the Layer
        :return:
        """

        self.label = str(label)
        self.neurons = []
        self.layerLength = neuronCount
        self.defaultBias = 1

        for i in range (0, self.layerLength):
            self.neurons.append(NeuronClass.Neuron(parentLayerLength, self.defaultBias, i))


    def getNeurons(self):
        return self.neurons


    def getAllWeightsOfNeurons(self):

        buffer = []

        for neuron in self.neurons:

            buffer.append(neuron.getWeights())

        return numpy.array(buffer)


    def getAllBiasOfNeurons(self):

        buffer = []

        for neuron in self.neurons:

            buffer.append(neuron.getBias())

        return numpy.array(buffer)


    def getLength(self):
        return self.layerLength


    def getLabel(self):
        return self.label
