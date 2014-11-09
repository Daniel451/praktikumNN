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

        # last output of all neurons on this layer
        self.lastOutput = numpy.zeros(self.layerLength)

        # last error of all neurons on this layer 
        self.error = numpy.zeros(self.layerLength)

        # last innerActivation
        self.innerActivation = numpy.zeros(self.layerLength)

        # weights
        
        
        # bias
        self.bias = numpy.ones(self.layerLength)    

        for i in range (0, self.layerLength):
            self.neurons.append(NeuronClass.Neuron(parentLayerLength, self.defaultBias, i))


    def getLastOutput(self):
        return self.lastOutput


    def setLastOutput(self, nparray):
        self.lastOutput = nparray


    def getLastInnerActivation(self):
        return self.innerActivation


    def setLastInnerActivation(self, nparray):
        self.innerActivation = nparray


    def getError(self):
        return self.error


    def setError(self, nparray):
        self.error = nparray


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
