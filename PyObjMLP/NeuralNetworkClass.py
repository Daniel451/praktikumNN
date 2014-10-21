__author__ = 'daniel'


import numpy
import LayerClass


class NeuralNetwork:

    def __init__(self, l_in, hiddenLayer):
        """

        :param l_in: InputLayer -> List of input values
        :param hiddenLayer: List of Hidden Layer
                            [2,3,2] would create 3 Hidden Layers with 2, 3 and 2 Neurons
        :return:
        """

        self.input = l_in
        self.layers = []
        self.output = []

        for i in range(0, len(hiddenLayer)):
            self.layers.append(LayerClass.Layer(hiddenLayer[i], i))


    def printNetWeights(self):

        print "######################################################"
        print "-- Gewichte --"

        for layer in self.layers:
            print "Layer " + layer.getName()

            for neuron in layer.getNeurons():
                print "Neuron " + neuron.getName() + " " + str(neuron.retWeights())







