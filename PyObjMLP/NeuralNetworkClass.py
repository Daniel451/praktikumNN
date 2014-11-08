__author__ = 'daniel'


import numpy
import LayerClass


class NeuralNetwork:

    def __init__(self, in_input, hiddenLayerList, out_input, out_expected):
        """
        :param in_input: InputLayer -> List of input values lists
                            Example for XOR-Problem:
                            [ [0,0], [0,1], [1,0], [1,1] ]
        :param hiddenLayerList: List of Hidden Layers
                            [2,3,2] would create 3 Hidden Layers with 2, 3 and 2 Neurons
        :param out_input: 
        """

        self.inputLayer = in_input
        self.outExpected = out_expected
        self.hiddenLayer = []
        self.outputLayer = []

        for i in range(0, len(hiddenLayerList)):
            self.hiddenLayer.append(LayerClass.Layer(hiddenLayerList[i], i))


    def printNetWeights(self):

        print "######################################################"
        print "-- Gewichte --"

        for layer in self.layers:
            print "Layer " + layer.getName()

            for neuron in layer.getNeurons():
                print "Neuron " + neuron.getName() + " " + str(neuron.retWeights())







