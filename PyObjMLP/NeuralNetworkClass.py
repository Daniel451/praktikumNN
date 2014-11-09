__author__ = 'daniel'


import numpy
import LayerClass
import sys

class NeuralNetwork:

    def __init__(self, in_input, hiddenLayerList, outputLayerLength, out_expected, label="no-name"):
        """
        :param in_input: InputLayer -> List of lists which hold input value/data for
                                        each input neuron

                                        Must have same length as out_expected

                            Example for XOR-Problem:
                            [ [0,0], [0,1], [1,0], [1,1] ]

        :param hiddenLayerList: List of Hidden Layers

                            Example:
                            [2,3,2] would create 3 Hidden Layers with 2, 3 and 2 Neurons

        :param outputLayerLength: Integer -> Number of output Neurons

        :param out_expected: List containing lists which hold expected output values/data

                                Must have same length as in_input

                            Example for XOR-Problem:
                            [ [0], [1], [1], [0] ]
        """
        
        self.checkForErrors(in_input, hiddenLayerList, outputLayerLength, out_expected)

        # set the label of the net
        self.label = str(label)

        # set the input layer data and length
        self.inputLayer = in_input
        self.inputLayerLength = len(self.inputLayer[0])

        # set the exptected output values for each entry in inputLayer
        self.outExpected = out_expected

        # initialize hiddenLayer
        self.hiddenLayer = []

        # insert the length of input layer to the beginning of the hiddenLayerList
        # this is needed to calculate the weights for layer 1 of the hidden layers
        hiddenLayerList.insert(0, self.inputLayerLength)
       
        # create all hidden layers in one loop
        for i in range(1, len(hiddenLayerList)):
            # append one complete layer to hiddenLayer
            # Layer( neuronCount, parentLayerLength, label )
            self.hiddenLayer.append(LayerClass.Layer(hiddenLayerList[i], hiddenLayerList[i-1], i))

        # create the output layer
        # Layer( neuronCount, parentLayerLength, label )
        self.outputLayer = LayerClass.Layer(outputLayerLength, self.hiddenLayer[len(self.hiddenLayer)-1].getLength(), "Output") 


    def teach(self):
        return


    def calculate(self):
        return


    def printNetWeights(self):

        print "######################################################"
        print "     neural net: " + self.label
        print "######################################################"
        print ""
        print "###############"
        print "### weights ###"
        print "###############"

        for layer in self.hiddenLayer:
            print "\n--- Layer: " + layer.getLabel() + " --- (" + str(layer.getLength()) + " neurons total)"

            for neuron in layer.getNeurons():
                print "Neuron " + neuron.getLabel() + "\n" + str(neuron.getWeights())

        print "\n--- Layer: " + self.outputLayer.getLabel() + " --- (" + str(self.outputLayer.getLength()) + " neurons total)"
        for neuron in self.outputLayer.getNeurons():
            print "Neuron " + neuron.getLabel() + "\n" + str(neuron.getWeights())


    def checkForErrors(self, in_input, hiddenLayerList, outputLayerLength, out_expected):
        """ checks for errors before starting actual init things """

        errStr = "!!!!!!!!!!!!!\n"
        errStr += "!!! ERROR !!!\n"
        errStr += "!!!!!!!!!!!!!\n"
        errStr += "\n"

        # check if input data and expected output and hiddenLayerList are lists
        if not ( (type(in_input) is list) and (type(out_expected) is list) and (type(hiddenLayerList) is list) ):
            errStr += "Error: in_input, hiddenLayerList and out_expected must be lists"
            sys.exit(errStr)

        if not ( (type(outputLayerLength) is int) and (outputLayerLength > 0) ):
            errStr += "Error: outputLayerLength has to be an integer > 0"
            sys.exit(errStr)

        # check input data length and expected output data length
        if len(in_input) != len(out_expected):
            errStr += "Error: the length of the input data list does not match the length "
            errStr += "of the expected output data list"
            sys.exit(errStr)


