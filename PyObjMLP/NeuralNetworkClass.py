__author__ = 'daniel'


from scipy.special import expit

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
        
        self.__checkForErrors(in_input, hiddenLayerList, outputLayerLength, out_expected)

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
        
        # make all layers easy accessable
        self.hiddenAndOutputLayer = []
        self.hiddenAndOutputLayer += self.hiddenLayer
        self.hiddenAndOutputLayer.append(self.outputLayer)
        

    def teach(self, iterations = 1000, epsilon=0.2):
        """
            train the net

            :param iterations: number of learning steps to make
            :param epsilon: learning rate
        """
        
       
        for i in range(0, iterations):
            # set input and expectedOutput
            randomSelect = numpy.random.randint(0, len(self.inputLayer))

            input = numpy.array(self.inputLayer[randomSelect])
            self.currentOutExpected = numpy.array(self.outExpected[randomSelect])

            # feedforward
            self.output = self.__feedforward(input)

            # calculate errors
            self.__updateErrors()       

            # update weights
            self.__updateWeights(input, epsilon)
            
            # update bias
            self.__updateBias(epsilon)


    def __updateErrors(self):
        
        debug = False 

        # calculate the erros
       
        # for index in range ( endOfTheList, beginningOfTheList, decrement index by 1 )
        # so this loop starts at the end of all layers (outputLayer)
        # and iterates its way to the top
        for i in range( len(self.hiddenAndOutputLayer) - 1, -1, -1 ):
            # error for output layer --- first iteration
            if i == (len(self.hiddenAndOutputLayer)-1):
                self.outputLayer.setError( self.currentOutExpected - self.output )
                if debug:
                    self.__print("output error:", self.outputLayer.getError())
            # error for hidden layers
            else: 
                # get current layer to update its error
                currentLayer = self.hiddenAndOutputLayer[i]

                # get the underlyingLayer (lambda+1) for backpropagation/calculation
                underlyingLayer = self.hiddenAndOutputLayer[i+1]
                # calculate new error
                currentLayer.setError( 
                       self.transferFunctionDeriv( currentLayer.getLastInnerActivation() )
                        *
                        numpy.dot( 
                            underlyingLayer.getError(),
                            underlyingLayer.getAllWeightsOfNeurons()
                            )
                        )

                if debug:
                    print("")
                    self.__print("layer", i)
                    self.__print("activation",
                            currentLayer.getLastInnerActivation())
                    self.__print("_deriv(activation)",
                            self.transferFunctionDeriv(currentLayer.getLastInnerActivation()))
                    self.__print("underlyingLayer.error",
                            underlyingLayer.getError())
                    self.__print("underlyingLayer.weights",
                            underlyingLayer.getAllWeightsOfNeurons())
                    self.__print("error x weights",
                            numpy.dot(underlyingLayer.getError(),underlyingLayer.getAllWeightsOfNeurons()) )
                    self.__print("new error (acti * (error x weights))",
                            currentLayer.getError())


    def __print(self, label, item):
        print(str(label) + " :\n" + str(item))


    def __updateBias(self, epsilon):
        """
            update the bias of the net
        """

        for layer in self.hiddenAndOutputLayer:

            layer.setBias( layer.getAllBiasOfNeurons() + ( epsilon * layer.getError() )  )


    def __updateWeights(self, u_input, epsilon):
        """
           update the weights of the net
        """
        
        debug = False

        # for each layer update the weights at once
        for key, layer in enumerate(self.hiddenAndOutputLayer):
            
            # for the first hidden layer it is epsilon * error * input
            if key == 0:
                layer.setWeights(
                        layer.getAllWeightsOfNeurons()
                        +
                        (
                         epsilon
                         *
                         (
                          layer.getError()[numpy.newaxis].T
                          *
                          u_input
                         )
                        )
                        )
                if debug:
                    print("")
                    self.__print("weights", layer.getAllWeightsOfNeurons())
                    self.__print("epsilon", epsilon)
                    self.__print("layer error", layer.getError()[numpy.newaxis].T)
                    self.__print("input", u_input)
                    self.__print("error x input", layer.getError()[numpy.newaxis].T * u_input)
                    self.__print("epsilon * (error x input)", epsilon
                             *
                             (
                              layer.getError()[numpy.newaxis].T
                              *
                              u_input
                             ))
                    self.__print("new weights: ", layer.getAllWeightsOfNeurons()
                            +
                            (
                             epsilon
                             *
                             (
                              layer.getError()[numpy.newaxis].T
                              *
                              u_input
                             )
                            ))
            # for all other hidden layers and output layer the calculation
            # is epsilon * error * output_of_parent_layer
            else:
                layer.setWeights(
                        layer.getAllWeightsOfNeurons()
                        +
                        ( 
                         epsilon 
                         * 
                         (
                             layer.getError()[numpy.newaxis].T
                             *
                             self.hiddenAndOutputLayer[key-1].getLastOutput()
                         )
                        )
                        )
                if debug:
                    print("")
                    self.__print("weights", layer.getAllWeightsOfNeurons())
                    self.__print("epsilon", epsilon)
                    self.__print("layer error", layer.getError()[numpy.newaxis].T)
                    self.__print("input (parent output)", self.hiddenAndOutputLayer[key-1].getLastOutput())
                    self.__print("error x input", layer.getError()[numpy.newaxis].T * self.hiddenAndOutputLayer[key-1].getLastOutput())
                    self.__print("epsilon * (error x input)", epsilon
                             *
                             (
                              layer.getError()[numpy.newaxis].T
                              *
                              self.hiddenAndOutputLayer[key-1].getLastOutput()
                             ))
                    self.__print("new weights: ", layer.getAllWeightsOfNeurons()
                            +
                            (
                             epsilon
                             *
                             (
                              layer.getError()[numpy.newaxis].T
                              *
                              self.hiddenAndOutputLayer[key-1].getLastOutput()
                             )
                            ))


    def __feedforward(self, f_input):
        """
            does an feedforward activation of the whole net

            :param f_input: has to be a numpy array matching the dimension of InputLayer
        """

        ######################
        ### initialization ###
        ######################

        # set last_out to current input
        last_out = f_input

        #######################################
        ### start calculating - feedforward ###
        #######################################

        # feedforward - loop through all layers    
        for layer in self.hiddenAndOutputLayer:
            
            # calculate inner activation without bias
            # numpy.dot of all neuron weights of the current layer and the output of the parent layer
            innerActivation = numpy.dot(layer.getAllWeightsOfNeurons(), last_out)

            # add the bias of each neuron to the innerActivation
            innerActivation += layer.getAllBiasOfNeurons() 

            # save new innerActivation values in layer
            layer.setLastInnerActivation(innerActivation)

            # calculate new output of the current layer
            # this is used as input for the next layer in the next iteration
            # or as output for the output layer when the loop is finished
            last_out = self.transferFuction(innerActivation) 
            
            # save new output values in layer
            layer.setLastOutput(last_out)

        return last_out


    def calculate(self, c_input):
        """
        calculate the output for some given input (feedforward activation)
        
        :param c_input: a list, containing the input data for input layer
        """


        ######################
        ### error checking ###
        ######################

        # check if c_input is set correctly
        if not ( (type(c_input) is list) and (len(c_input) == self.inputLayerLength)  ):
            sys.exit("Error: c_input has to be a list containing the input data of length " + str(self.inputLayerLength))
        
        # set the output of the net
        self.output = self.__feedforward(numpy.array(c_input))
        
        print("")
        print("#################################################################")
        print("## calculating...                                              ##")
        print("#################################################################")
        print("## Input was:                                                  ##")
        print( c_input ) 
        print("##                                                             ##")
        print("## Output is:                                                  ##")
        print( self.output )
        print("##                                                             ##")
        print("#################################################################")
        print("")


    def transferFuction(self, x):
        return numpy.tanh(x)
        

    def transferFunctionDeriv(self, x):
        return 1 - numpy.power(numpy.tanh(x), 2)


    def printNetWeights(self):

        print("######################################################")
        print("     neural net: " + self.label)
        print("######################################################")
        print("")
        print("###############")
        print("### weights ###")
        print("###############")

        print("")

        print("-> " + str(len(self.hiddenAndOutputLayer)) + " layers total (hidden + output layer, input is not counted)")
        print("-> layer " + str(len(self.hiddenAndOutputLayer)-1) + " is the output layer") 
        print("")
    
        for key, layer in enumerate(self.hiddenAndOutputLayer):

            print("--- Layer: " + str(key) 
                    + " --- (" + str(layer.getLength()) 
                    + " neurons total | each row represents the weights of one neuron)")
            print(layer.getAllWeightsOfNeurons())
            print("")
            

    def __checkForErrors(self, in_input, hiddenLayerList, outputLayerLength, out_expected):
        """ checks for errors before starting current init things """

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


