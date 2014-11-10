

import numpy as np



class NeuralNet:

   def __init__(self, hiddenNeuronCount=5):

      # Prnting
      print("\n### Starte Programm ###\n")

      # Weights
      self.W = []
      self.W.append(
            [np.random.uniform(0.1, 1.9, 2)
               for i in range(0,hiddenNeuronCount)
            ])
      self.W.append([np.random.uniform(0.1, 1.9, hiddenNeuronCount)])

      # Bias
      self.B = []
      self.B.append(np.random.uniform(0, 0.2, hiddenNeuronCount))
      self.B.append(np.random.uniform(0, 0.2, 1))

   
   def _tanh_deriv(self, x):
      return 1 - np.power(np.tanh(x),2)


   def _tanh(self, x):
      return np.tanh(x)
      

   def teach(self, inputList, teachcount=10000, epsilon=0.1):
   
      

      for i in range(0, teachcount):

         selectRandomInput = inputList[np.random.randint(0,len(inputList))]

         locInArray = np.array(selectRandomInput[0])

         goalOutput = selectRandomInput[1]

         # compute and calculate error
         self.compute(selectRandomInput[0])
         locInArray = self.Output
         self.errorOut = goalOutput - self.Output

         # new weights and bias for hidden layer
         self.errorsHidden = self._tanh_deriv(self.innerActivation) * np.array(
                  [ self.errorOut * it_weight
                     for it_weight in self.W[1] ][-1]
               )

         self.W[0] += np.array([  epsilon * errHid * locInArray
                     for errHid in self.errorsHidden
                  ])
         self.B[0] += epsilon * self.errorsHidden

         # new weights and bias for output layer
         self.W[1] += epsilon * self.errorOut * self.innerOut
         self.B[1] += epsilon * self.errorOut

      
      print(self.errorOut)


   def compute(self, inputList):
      
      if(len(inputList) != 2):
         print("Die inputList muss exakt Laenge 2 haben")
         return

      inputArray = np.array(inputList)
      self.inputArray = inputArray

      self.innerActivation = np.array(
            [ np.dot(item, inputList) 
            for item in self.W[0] ] 
         )

      innerBias = self.innerActivation + self.B[0] 

      self.innerOut = np.array(
            [ self._tanh(item)
            for item in innerBias ]
            )
      
      OutputActivation = np.array(
            [ np.dot(item, self.innerOut)
            for item in self.W[1] ]
            )
      
      OutputBias = OutputActivation + self.B[1] 

      Output = np.array(
            [ self._tanh(item)
            for item in OutputBias ]
            )

      self.Output = Output


   def printOutput(self):
      print("Der Output ist: "
            + str(
               [ round(item,2) for item in self.Output ]
               )
            + " bei dem Input " + str(self.inputArray))



   def printWeights(self):
      
      print("### Weights ###\n")
      
      layercount = 0
      for sublist in self.W:
        
         print("Layer: " + str(layercount))
         
         neuroncount = 0
         for array in sublist:
            print("-> " + str(array) + " (Neuron: " + str(neuroncount) + ")")
            neuroncount += 1
         layercount += 1
      
      print("\n")


   def printBias(self):
      print("### Bias ###")

      for item in self.B:
         print(item)

      print("\n")


nn = NeuralNet(20)

nn.printWeights()
nn.printBias()

inputs = [
           [[0,0], 0],
           [[0,1], 1],
           [[1,0], 1],
           [[1,1], 0]
         ]

nn.teach(inputs, 100000, 0.1)

nn.printWeights()
nn.printBias()

nn.compute([0, 0])
nn.printOutput()
nn.compute([0, 1])
nn.printOutput()
nn.compute([1, 0])
nn.printOutput()
nn.compute([1, 1])
nn.printOutput()
