#!/usr/bin/env python3.4

import numpy as n

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2,6,1])
s_in = n.array([[0, 0], [0, 1], [1, 0], [1, 1]])
s_teach = n.array([[0], [1], [1], [0]])
#s_teach = n.array([[0,0], [1,1], [1,1], [0,0]])

#s_in = n.array([[0, 1]])
#s_teach = n.array([ [1]])

nn.teach(s_in, s_teach ,0.2,100000)


#test:



for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.guess(i))
    
