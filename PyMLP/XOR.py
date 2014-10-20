#!/usr/bin/env python3.4

import numpy as n

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2,3,4,7,1])
s_in = n.array([[0, 0], [0, 1], [1, 0], [1, 1]])
s_teach = n.array([0, 1, 1, 0])
nn.teach(s_in, s_teach ,0.2,3)


for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.guess(i))