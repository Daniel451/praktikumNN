#!/usr/bin/env python3.4

import numpy as n

n.set_printoptions(precision=2)

def _tanh(x):  
    return n.tanh(x)

def _tanh_deriv(x):  
    return 1.0 - x**2


class NeuralNetwork:
    def __init__(self, layer):  
        """  
        :param layer: A list containing the number of units in each layer.
        Should be at least two values  
        """  
        
        self.W = []  
        for i in range(1, len(layer) - 1):  
            self.W.append((2*n.random.random((layer[i - 1] + 1, layer[i] ))-1)*0.25)
            self.W.append((2*n.random.random((layer[i]     + 1, layer[i + 1]    ))-1)*0.25)
            print((2*n.random.random((layer[i - 1] + 1, layer[i]     ))-1)*0.25)
            print((2*n.random.random((layer[i]     + 1, layer[i + 1]    ))-1)*0.25)
            #print((layer[i]     + 1, layer[i + 1]    )-1)
            
            print('i ist: ' + str(i))
            print('layer[i] ist : ' + str(layer[i])) 
            

        print(self.W)

    def guess(self, s_in):
        """Feedforward Activation of the MLP.

        Keyword arguments:
        s_in -- input value, should be a list of values, quantity like the input layer!
        """
        s_in = n.array(s_in)
        temp = n.ones(s_in.shape[0]+1)
        temp[0:-1] = s_in
        a = temp
        for l in range(0, len(self.W)):
            a = _tanh(n.dot(a, self.W[l]))
        return a


    def teach(self, s_in, s_teach, epsilon=0.2, epochs=10000):
        """Learning function for the MLP.
        
        Keyword arguments:
        s_in -- input value, should be a array of arrays of values. Quantity like the input layer! Order!
        s_teach -- result value, should be a array of arrays of values. Quantity like the output layer! Order!
        epsilon -- Factor for learning rate (default 0.2)
        epochs -- number of repeated learning steps (default 1000)
        """
        s_in = n.atleast_2d(s_in) #format the input data to good arrays
        s_teach = n.array(s_teach)      # -> faster!
        
        
        for k in range(epochs):
            i = n.random.randint(s_in.shape[0])
            a = [s_in[i]]
            
            print('w ist: ' + str(self.W)) 
            print('i ist: ' + str(i)) 
            print('a ist: ' + str(a)) 
            print('k ist: ' + str(k)) 
        #print('Gewichte nach lernen:')
        #print(self.W)



