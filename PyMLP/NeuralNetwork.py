#!/usr/bin/env python3.4

import numpy as n

n.set_printoptions(precision=10)

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
        self.B = []
        for i in range(1, len(layer)):
            self.W.append(n.random.random((layer[i],layer[i - 1]))) #erzeuge layer[i - 1] Gewichte für jedes layer für jedes Neuron
            self.B.append(n.random.random((layer[i]))) #erzeuge 1 Bias für jedes Neuron
            
        #print('W ist: ' + str(self.W))
        #print('B ist: ' + str(self.B))

    def guess(self, s_in):
        """Feedforward Activation of the MLP.

        Keyword arguments:
        s_in -- input value, should be a list of values, quantity like the input layer!
        """
        
        a = n.atleast_2d(s_in)
        #print('a ist: ' + str(a))
        
        for l in range(0, len(self.W)): #für alle Layer...
            #print('self.W[l] ist ' + str(self.W[l]))
            a = _tanh(n.dot(a, n.transpose(self.W[l]))) + self.B[l] #multipliziere die Arraygewichte...
            # Achtung: a ist ein [[ x y z]] und self.B[l] ist ein [] array! Funktioniert trotzdem!
            #print('a ist: ' + str(a))
            #print('self.B[l] ist: ' + str(self.B[l]))
              
            #print('a ist: ' + str(a))
        return a


    def teach(self, s_in, s_teach, epsilon=0.2, repeats=10000):
        """Learning function for the MLP.
        
        Keyword arguments:
        s_in -- input value, should be a array of arrays of values. Quantity like the input layer! Order!
        s_teach -- result value, should be a array of arrays of values. Quantity like the output layer! Order!
        epsilon -- Factor for learning rate (default 0.2)
        epochs -- number of repeated learning steps (default 1000)
        """
        
        s_teach = n.atleast_2d(s_teach)      # -> faster!
        
        
        for k in range(repeats):
            i = n.random.randint(s_in.shape[0])
            a = n.atleast_2d(s_in)
            R = n.array([])
            
            #print('a ist: ' + str(a))
            for l in range(0, len(self.W)): #für alle Layer...
                #print('self.W[l] ist ' + str(self.W[l]))
                a = _tanh(n.dot(a, n.transpose(self.W[l]))) + self.B[l] #      HIER müssen irgendwie die Daten gesichert werden! TODO
            #print(R)
            #print('a: ' + str(a[-1]))
            #print('s_teach[i]: ' + str(s_teach[i]))
            
            #delta_out = s_teach[i] - a[-1]  #calculate error on output layer
            
            
            delta_lambda = (s_teach[i] - a[-1]) * _tanh_deriv(a[-1]) #calculate error on all hidden layers
            
            #print('delta_out ist: ' + str(delta_out))
            #print('delta_lambda ist: ' + str(delta_lambda))
            
            #for j in range(len(self.W)):
                #self.W[j] = self.W[j] + epsilon * delta_lambda
            
            
            
            
        #print('Gewichte nach lernen:')
        #print(self.W)



