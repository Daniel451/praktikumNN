#!/usr/bin/env python3.4
#-*- coding: utf-8 -*-

import numpy as n
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([1,30,30,1]) # Erstelle neues Neurales Netzwerk mit 2 Eingangsneuronen, 6 Hidden-Neuronen und 1 Ausgangsneuronen.
# Möglich wäre auch: NeuralNetwork([2,6,7,3,1]) also mit mehren Hidden Layern!


s_in = []
s_teach = []

for t in range(-200,200,1):
    s_in.append([t/100])
    s_teach.append([n.sin(n.pi*t/100)])
    

s_in = n.array(s_in)
s_teach = n.array(s_teach)

nn.teach(s_in, s_teach ,0.02,50000)  # Trainiren: 


# Anzeige: 


t = n.arange(-2.0, 2.0, 0.05)

nnPlot = []
for _t in t:
    nnPlot.append(nn.guess(_t)[-1][-1])
    

l1, = plt.plot(t, n.sin(n.pi*t))
l2, = plt.plot(t, nnPlot)


plt.legend( (l1, l2), ('n.sin(t)', 'nn.guess(t)'), loc='upper right', shadow=True)
plt.xlabel('t')
plt.title('Vergleich')
plt.show()
