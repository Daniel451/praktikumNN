#!/usr/bin/env python3.4

import numpy as n

import glob
print( str(glob.glob("/*.*")))

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2,6,1]) 

s_in = n.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #Trainingsdaten Input
s_teach = n.array([[0], [1], [1], [0]])          #Trainingsdaten Output

nn.teach(s_in, s_teach ,0.3,25000)  # Trainiren: 


for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.guess(i))

# whoaaaa: sichern von Daten zum laden fürs nächste Mal ;)
#nn.save('savetest') # erzeugt eine 'savetest.npz' Datei! (alles unchecked!, überschreiben ohne Warnung!)
    
#nn.load('savetest') # läd eine 'savetest.npz' Datei! (alles unchecked!)
