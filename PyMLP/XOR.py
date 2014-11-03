#!/usr/bin/env python3.4

import numpy as n

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2,6,1]) # Erstelle neues Neurales Netzwerk mit 2 Eingangsneuronen, 6 Hidden-Neuronen und 1 Ausgangsneuronen.
# Möglich wäre auch: NeuralNetwork([2,6,7,3,1]) also mit mehren Hidden Layern!

s_in = n.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #Trainingsdaten Input
s_teach = n.array([[0], [1], [1], [0]])          #Trainingsdaten Output


print( str(s_in.shape)) 
print( str(s_teach.shape) )

print( str(type(s_in))) 
print( str(type(s_teach)) )


print( str(s_in[0].shape)) 
print( str(s_teach[0].shape) )

print( str(type(s_in[0]))) 
print( str(type(s_teach[0])) )

nn.teach(s_in, s_teach ,0.3,25000)  # Trainiren: 
#s_in: Input Daten als numpy-Array
#s_teach: Output Daten als numpy-Array
# optional: epsilon=0.2: Lernfaktor
# optional: repeats=10000: Wiederholungen


#for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
#    print(i,nn.guess(i))

# whoaaaa: sichern von Daten zum laden fürs nächste Mal ;)
#nn.save('savetest') # erzeugt eine 'savetest.npz' Datei! (alles unchecked!, überschreiben ohne Warnung!)
    
#nn.load('savetest') # läd eine 'savetest.npz' Datei! (alles unchecked!)
