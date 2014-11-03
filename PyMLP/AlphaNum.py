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





class world_digits:
"""reads the 8x8 image files in digits_alph that display digits and capital letters"""

   def __init__(self):
      self.digitsalph = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      self.values = numpy.zeros((self.len(digitsalph), 8*8))
      for tt in range(len(digitsalph)):
         filename = "digits_alph/digit" + self.digitsalph[tt] + ".pgm"
         self.values[tt], h, w = KT.importimage (filename)
         if  h != 8 or w != 8:
            print "digits_alph files don't match expected size!"
      self.datumsize = h*w
      self.seqlen = len(self.digitsalph)

   def dim(self):
      return (self.datumsize, self.seqlen)

   def newinit(self):
      self.t = 0

   def act(self):
      # world reaction
      self.t += 1

   def sensor(self):
      # returns a list
      # [0] -> one image as an 8x8 long vector
      # [1] -> alphanumerical thing
      return [self.values[self.t], self.digitsalph[self.t]]
