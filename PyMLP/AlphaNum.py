#!/usr/bin/env python3.4

import numpy as n
import numpy 
import glob
import KTimage as KT

class world_digits:
    """reads the 8x8 image files in digits_alph that display digits and capital letters"""

    def __init__(self):
        self.t = 0
        self.digitsalph = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.values = numpy.zeros((len(self.digitsalph), 8*8))
        for tt in range(len(self.digitsalph)):
            filename = "./digits_alph/digit" + self.digitsalph[tt] + ".pgm"
            self.values[tt], h, w = KT.importimage(filename)
            if  h != 8 or w != 8:
                print("digits_alph files don't match expected size!")
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
        
    def getall(self):
        return [self.values, list(self.digitsalph)]
        




s_teach =[]


picture = world_digits()

data = picture.getall()

for char in data[1]:
    code = ord(char) - 48
    binary = list(("{0:b}".format(code)).zfill(6))
    binary = [float(i) for i in binary]
    print(char, code, str(binary))
    s_teach.append(binary)


s_in = n.array(data[0])
s_teach = n.array(s_teach)


print( str(s_in.shape)) 
print( str(s_teach.shape) )

print( str(type(s_in))) 
print( str(type(s_teach)) )


from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([64,100,6]) 

print(type(s_in[1]))

nn.teach(n.array(s_in), n.array(s_teach) ,0.3,25000)  # Trainiren: 

#nn.teach(s_in, s_teach ,0.3,25000)  # Trainiren: 
for i in [3, 7, 9, 14, 25]:
    print(data[1][i],nn.guess(s_in[i]))

# whoaaaa: sichern von Daten zum laden fürs nächste Mal ;)
#nn.save('savetest') # erzeugt eine 'savetest.npz' Datei! (alles unchecked!, überschreiben ohne Warnung!)
    
#nn.load('savetest') # läd eine 'savetest.npz' Datei! (alles unchecked!)

print('fertig')



