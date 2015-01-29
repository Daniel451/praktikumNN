#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
""" bla bla bla"""   #TODO: Write summary about this file!

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

import pprint


import numpy as n
import time


def _tanh(x):  # diese Funktion stellt die Übertragungsfunktion der Neuronen dar. Forwardpropagation
    return n.tanh(x)


def _tanh_deriv(
        x):  # diese Funktion stellt die Ableitung der Übertragungsfunktion dar. Sie ist für die Backpropagation nötig.
    return 1.0 - n.power(n.tanh(x), 2)



class NeuralNetwork:
    def __init__(self, layer, tmax):
        """  
        :param layer: A list containing the number of units in each layer.
        Should be at least two values  
        """

        self.tmax = tmax  #del lst[-1]

        self.W = []  # Erstelle das Array der Gewichte zwischen den Neuronen.
        self.B = []  # Erstelle das Array der Biase für alle Neuronen.
        self.RW = []  # Erstelle das Array der Gewichte zu den Recurrenten Daten intern, dh. zu sich selbst.
        self.RS = []  # Daten halten für Recurrente Daten.
        for i in range(1, len(layer)):
            self.W.append((n.random.random((layer[i-1], layer[i])) - 0.5))  # erzeuge layer[i - 1] Gewichte für jedes layer für jedes Neuron zufällig im Bereich von -0.5 bis 0.5.
            self.B.append((n.random.random((1,layer[i])) - 0.5))  # ebenso zufällige Werte für Bias. Bereich: -0.5 bis 0.5.

        for i in range(1, len(layer)-1):
            #Erzeuge für jedes Layer eine Matrix mit n mal n Gewichten. Gewichte für die Rekurenz!
            self.RW.append((n.random.random((layer[i], layer[i])) - 0.5)*0.1)  # erzeuge layer[i - 1] Gewichte für jedes layer für jedes Neuron zufällig im Bereich von -0.5 bis 0.5.

        __temp_RD = []
        for i in range(0, len(layer)):
            __temp_RD.append(n.zeros((1, layer[i])))
        for t in range(0, self.tmax+1):  # für t Zeitschritte leere "templates" zum Speichern der Daten
            self.RS.append(__temp_RD)


        self.Activation = [] #wird zu h
        self.Output = [] # wird zu s



    def predict(self, s_in):
        """Learning function for the MLP.

        Keyword arguments:
        s_in -- input value, should be a array of arrays of values. Quantity like the input layer! Order!
        s_teach -- result value, should be a array of arrays of values. Quantity like the output layer! Order!
        epsilon -- Factor for learning rate (default 0.2)
        epochs -- number of repeated learning steps (default 1000)

        Feedforward Activation of the MLP.

        Keyword arguments:
        s_in -- input value, should be a list of values, quantity like the input layer!
        """

        s = n.atleast_2d(s_in)  # Das Eingabe-Array, der input für den Input-Layer sollten als 2D Array aufgebaut sein: [[ 0, 0.7, 1 ]] Somit ist sichergestellt, dass im ganzen Programm, einfache Transpositionen möglich sind. Alle Gewichte/Biase/Output sehen "immer" so aus.
        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+



        #ist  H
        self.Activation = []  # init des Activation Zwischenspeichers:
        self.Activation.append(s)  # Das erste Layer braucht nicht berechnet zu werden, es sind gleichzeitig die Activation wie auch Input Daten des NN.

        #ist S
        self.Output = []  # init des Activation Zwischenspeichers: Übertragungsfunktion( Activation )
        self.Output.append(s)  # Die Inputlayer haben eine lineare Übertragungsfunktion sie sind daher "dumme Werteträger", Input Neuronen eben!


        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+

        for l in range(0, len(self.W)):  #für alle hidden Layer...

            #print('Trying to Calculate Layer: ' + str(l))
            #print('s:')
            #pprint.pprint(s)

            #print('self.W[l]:')
            #pprint.pprint(self.W[l])

            #print('s.dot(self.W[l]):')
            #pprint.pprint(s.dot(self.W[l]))

            #print('self.B[l]:')
            #pprint.pprint(self.B[l])


            if l < len(self.W)-1 : #es ist nicht das input noch das output layer!
                #print('self.RS[0][l+1]:')
                #pprint.pprint(self.RS[0][l+1])

                #print('self.RW[l]:')
                #pprint.pprint(self.RW[l])

                #print('self.RS[0][l+1].dot(self.RW[l]:')
                #pprint.pprint(self.RS[0][l+1].dot(self.RW[l]))

                h = s.dot(self.W[l]) + self.RS[0][l+1].dot(self.RW[l]) + self.B[l]
                #h = s.dot(self.W[l])  + self.B[l]
            else:

                h = s.dot(self.W[l]) + self.B[l]



            # Last layer is linear!
            if len(self.W) - 1 == l:  # siehe oben...
                s = h
            else:
                s = _tanh(h)

            self.Activation.append(h)  #merken des Activation-Wertes!
            self.Output.append(s)  #merken des Output-Wertes!


        self.RS.insert(0, self.Output)
        del self.RS[-1]  # letzen Datensatz löschen, da dann 2*self.tmax+1 lang

        return s

        #seems to be ok...



    def reward(self, diff, epsilon = 0.2):


        # +========================================+
        # +********* Backpropagation-Algo *********+
        # +========================================+
        #print('learning with last '+ str(self.tmax) +' Datasets ...')

        # Das NN ist im untrainierten Zustand (Zufallszahlen in den Gewichten) sehr warscheinlich falsch, es gibt einen Error-Wert: delta:
        delta = n.atleast_2d(diff)  #Erzeugt delta zum ersten Mal: SollAusgabe - IstAusgabe mittels der Trainingsdaten.


        starttime = time.time()

        for l in range(len(self.W) -1, -1, -1):  #für alle Layer, diesmal jedoch von hinten nach von!
            #print('Layer: ' + str(l) )
            #Wenn ich es richtig sehe, sind, für den delta_next die gewichte uninterressant!

            delta_next = _tanh_deriv(self.Activation[l]) * delta.dot(self.W[l].T)


            #print('<delta>')
            #pprint.pprint(delta)
            #print('</delta>')

            #print('<Output>')
            #pprint.pprint(self.Output[l])
            #print('</Output>')

            #print('<self.W[l]>')
            #pprint.pprint(self.W[l])
            #print('</self.W[l]>')

            self.B[l] += epsilon * delta
            self.W[l] += epsilon * delta * self.Output[l].T


            if l < len(self.W)-1: #erstes (input) und letztes (output) Layer haben keine Rekursion!
            # daher muessen auch keine Gewichte angepasst werden!
               self.RW[l] += epsilon * delta * self.RS[0][l+1].T

            delta = delta_next
                #print('---------------------------------------')
                # wie oben schon kurz angesprochen, hier wird nun delta_next zu delta, damit der passende Wert für das nächste Layer zur Verfügung steht.
                # Da delta von der Gewichtsanpassung und die Gewichte für das delta_next gebraucht wird, muss es so auseinander gezogen werden.
        # erster Datensatz gelernt, nun um einen Zeitschritt in die Vergangenheit gehen: t = t_(x-i)

        epsilon = 0.02


        for i in range(1, self.tmax):
            #delta = n.atleast_2d(poi-self.RS[i][-1])
            delta = n.atleast_2d(diff) # todo: vieleicht wäre eine ordentlich angepasster diff besser!

            ## RS[i] = [array([[-0.91446288, -0.47661583]]), array([[ 0.24259971,  0.16527083,  0.17044024]]), array([[-0.27864748]])]
            ###  [[-0.27864748]]
            for l in range(len(self.W) -1, -1, -1):  #für alle Layer, diesmal jedoch von hinten nach von!
                #print('Layer: ' + str(l) )
                #Wenn ich es richtig sehe, sind, für den delta_next die gewichte uninterressant!

                delta_next = _tanh_deriv(self.RS[i][l]) * delta.dot(self.W[l].T)


                #print('<delta>')
                #pprint.pprint(delta)
                #print('</delta>')

                #print('<Output>')
                #pprint.pprint(self.Output[l])
                #print('</Output>')

                #print('<self.W[l]>')
                #pprint.pprint(self.W[l])
                #print('</self.W[l]>')

                self.B[l] += epsilon * delta
                self.W[l] += epsilon * delta * self.RS[i][l].T


                if l < len(self.W)-1: #erstes (input) und letztes (output) Layer haben keine Rekursion!
                # daher muessen auch keine Gewichte angepasst werden!
                   self.RW[l] += epsilon * delta * self.RS[i+1][l+1].T

                delta = delta_next


        #print('==============================================')
        #print('...done in ' + str(time.time()-starttime) + 'sec')




    def save(self, file):  #untested: sichere Gewichte und Bias
        print(file)
        pprint.pprint(self.B)
        #n.savez(file, W=self.W) #, B=self.B)    #, R=self.RW)
        n.savez(file, W=self.W, RW=self.RW,B=[self.B])    #, R=self.RW)

    def load(self, file):  #untested: lade Gewichte und Bias
        #TODO Schön waere eine Ueberpruefung ob konsistent zu config!
        data = n.load(file + '.npz')
        self.W = data['W']
        self.B = data['B']
        self.RW = data['R']

    def debug(self):
        dataset = {
            'W':self.W,
            'B':self.B,
            'RD':self.RS,
            'RW':self.RW,
            'O':self.Output,
            'A':self.Activation
        }
        return dataset
