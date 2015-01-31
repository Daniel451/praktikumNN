#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
"""In dieser Datei befindet sich das MLP in etwa wie wir es im Praktikum zu Neuralen Netzen kennengelernt haben.
Es unterscheidet sich dahin, dass wir die Prediction- (Feedforward) und Learn- (Backpropagation) Funktion
aufgeteilt haben. Dies hat für uns den Vorteil, dass immer alle Werte für eine evtl. Backpropagation zur
Verfügung stehen. Weiterhin wird für jede Hidden-Schicht eine Rekurenz erstellt. Näheres dazu in der Implementation
"""

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

def _tanh(x):
    """
    Übertragungsfunktion für die Feedforward-Berechnung
    :param x: Parameter x in f(x) = tanh(x)
    :type x: numpy
    :return: Funktionswert f aus f(x) = tanh(x)
    :rtype: numpy
    """
    return n.tanh(x)


def _tanh_deriv(x):
    """
    Übertragungsfunktion bei der Backpropagation
    :param x: Parameter x in f(x) = 1 - tanh(x)^2
    :type x: numpy
    :return: Funktionswert f aus f(x) = 1 - tanh(x)^2
    :rtype: numpy
    """
    return 1.0 - n.power(n.tanh(x), 2)



class NeuralNetwork:
    def __init__(self, layer, tmax):
        """
        Diese Funktion initialisiert das KNN. Sie setzt Standard- bzw. Zufallswerte und baut die Datenstruktur für die
        dynamische Berechnung auf.

        :param layer: Der Aufbau des KNN wird hier übergeben. Diese Struktur ist folgendermaßen zu Verstehen: Das KNN
        wird vom Input-Layer aus beschrieben mit jeweils einer Zahl größer Null die die Anzahl der Neuronen im
        entsprechnenden Layer angibt. Hierbei wird das Input- und s-Layer mit einbezogen. Eine folgende
        Konfiguration [5,8,10,4] kann also verstanden als ein KNN mit 5 Input Neuronen,
        8 Hiddenneuronen in der 1. Schicht, 10 Hiddenneuronen in der 2. Schicht und 4 Ausgabeneuronen.
        Die Hiddenschichten haben jeweils in eine rekurente Schicht die sie dazu befähigt "in die Vergangenheit zu
        sehen." len(layer) > 1.
        :type layer: list

        :param tmax: Hier wird definiert, wie viele Zeitschritte bzw Iterationen sich das KNN merken soll um beim Lernen
        ebenfalls vergangenen Zeiten einzubeziehen bei denen nicht klar war, ob der jeweilige Zustand gut oder
        schlecht war. tmax > 0. (Im Beispiel von Pong: Wärend der Ball auf dem Spielfeld noch unterwegs ist)
        :type tmax: int

        :return: none
        :rtype: void
        """

        #Sichern der Daten für spätere Funktionen (z.B. dem Lernen)
        self.tmax = tmax

        # Initialisiere das Array für die Gewichte zwischen den Neuronen
        self.W = []
        # Initialisiere das Array der Biase für alle Schichten
        self.B = []
        # Initialisiere das Array für die rekurrenten Gewichte
        self.RW = []
        # Initialisiere den Speicher für die vergangenen Vorhersagen. Gleichzeitig ist hier auch der Speicher der
        # rekurenten Daten enthalten.
        self.RS = []

        # Anlegen der Struktur für die Gewichte (W) und Bias (B)
        for i in range(1, len(layer)):
            # Erzeuge zwischen jedem Layer eine Gewichtematrix mit m*n zufälligen (-0.5 bis +0.5) Gewichten.
            # m ist die Anzahl der Neuronen in der unteren Schicht und
            # n ist die Anzahl der Neuronen in der oberen Schicht.
            # Somit ergibt sich eine Liste aus Matrizen die wie folgt aussehen könnte:
            # angenommene Konfiguration: [3,5,7,2] -> list( (3x5), (5x7), (7x2) )
            self.W.append((n.random.random((layer[i-1], layer[i])) - 0.5))
            # Erzeuge zwischen jedem Layer eine Gewichtematrix mit 1*n zufälligen (-0.5 bis +0.5) Gewichten.
            # n ist die Anzahl der Neuronen in der oberen Schicht.
            # Somit ergibt sich eine Liste aus Matrizen die wie folgt aussehen könnte:
            # angenommene Konfiguration: [3,5,7,2] -> list( (1x5), (1x7), (1x2) )
            self.B.append((n.random.random((1,layer[i])) - 0.5))

        # Anlegen der Struktur für die rekurrenten Gewichte (RW).
        # Input- und s-Layer erhalten keine Rekurenz!
        for i in range(1, len(layer)-1):
            # Erzeuge zwischen jedem Layer eine Gewichtematrix mit m*m zufälligen (-0.05 bis +0.05) Gewichten.
            # m ist die Anzahl der Neuronen in der oberen Schicht.
            # Somit ergibt sich eine Liste aus Matrizen die wie folgt aussehen könnte:
            # angenommene Konfiguration: [3,5,7,2] -> list( (5x5), (7x7) )
            self.RW.append((n.random.random((layer[i], layer[i])) - 0.5)*0.1)

        # Anlegen von leeren (Nullen) Daten für die Vergangenheit. Diese werden wie in einem Ringpuffer gespeichert
        #  und später bei jedem Predict erwgänzt. Sobald der Puffer voll ist, wird das älteste Element daraus
        # entfernt bzw. überschrieben.

        # Template erzeugen
        __temp_RS = []
        for i in range(0, len(layer)):
            # Erzeugt für jedes Layer eine 1*i Matrix mit Nullen
            # i ist die Anzahl der Neuronen in der aktuellen Schicht.
            # Somit ergibt sich eine Liste aus Matrizen die wie folgt aussehen könnte:
            # angenommene Konfiguration: [3,5,7,2] -> list( (1x3), (1x5), (1x7) (1x2) )
            __temp_RS.append(n.zeros((1, layer[i])))

        # Ringpuffer füllen falls frühzeitig (t < tmax) die Lernfunktion aufgerufen wird.
        for t in range(0, self.tmax+1):
            # für alle Zeitschritte von 0 bis tmax wird jeweils ein leers Template hinzugefügt.
            self.RS.append(__temp_RS)

        # Speicher für die Activation der Neuronen initialisieren
        self.h = []
        # Speicher für den s der Neuronen initialisieren
        self.s = []


    def predict(self, s_in):
        """
        Die Vorhesagefunktion soll aus einem Input einen passenden Output erzeugen. Dieses wird durch die angepassten,
        dazu mehr in der reward()-Funktion, Gewichte erreicht.
        :param s_in: Input für das KNN. Diese müssen zu der Struktur der Inputneuronen stimmen. z.B.: [[ 1 , 0.3 ]] bei
        zwei Inputneuronen.
        :type s_in: numpy
        :return: Die Ausgabe des KNN entspricht immer dem Ausgabelayer. z.B.: [[ 1 , 0.3 , 0.8 ]] bei 3 Outputneuronen.
        :rtype: numpy
        """

        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+

        # Input Daten wandeln zu einem numpy Array, wenn sie es nicht schon sind.
        s = n.atleast_2d(s_in)



        # Initialisierung des h Zwischenspeichers
        self.h = []
        # Das erste Layer braucht nicht berechnet zu werden, es kann direkt als Activation übernommen werden.
        self.h.append(s)

        # Initialisierung des s Zwischenspeichers
        self.s = []
        # Das erste Layer braucht nicht berechnet zu werden, es kann direkt als Output übernommen werden, da die
        # lineare Funktion genutzt wird.
        self.s.append(s)


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

            self.h.append(h)  #merken des h-Wertes!
            self.s.append(s)  #merken des s-Wertes!


        self.RS.insert(0, self.s)
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

            delta_next = _tanh_deriv(self.h[l]) * delta.dot(self.W[l].T)



            self.B[l] += epsilon * delta
            self.W[l] += epsilon * delta * self.s[l].T


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

                #print('<s>')
                #pprint.pprint(self.s[l])
                #print('</s>')

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
            'O':self.s,
            'A':self.h
        }
        return dataset
