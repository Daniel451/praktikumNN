#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as n


def _tanh(x):  # diese Funktion stellt die Übertragungsfunktion der Neuronen dar. Forwardpropagation
    return n.tanh(x)


def _tanh_deriv(
        x):  # diese Funktion stellt die Ableitung der Übertragungsfunktion dar. Sie ist für die Backpropagation nötig.
    return 1.0 - n.power(n.tanh(x), 2)


# Welche ist nun Richtig?
#def _tanh_deriv(x):
#    return 1.0 - x**2


class NeuralNetwork:
    def __init__(self, layer, tmax):
        """  
        :param layer: A list containing the number of units in each layer.
        Should be at least two values  
        """

        self.tmax = tmax + 5  #del lst[-1]

        self.W = []  # Erstelle das Array der Gewichte zwischen den Neuronen.
        self.B = []  # Erstelle das Array der Biase für alle Neuronen.
        self.RWI = []  # Erstelle das Array der Gewichte zu den Recurrenten Daten intern, dh. zu sich selbst.
        self.RWE = []  # Erstelle das Array der Gewichte zu den Recurrenten Daten extern, dh. von output nach Input.
        self.RD = []  # Daten halten für Recurrente Daten.
        for i in range(1, len(layer)):
            self.W.append(n.random.random((layer[i], layer[
                i - 1])) - 0.5)  # erzeuge layer[i - 1] Gewichte für jedes layer für jedes Neuron zufällig im Bereich von -0.5 bis 0.5.
            # self.RWI.append(n.random.random((layer[i],layer[i]))-0.5)
            self.B.append(
                n.random.random((layer[i], 1)) - 0.5)  # ebenso zufällige Werte für Bias. Bereich: -0.5 bis 0.5.

        for t in range(0, self.tmax):  # für t Zeitschritte...
            RD = []
            RWI = []
            for i in range(1, len(layer)):
                RD.append(n.zeros((1, layer[i])))
                # RWI.append(n.random.random((1,layer[i]))-0.5)
                RWI.append(n.zeros((1, layer[i])))
            self.RD.append(RD)
            self.RWI.append(RWI)

        self.Activation = []
        self.Output = []
        self.RDtemp = []

        self.hitratio = 0.5
        self.impact = 100.0 # impact of moving average



        print('')
        print('============================================================================')
        print('')

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

        a = n.atleast_2d(
            s_in)  # Das Eingabe-Array, der input für den Input-Layer sollten als 2D Array aufgebaut sein: [[ 0, 0.7, 1 ]] Somit ist sichergestellt, dass im ganzen Programm, einfache Transpositionen möglich sind. Alle Gewichte/Biase/Output sehen "immer" so aus.
        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+




        self.Activation = []  # init des Activation Zwischenspeichers: Summe(Gewichte, Output vom vorherigen Layer) + Bias
        self.Activation.append(a)  # Das erste Layer braucht nicht berechnet zu werden, es sind gleichzeitig die Activation wie auch Input Daten des NN.
        self.Output = []  # init des Activation Zwischenspeichers: Übertragungsfunktion( Activation )
        self.Output.append(a)  # Die Inputlayer haben keine Übertragungsfunktion sie sind nur "dumme Werteträger", Input Neuronen eben!

        self.RDtemp = []

        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+
        for l in range(0, len(self.W)):  #für alle Layer...
            #Wie in guess(self, s_in), werden auch hier identisch (!!) die Activation und Output Daten mittels Feedforward-Algo berechnet. Der Unterschied ist jedoch, dass wir uns hier nun die Daten für den folgenden Backpropagation-Algo merken müssen!

            #todo Normieren der acticvation



            recurrentData = n.atleast_2d(n.zeros(len(self.B[l])))

            for r in range(0, self.tmax):
                #print(r, self.RD[r][l] , self.RWI[r][l], recurrentData)
                recurrentData += self.RD[r][l] * self.RWI[r][l]  # multipliziere die Recurenten Daten mit den entsprechenden Gewichten und addiere sie dann miteinander...

            int_a = (n.atleast_2d(a.dot(self.W[l].T)) + self.B[l].T + recurrentData) / len(self.B[l])

            self.Activation.append(int_a)  #merken des Activation-Wertes!

            if len(self.W) - 1 == l:  # siehe oben...
                a = int_a
            else:
                a = _tanh(int_a)

            self.Output.append(a)  #merken des Output-Wertes!
            self.RDtemp.append(a)

        self.RD.insert(0, self.RDtemp)
        del self.RD[-1]  # letzen Datensatz löschen, da dann self.tmax+1 lang
        return a

    def reward(self, diff, epsilon = 0.2):

        # +========================================+
        # +********* Backpropagation-Algo *********+
        # +========================================+
        print('Angeblich lerne ich nun...')
        # Das NN ist im untrainierten Zustand (Zufallszahlen in den Gewichten) sehr warscheinlich falsch, es gibt einen Error-Wert: delta:
        delta = n.atleast_2d(diff)  #Erzeugt delta zum ersten Mal: SollAusgabe - IstAusgabe mittels der Trainingsdaten.


        for l in range(len(self.W) - 1, -1, -1):  #für alle Layer, diesmal jedoch von hinten nach von!
            if l > 0:  #solange wir uns über dem Input-Layer befinden, berechnen wir schon mal den delta-Wert für die nächste Iteration, den Delta-Wert dieses Layers wird jedoch noch für das Anpassen der Gewichte benötigt
                # DontCareTODO was muss hier hin? Activation oder Output? -> müsste Activation sein....
                delta_next = _tanh_deriv(self.Activation[l]) * (self.W[l] * n.transpose(delta)).sum(axis=0)

            # DontCareTODO Evtl. könnte aus der _tanh_deriv Funktion die tanh(x) Funktion entfernt werden, wenn ich unten nun statt Output[] Activation[] nutzen würde. Richtig? Falsch?!

            # DontCareTODO was muss hier hin? Activation oder Output? -> müsste Output sein....


            self.W[l] += epsilon * delta.T * self.Output[l]  #Anpassen der Gewichte
            self.B[l] += epsilon * delta.T  # Anpassen des Bias

            for r in range(0, self.tmax): # Anpassen der rekurenten Gewichte
                self.RWI[r][l] += epsilon * delta * self.RD[r][l] * 0.001

            delta = delta_next

            # wie oben schon kurz angesprochen, hier wird nun delta_next zu delta, damit der passende Wert für das nächste Layer zur Verfügung steht.
            # Da delta von der Gewichtsanpassung und die Gewichte für das delta_next gebraucht wird, muss es so auseinander gezogen werden.

    def save(self, file):  #untested: sichere Gewichte und Bias
        n.savez(file + '.npz', W=self.W, B=self.B, R=self.RWI)

    def load(self, file):  #untested: lade Gewichte und Bias
        #TODO Schön waere eine Ueberpruefung ob konsistent zu config!
        data = n.load(file + '.npz')
        self.W = data['W']
        self.B = data['B']
        self.RWI = data['R']

    def debug(self):
        print('Gewichte:')
        print(self.W)
        print('=======================================')
        print()
        print('Recurrent Data:')
        print(self.RD)
        print('=======================================')
        print()
        print('Recurrent W (intern):')
        print(self.RWI)
        print('=======================================')
        print()
        print('Recurrent W (extern):')
        print(self.RWE)
        print('=======================================')
        print()
        print('Bias:')
        print(self.B)
        print('=======================================')
        
    
