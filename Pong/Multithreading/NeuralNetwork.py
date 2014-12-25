#!/usr/bin/env python3.4
#-*- coding: utf-8 -*-

import numpy as n

def _tanh(x):                 # diese Funktion stellt die Übertragungsfunktion der Neuronen dar. Forwardpropagation 
    return n.tanh(x)

def _tanh_deriv(x):           # diese Funktion stellt die Ableitung der Übertragungsfunktion dar. Sie ist für die Backpropagation nötig.
    return 1.0 - n.power(n.tanh(x),2)

# Welche ist nun Richtig?
#def _tanh_deriv(x):  
#    return 1.0 - x**2


class NeuralNetwork:
    def __init__(self, layer, tmax):  
        """  
        :param layer: A list containing the number of units in each layer.
        Should be at least two values  
        """ 
        
        self.tmax = tmax #del lst[-1]
        
        self.W = [] #Erstelle das Array der Gewichte zwischen den Neuronen. 
        self.B = [] #Erstelle das Array der Biase für alle Neuronen.
        self.RWI = [] #Erstelle das Array der Gewichte zu den Recurrenten Daten intern, dh. zu sich selbst.
        self.RWE = [] #Erstelle das Array der Gewichte zu den Recurrenten Daten extern, dh. von output nach Input.
        self.RD = [] #Daten halten für Recurrente Daten.
        for i in range(1, len(layer)):
            self.W.append(n.random.random((layer[i],layer[i - 1]))-0.5) #erzeuge layer[i - 1] Gewichte für jedes layer für jedes Neuron zufällig im Bereich von -0.5 bis 0.5.
            #self.RWI.append(n.random.random((layer[i],layer[i]))-0.5)
            self.B.append(n.random.random((layer[i],1))-0.5) #ebenso zufällige Werte für Bias. Bereich: -0.5 bis 0.5.
            
        for t in range(0,self.tmax):  #für t Zeitschritte...
            RD = []
            RWI = []
            for i in range(1, len(layer)):
                RD.append( n.zeros((1,layer[i])) )
                RWI.append(n.random.random((1,layer[i]))-0.5)
            self.RD.append(RD)
            self.RWI.append(RWI)
                
        #TODO:
        #self.RWE.append(n.random.random((layer[-1],layer[1]))-0.5) #Erstelle die Rückkoplung von den Daten vom Ausgang zum Eingang. Sie sind dann Paralell zu den Eingangsneuronen
        
        
                
        
        print('')
        print('============================================================================')
        print('')

    def guess(self, s_in):
        """Feedforward Activation of the MLP.

        Keyword arguments:
        s_in -- input value, should be a list of values, quantity like the input layer!
        """
        
        
        a = n.atleast_2d(s_in) # Das Eingabe-Array, der input für den Input-Layer sollten als 2D Array aufgebaut sein: [[ 0, 0.7, 1 ]] Somit ist sichergestellt, dass im ganzen Programm, einfache Transpositionen möglich sind. Alle Gewichte/Biase/Output sehen "immer" so aus.
        # +====================================+
        # +********* Feedforward-Algo *********+
        # +====================================+
        
        RDtemp = []
        
        for l in range(0, len(self.W)): #für alle Layer... 
        #(Die Neuronen werden in einem Layer immer Gleichzeitig bearbeitet! Hier wir ausgenutzt, dass alle Informationen in den Arrays synchron und gleich groß sind. Es gibt (muss geben!) also zu jedem Neuron in einem Layer ein Subarray mit den Gewichten. Ebenso gibt es für jedes Neuron ein Bias und später vieleicht auch noch weitere Daten...)
            
            
            
            
            recurrentData = n.atleast_2d(n.zeros(len(self.B[l])))
            
            for r in range(0, self.tmax):
                recurrentData += self.RD[r][l].dot(self.RWI[r][l].T)   # multipliziere die Recurenten Daten mit den entsprechenden Gewichten und addiere sie dann miteinander...
            
            
            int_a=n.atleast_2d(a.dot(self.W[l].T)) + self.B[l].T + recurrentData
            # Summe der Produkte (von jeweiligen Gewichten und Output der Neuronen) und des Biases des Neurons wird... 
            #TODO: Recurente Daten (zumindest die lokalen ) wurden addiert, jetzt fehlt noch das Speichern und die Exterenen Daten.
            
            if len(self.W) - 1 == l: #... wenn es sich um ein Output Neuron handelt (LastLayer!)...
                a = int_a            # direkt ausgegeben (Linear!)
            else:
                a = _tanh(int_a)     # an sonsten wird diese mit der Übertragungsfunktion "angepasst" und dann an das nachste Layer weitergegeben.
            
            #print('Aktuelle Daten vs. Recurente Daten: ' , self.RD[0][l] , a )
            RDtemp.append(a)
            
        
        self.RD.insert(0,RDtemp)
        del self.RD[-1] # letzen Datensatz löschen, da dann self.tmax+1 lang
        return a #Es wird hier ein Array mit Count(Ausgangsneuronen) zurückgegeben. Die Output Werte 
    
    def teach(self, s_in, s_teach, epsilon=0.2, repeats=10000):
        """Learning function for the MLP.
        
        Keyword arguments:
        s_in -- input value, should be a array of arrays of values. Quantity like the input layer! Order!
        s_teach -- result value, should be a array of arrays of values. Quantity like the output layer! Order!
        epsilon -- Factor for learning rate (default 0.2)
        epochs -- number of repeated learning steps (default 1000)
        """
        
        s_teach = n.atleast_2d(s_teach)      # Wie Oben beschriben, sollten alle Werte 2D-Arrays sein.
        
        for k in range(repeats):
            i = n.random.randint(s_in.shape[0]) #Wähle zufällig einen Lern-Datensatz aus
            a = n.atleast_2d(s_in[i])           # der gegebenen Menge der Beispieldaten aus.
            
            Activation = []           # init des Activation Zwischenspeichers: Summe(Gewichte, Output vom vorherigen Layer) + Bias
            Activation.append(a)      # Das erste Layer braucht nicht berechnet zu werden, es sind gleichzeitig die Activation wie auch Input Daten des NN.
            Output = []               # init des Activation Zwischenspeichers: Übertragungsfunktion( Activation )
            Output.append(a)          # Die Inputlayer haben keine Übertragungsfunktion sie sind nur "dumme Werteträger", Input Neuronen eben!
            
            # +====================================+
            # +********* Feedforward-Algo *********+
            # +====================================+
            for l in range(0, len(self.W)): #für alle Layer... 
                #Wie in guess(self, s_in), werden auch hier identisch (!!) die Activation und Output Daten mittels Feedforward-Algo berechnet. Der Unterschied ist jedoch, dass wir uns hier nun die Daten für den folgenden Backpropagation-Algo merken müssen!
                int_a=n.atleast_2d(a.dot(self.W[l].T))+self.B[l].T
                
                Activation.append(int_a) #merken des Activation-Wertes!
                if len(self.W) - 1 == l: # siehe oben...
                    a = int_a
                else:
                    a = _tanh(int_a)
                Output.append(a) #merken des Output-Wertes!
                
            
            # +========================================+
            # +********* Backpropagation-Algo *********+
            # +========================================+
            
            # Das NN ist im untrainierten Zustand (Zufallszahlen in den Gewichten) sehr warscheinlich falsch, es gibt einen Error-Wert: delta:
            delta = n.atleast_2d((s_teach[i] - a[-1]))    #Erzeugt delta zum ersten Mal: SollAusgabe - IstAusgabe mittels der Trainingsdaten. 
            
            for l in range(len(self.W)-1, 0, -1): #für alle Layer, diesmal jedoch von hinten nach von!
                if l > 0: #solange wir uns über dem Input-Layer befinden, berechnen wir schon mal den delta-Wert für die nächste Iteration, den Delta-Wert dieses Layers wird jedoch noch für das Anpassen der Gewichte benötigt 
                    #TODO was muss hier hin? Activation oder Output? -> müsste Activation sein....
                    delta_next = _tanh_deriv(Activation[l]) * (self.W[l] * n.transpose(delta)).sum(axis=0)
                
                #TODO Evtl. könnte aus der _tanh_deriv Funktion die tanh(x) Funktion entfernt werden, wenn ich unten nun statt Output[] Activation[] nutzen würde. Richtig? Falsch?!
                
                #TODO was muss hier hin? Activation oder Output? -> müsste Output sein....
                self.W[l] = self.W[l] + ((epsilon * n.transpose(delta)) * Output[l]) #Anpassen der Gewichte
                self.B[l] = self.B[l] + epsilon * n.transpose(delta) # Anpassen des Bias
                
                delta = delta_next # wie oben schon kurz angesprochen, hier wird nun delta_next zu delta, damit der passende Wert für das nächste Layer zur Verfügung steht.
                # Da delta von der Gewichtsanpassung und die Gewichte für das delta_next gebraucht wird, muss es so auseinander gezogen werden.
                
                
    def save(self, file): #untested: sichere Gewichte und Bias
        n.savez(file + '.npz', W=self.W,B=self.B)
        
    def load(self, file): #untested: lade Gewichte und Bias
        data = n.load(file + '.npz')
        self.W = data['W']
        self.B = data['B']
        
    def debug(self): #untested: lade Gewichte und Bias
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
        
    
