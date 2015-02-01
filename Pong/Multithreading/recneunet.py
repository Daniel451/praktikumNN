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

import numpy as n
import copy as c

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
        self.RH = []
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
        __temp_R = []
        for i in range(0, len(layer)):
            # Erzeugt für jedes Layer eine 1*i Matrix mit Nullen
            # i ist die Anzahl der Neuronen in der aktuellen Schicht.
            # Somit ergibt sich eine Liste aus Matrizen die wie folgt aussehen könnte:
            # angenommene Konfiguration: [3,5,7,2] -> list( (1x3), (1x5), (1x7) (1x2) )
            __temp_R.append(n.zeros((1, layer[i])))

        # Ringpuffer füllen falls frühzeitig (t < tmax) die Lernfunktion aufgerufen wird.
        for t in range(0, self.tmax+1):
            # für alle Zeitschritte von 0 bis tmax wird jeweils ein leers Template hinzugefügt.
            self.RH.append(c.deepcopy(__temp_R))
            self.RS.append(c.deepcopy(__temp_R))


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
        # lineare Funktion im Inputlayer genutzt wird.
        self.s.append(s)

        # Jeder Ebene der Gewichte durchgehen, also über die "Verbindngslayer" iterieren. Das Nullte befindet sich
        #  zwischen der Input und der ersten Hiddenschicht.
        for l in range(0, len(self.W)):

            # Wenn wir uns unter der Output schicht befinden, dann...
            if l < len(self.W)-1 :
                # Für alle Hiddenlayer sollen die rekurrenten Daten berücksichtigt werden...
                h = s.dot(self.W[l]) / len(self.W[l]) + self.RS[0][l+1].dot(self.RW[l]) / len(self.RW[l]) + self.B[l]
                # ... um dann mittels der Übertragungsfunktion den Output von den Neuronen zu errechnen.
                s = _tanh(h)
                # Mathematik: (Hier möge auf die im Kurs verwendeten Unterlagen verwiesen sein.)

            else: # ... sonst sind wir in der Outputschicht,
                  #   hier ist die Rekurenz nicht erwünscht, außerdem ...
                h = s.dot(self.W[l]) / len(self.W[l]) + self.B[l]
                 # ... soll eine lineare Output-Funktion genutzt werden.
                s = h

            # Nun werden die s und h - Werte für jedes Layer gesichert, und...
            self.h.append(h)
            self.s.append(s)

        # ... für zukünftige Lernschritte als ganzes "Abbild des KNN" im Ringpuffer gesichert.
        self.RH.insert(0, c.deepcopy(self.h)) # (Eintragen an 0-Stelle, dann...
        del self.RH[-1]  # ... den letzen Datensatz löschen. Der Puffer ist wieder genauso lang, wie zuvor, jedoch
                         # sind alle Abbilder um eine Stelle nach hinten verschoben worden.


        self.RS.insert(0, c.deepcopy(self.s)) # (Eintragen an 0-Stelle, dann...
        del self.RS[-1]  # ... den letzen Datensatz löschen. Der Puffer ist wieder genauso lang, wie zuvor, jedoch
                         # sind alle Abbilder um eine Stelle nach hinten verschoben worden.



        # Ausgeben der Outputs (s) der letzten Schicht.
        return s



    def reward(self, diff, epsilon = 0.2):
        """
        Die Lernfunktion, hier Reward, da jetzt bekannt ist, ob die Aktion gut oder schlecht war, kann, anhand
        eines Deltas, die Gewichte zu den Neuronen verbessern. Beim nächsten Aufruf von Predict() sollten dann
        schon eine bessere, genauere Vorhersage generiert werden können.
        :param diff: Differenz, Delta, zwischen dem Soll und dem Ist-Punkt. Diese müssen zu der Struktur der
        Outputneuronen stimmen. z.B.: [[ 1 , 0.3 ]] bei zwei Outputneuronen.
        :type diff: numpy
        :param epsilon: Lernfaktor
        :type epsilon: float
        :return: none
        :rtype: void
        """

        # +========================================+
        # +********* Backpropagation-Algo *********+
        # +========================================+

        # Delta Daten wandeln zu einem numpy Array, wenn sie es nicht schon sind.
        delta = n.atleast_2d(diff)

        # Jeder Ebene der Gewichte durchgehen, also über die "Verbindngslayer" iterieren. Das Nullte befindet sich
        #  zwischen der Input und der ersten Hiddenschicht. Hier jedoch vom Output Layer nach vorn zum Input Layer
        for l in range(len(self.W) -1, -1, -1):

            # Da im nächsten Schritt die Gewichte verändert werden, wir jedoch für das Berechnen des nächsten Deltas
            #  noch die originalen Gewichte benötigen, errechnen wir deshalb schon jetzt das Delta des nächsten
            #  Layers:
            delta_next = _tanh_deriv(self.h[l]) * delta.dot(self.W[l].T)
            # Mathematik: (Hier möge auf die im Kurs verwendeten Unterlagen verwiesen sein.)

            # Anpassen der Gewichte zu den nächsten Layern:
            self.B[l] += epsilon * delta
            self.W[l] += epsilon * delta * self.s[l].T

            # Anpassen der Gewichte zu den rekurrenten Daten, wobei erstes (Input) und letztes (Output) Layer haben
            # keine Rekursion ...
            if l < len(self.W)-1:
                # ... daher müssen auch keine Gewichte angepasst werden!
                self.RW[l] += epsilon * delta * self.RS[0][l+1].T

            # für das nächste Layer kann nun das Delta für gültig erklärt werden.
            delta = delta_next

        # Die Lernrate für die zurückligenden Ballpositionen muss recht klein sein, da sonst die Gewichte sich nicht
        #  auf passende Werte einstellen können.
        epsilon /= self.tmax # Die Positionierung der aktuellen Situation ist deutlich wichtiger, hängt
                                    #  aber auch von der Anzahl der zu lernenden Schritte ab.
        #epsilon = 0.0
        # Errechnen des Referenz-Deltas, das dazu genutzt wird, die vergangenen Situationen zu bewerten.
        poi = self.RS[0][-1] + n.atleast_2d(diff)
            #      S_t + delta_t = S_(t+1) + delta_(t+1)
            # <=>  S_t + delta_t - S_(t+1) = delta_(t+1)

        # Für die gewünschte Anzahl der zu lernenden Situationen in der Vergangenheit wird nun mit den passenden
        #  Daten der Backpropagation Algorithmus (BPA) ausgeführt.
        for i in range(1, self.tmax):
            # (ab 1, da 0 die aktuelle Situation war, diese wurde jedoch schon oben abgearbeitet)

            # Siehe "Errechnen des Referenz-Deltas" ca. Zeile 248. Aus der Referenz kann nun ein passendes Delta für
            #  diesen Datensatz mit seinem Outputwert gebildet werden.
            delta = poi - self.RS[i][-1]

            # Wie schon im oberen BPA beschrieben, wird der Fehler vom Output-Layer Richtung Input-Layer propagiert.
            for l in range(len(self.W) -1, -1, -1):

                # Da im nächsten Schritt die Gewichte verändert werden, wir jedoch für das Berechnen des nächsten Deltas
                #  noch die originalen Gewichte benötigen, errechnen wir deshalb schon jetzt das Delta des nächsten
                #  Layers:
                delta_next = _tanh_deriv(self.RH[i][l]) * delta.dot(self.W[l].T)

                # Anpassen der Gewichte zu den nächsten Layern:
                self.B[l] += epsilon * delta
                self.W[l] += epsilon * delta * self.RS[i][l].T

                # Anpassen der Gewichte zu den rekurrenten Daten, wobei erstes (Input) und letztes (Output) Layer haben
                # keine Rekursion ...
                if l < len(self.W)-1: #erstes (input) und letztes (output) Layer haben keine Rekursion!
                # ... daher müssen auch keine Gewichte angepasst werden!
                   self.RW[l] += epsilon * delta * self.RS[i+1][l+1].T

                # für das nächste Layer kann nun das Delta für gültig erklärt werden.
                delta = delta_next





    def save(self, file):
        """
        Speichert die Konfiguration des MLPs in eine Datei, sie kann über load(file) wieder eingelesen werden.
        :param file: Dateiname der Datei
        :type file: str
        :return: none
        :rtype: void
        """
        raise NotImplementedError() #(Note: Im GIT befindet sich eine halbfunktionierende Lösung...)

    def load(self, file):
        """
        Läd die Konfiguration des MLPs aus einer Datei, sie kann über save(file) gespoeichert werden.
        :param file: Dateiname der Datei
        :type file: str
        :return: none
        :rtype: void
        """
        raise NotImplementedError() #(Note: Im GIT befindet sich eine halbfunktionierende Lösung...)

    def debug(self):
        """
        Debug-Funktion die die Konfiguration als Liste ausgibt. Sie gibt einen schnellen überblick
        über die interne Struktur.
        :return: Debugdaten
        :rtype: dict
        """
        dataset = {
            'W':self.W,
            'B':self.B,
            'RD':self.RS,
            'RW':self.RW,
            's':self.s,
            'h':self.h
        }
        return dataset
