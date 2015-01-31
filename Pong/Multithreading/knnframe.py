#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
"""
Hier ist der Frame für die Anbindung von Spiel (Pong) zu KNN enthalten.
Hier geht es vor allem um die Anbindung von unserem NeuralNetwork.py.
"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

import logging
from NeuralNetwork import NeuralNetwork
import numpy
import os.path
from concol import ConCol

class Knnframe:
    def __init__(self, loadconfig, playerid):
        """
        Init vom KNN Frame
        Setzen von default Werten bzw. Erstellen des echten KNN.
        :param loadconfig: Pfad/Konfigurationsdatei eines schon bestehenden KNN (nicht Implementiert!)
        :type loadconfig: str
        :param playerid: Spieler Identifikationsnummer (wird hauptsächlich für Dateinamen gebraucht)
        :type playerid: int
        :return: none
        :rtype: void
        """

        # Für effizientes debugging und logging können Werte und Informationen in eine
        #  Datei geschrieben werden
        file = 'log_player_' + str(playerid) + '.log'
        self.createlogfile(file)
        logging.basicConfig(filename=file, level=logging.INFO)
        logging.info('==================')
        logging.info('====  Started  ===')
        logging.info('==================')

        # Spieler Identifikationsnummer (wird hauptsächlich für Dateinamen und Debuggingausgaben gebraucht)
        self.playerid = playerid

        # Um eine Bewertung des Spieles wärend der Spielzeit durchführen zu können, werden die Treffer zu den
        #  nicht-Treffern (Outs) gezählt und ins Verhältnis gebracht. Dies geschieht über einen gleitenden Durchschnitt
        self.timesteps = 20.0 # die letzten X Belohnungen sollen Zählen (Formel hierzu: siehe unten in reward_pos bzw reward_neg)
        self.hitratio = 0.5 # der initiale Wert der Treffer = 1 zu Outs = 0.

        # um eine sinnvolle Ausgabe zu erzeugen, werden die Belohnungen gezählt.
        self.reward_count = 0

        # Debugausgaben nur alle X Belohnungen:
        self.printcount = 10

        # Erstellen des neuralen Netzwerk Objektes.
        # Konfiguration ist wie folgt zu verstehen:
        #  NeuralNetwork(layer, tmax):
        #  bei NeuralNetwork([2,5,1],8) ist dies also:
        #   Aufbau:
        #   - 2 input Neuronen
        #   - 5 hidden Neuronen
        #   - 1 output Neuron
        #   Lernschritte in die Vergangenheit:
        #   - 8 Speicherstellen für das lernen von Situationen vor einer Bestätigung
        #   (siehe hierzu die NeuralNetwork.py)
        #
        self.knn = NeuralNetwork([2, 5, 1], 8)

    @staticmethod
    def createlogfile(logfilename):
        """
        Legt ein logfile an, wenn Keines existiert.
        :param logfilename: Dateiname
        :type logfilename: str
        :return: none
        :rtype: void
        """
        if not os.path.exists(logfilename):
            file = open(logfilename, "w+")
            file.close()

    def saveconfig(self,filename):
        """
        Stellt eine Funktion zur Verfügung, die das KNN Anweist die Konfiguration zu speichern
        :param filename: Dateiname
        :type filename: str
        :return: none
        :rtype: void
        """
        self.knn.save(filename)
        
    def predict(self,xpos,ypos,mypos):
        """
        Stellt eine Funktion zur Verfügung, die das KNN Anweist aus gegebenen Werten, hier ein normierter
        Ortsvektor, eine Ausgabe bzw. Aktion vorherzusagen.
        :param xpos: normierte x-Komponente des Ortsvektors
        :type xpos: float
        :param ypos: normierte y-Komponente des Ortsvektors
        :type ypos: float
        :param mypos: Position des eigenen Schlägers, normiert (wird hier nicht weiter genutzt)
        :type mypos: float
        :return: Aktion die das KNN vorhersagt (kann float oder str sein)
        :rtype: float
        """

        #Rufe die passende Funktion im KNN auf, hierbei ist zu beachten, das es, wie in NeuralNetwork.py beschrieben
        # praktisch immer ein 2-Dimensionales (Numpy-) Array ist. Daher die [[ ]] doppelte Klammerung: ".shape=1,2"
        # Ebenfalls muss diese eingabe zur Konfiguration des KNN passen. Hier ist es mit 2 Input-Neuronen erstellt,
        # somit müssen auch 2 Werte eingegeben werden.
        pred = self.knn.predict([[xpos,ypos]])

        # Auch das zurückgegebene Ergebnis ist 2-Dimensional, enthält aber nur einen Wert wie
        #  in der Konfiguration angegeben.
        return float(pred[0][0])


    def reward_pos(self,err):
        """
        Stellt eine Funktion zur Verfügung, die aufgerufen werden soll, wenn das KNN eine positive Belohnung
        erhalten soll. Neben dem Aufrufen des KNNs wird auch dafür gesorgt, dass der gleitende Durchschnitt,
        siehe __init__() aktualisiert wird.
        :param err: skalierter Deltawert zwischen Schlägermittelpunkt und Auftreffpunkt des Balles
        :type err: float
        :return: none
        :rtype: void
        """

        # rufe die Belohnungsfunktion (hier: supervised lernen) vom KNN auf
        self.knn.reward(err)

        # Aktualisieren vom gleitenden Durchschnitt nach oben hin.
        self.hitratio += 1.0/self.timesteps
        if self.hitratio > 1.0: # Bergenzen auf 1.0
            self.hitratio = 1.0

        # Debug ausgaben in die Konsole, um zu sehen, ob das Spiel im Gange ist und wie der aktuelle Zustand ist.
        #  Die eigentlichen Informationen sind jedoch in der dedizierten Visualisierung zu sehen.
        print( ConCol.OKGREEN + 'Player ' + self.playerid + ': got positive reward! Hitratio is now: ' + str(self.hitratio) + ConCol.ENDC )

    def reward_neg(self,err):
        """
        Stellt eine Funktion zur Verfügung, die aufgerufen werden soll, wenn das KNN eine negative Belohnung
        erhalten soll. Neben dem Aufrufen des KNNs wird auch dafür gesorgt, dass der gleitende Durchschnitt,
        siehe __init__() aktualisiert wird.
        :param err: skalierter Deltawert zwischen Schlägermittelpunkt und Auftreffpunkt des Balles
        :type err: float
        :return: none
        :rtype: void
        """
        # rufe die Belohnungsfunktion (hier: supervised lernen) vom KNN auf
        self.knn.reward(err)

        # Aktualisieren vom gleitenden Durchschnitt nach unten hin.
        self.hitratio -= 1.0/self.timesteps
        if self.hitratio < 0.0: # Bergenzen auf 0.0
            self.hitratio = 0.0

        # Debug ausgaben in die Konsole, um zu sehen, ob das Spiel im Gange ist und wie der aktuelle Zustand ist.
        #  Die eigentlichen Informationen sind jedoch in der dedizierten Visualisierung zu sehen.
        print( ConCol.OKBLUE + 'Player ' + self.playerid + ': got negative reward! Hitratio is now: ' + str(self.hitratio) + ConCol.ENDC)





