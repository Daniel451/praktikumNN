#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

"""
Hier ist der Frame für die Anbindung von Spiel (Pong) zu KNN enthalten.
Hier geht es vor allem um die Anbindung von unserem recneunet.py.
"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

from recneunet import NeuralNetwork
from concol import ConCol

import logging
import os.path


class Knnframe:

    def __init__(self, loadconfig, playerid):
        """
        Initialisierung von Knnframe.
        Setzen von default-Werten bzw. Erstellen des "echten" KNN.

        :param loadconfig: Pfad/Konfigurationsdatei eines schon bestehenden KNN (noch nicht Implementiert!)
        :type loadconfig: str

        :param playerid: Spieleridentifikationsnummer (wird hauptsächlich für Dateinamen gebraucht)
        :type playerid: int

        :return: none
        :rtype: void
        """

        # Für effizientes debugging und logging können Werte und Informationen in eine
        # Datei geschrieben werden
        file = 'log_player_' + str(playerid) + '.log'  # Dateipfad
        self.createlogfile(file)
        logging.basicConfig(filename=file, level=logging.INFO)
        logging.info('==================')
        logging.info('====  Started  ===')
        logging.info('==================')

        # Spieleridentifikationsnummer (wird hauptsächlich für Dateinamen und Debuggingausgaben benötigt)
        self.playerid = playerid

        # Um eine Bewertung des Spieles während der Spielzeit durchführen zu können, werden die Treffer zu den
        # Nicht-Treffern (Outs) gezählt und in ein Verhältnis gebracht.
        # Dies geschieht über einen gleitenden Durchschnitt.

        # Anzahl der letzten x Belohnungen, die in die Berechnung der Hitratio eingehen sollen.
        # (Formel hierzu: siehe unten in reward_pos() bzw. reward_neg())
        self.timesteps = 100.0

        # Das initiale Verhältnis von Treffern zu Outs (100% Treffer = 1.0, 100% Outs = 0.0).
        self.hitratio = 0.5

        # Zu Ausgabe-/Debug-/Logzwecken werden die Belohnungen gezählt.
        self.reward_count = 0

        ########################################################
        # Erstellen des künstlichen neuronalen Netzwerkobjekts #
        ########################################################
        #
        # Die Konfiguration ist wie folgt zu verstehen:
        #
        # NeuralNetwork(layer, tmax)
        #
        # layer: Liste -> [Input-Neuronen, Hidden-Neuronen*, Output-Neuronen]
        # *: Es sind beliebig viele Hidden-Layer möglich. Die Konfiguration [2,10,5,1] würde
        # z.B. ein Netz mit 2 Hidden-Layern erstellen,
        # das Erste Hidden-Layer mit 10, das Zweite Hidden-Layer mit 5 Neuronen.
        #
        # tmax: Integer
        #
        # Beispiel:
        #
        #   bei NeuralNetwork([2,5,1],8) würde dies folgendem Aufbau entsprechen:
        #
        #   - 2 input Neuronen
        #   - 5 hidden Neuronen
        #   - 1 output Neuron
        #
        #   Lernschritte in die Vergangenheit:
        #
        #   - 8 Speicherstellen für das zeitverzögerte Lernen von Situationen vor einer Belohnung
        #   (siehe hierzu die recneunet.py)
        #
        self.knn = NeuralNetwork([2, 20, 1], 20)

    @staticmethod
    def createlogfile(logfilename):
        """
        Legt ein logfile an, wenn keines existiert.

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
        
    def predict(self, xpos, ypos, mypos):
        """
        Stellt eine Funktion zur Verfügung, die das KNN Anweist aus gegebenen Werten, hier ein normierter
        Ortsvektor, eine Ausgabe bzw. Aktion vorherzusagen.

        :param xpos: Normierte x-Komponente des Ortsvektors
        :type xpos: float

        :param ypos: Normierte y-Komponente des Ortsvektors
        :type ypos: float

        :param mypos: Position des eigenen Schlägers, normiert (wird hier nicht weiter genutzt)
        :type mypos: float

        :return: Aktion, die das KNN berechnet (kann float oder str sein)
        :rtype: float
        """

        # Rufe die passende Funktion im KNN auf, hierbei ist zu beachten, das es, wie in recneunet.py beschrieben
        # praktisch immer ein 2D (Numpy-)Array ist. Daher die doppelte Klammerung [[ ]]: ".shape=1,2"
        #
        # Ebenfalls muss diese Eingabe zur Konfiguration des KNN passen. Hier ist es mit 2 Input-Neuronen erstellt,
        # daher müssen auch 2 Werte eingegeben werden.
        pred = self.knn.predict([[xpos, ypos]])

        # Die zurückgegebene Prädiktion ist ebenfalls zweidimensional, enthält aber nur einen Wert wie
        # in der Konfiguration angegeben.
        return float(pred[0][0])


    def reward_pos(self, err):
        """
        Stellt eine Funktion zur Verfügung, die aufgerufen werden soll, wenn das KNN eine positive Belohnung
        erhält. Außerdem wird der gleitende Durchschnitt (siehe __init__()) aktualisiert.

        :param err: skalierter Deltawert zwischen Schlägermittelpunkt und Auftreffpunkt des Balles
        :type err: float

        :return: none
        :rtype: void
        """

        # Rufe die Belohnungsfunktion (hier: supervised learning) vom KNN auf
        self.knn.reward(err)

        # Anzahl der bisherigen Rewards erhöhen/aktualisieren
        self.reward_count += 1

        # Aktualisieren vom gleitenden Durchschnitt nach oben hin.
        self.hitratio += 1.0 / self.timesteps

        # Da die hitratio ebenfalls normiert ist (Interval: [0.0, 1.0]),
        # muss korrigiert werden, falls die obere Schranke überschritten wurde.
        if self.hitratio > 1.0:
            self.hitratio = 1.0

        # Debug-Ausgaben in die Konsole schreiben, um zu sehen, ob das Spiel im Gange ist
        # sowie den aktuellen Zustand beschreiben.
        # Die eigentlichen Informationen sind jedoch in der dedizierten Visualisierung zu sehen.
        print(ConCol.OKGREEN + 'Player ' + str(self.playerid) + ': got positive reward! Hitratio is now: '
               + str(self.hitratio) + " (" + str(self.reward_count) + " rewards total)" + ConCol.ENDC)

    def reward_neg(self, err):
        """
        Stellt eine Funktion zur Verfügung, die aufgerufen werden soll, wenn das KNN eine negative Belohnung
        erhält. Außerdem wird der gleitende Durchschnitt (siehe __init__()) aktualisiert.

        :param err: skalierter Deltawert zwischen Schlägermittelpunkt und Auftreffpunkt des Balles
        :type err: float

        :return: none
        :rtype: void
        """

        # Rufe die Belohnungsfunktion (hier: supervised learning) vom KNN auf
        self.knn.reward(err)

        # Anzahl der bisherigen Rewards erhöhen/aktualisieren
        self.reward_count += 1

        # Aktualisieren vom gleitenden Durchschnitt nach unten hin.
        self.hitratio -= 1.0/self.timesteps

        # Da die hitratio ebenfalls normiert ist (Interval: [0.0, 1.0]),
        # muss korrigiert werden, falls die untere Schranke unterschritten wurde.
        if self.hitratio < 0.0:
            self.hitratio = 0.0

        # Debug-Ausgaben in die Konsole schreiben, um zu sehen, ob das Spiel im Gange ist
        # sowie den aktuellen Zustand beschreiben.
        # Die eigentlichen Informationen sind jedoch in der dedizierten Visualisierung zu sehen.
        print(ConCol.OKBLUE + 'Player ' + str(self.playerid) + ': got negative reward! Hitratio is now: '
              + str(self.hitratio) + " (" + str(self.reward_count) + " rewards total)" + ConCol.ENDC)

    def v_gethitratio(self):
        """
        Stellt für die Visualisation Daten zur Vefügungung.

        :return: Treffer zu nicht Treffer
        :rtype: float
        """

        return self.hitratio

    def v_getrewcount(self):
        """
        Stellt für die Visualisation Daten zur Vefügungung.

        :return: Anzahl der Belohnungen
        :rtype: int
        """

        return self.reward_count