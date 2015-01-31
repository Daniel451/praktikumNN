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
        self.knn.save(filename) #TODO correct this!
        
    def predict(self,xpos,ypos,mypos):
        #return action  (up:      u
                      #  down:    d
                      #  nothing: n

        pred = self.knn.predict([[xpos,ypos]])
        #print( bcolors.FAIL + 'Player ' + self.playerid + ' predicted: ' + str(pred[0][0]) + ' with sourcedata: ' + str([xpos,ypos]) + bcolors.ENDC)
        #logging.debug('predicting...')
        #logging.debug(self.knn.debug())


        return float(pred[0][0])

        self.fakediff = 0.0 #TODO seems not to work like this... damn!
        diff = mypos - pred[0][0]



        #print( bcolors.HEADER + 'Player ' + self.playerid + ' diff: ' + str(diff) + bcolors.ENDC)

        if diff > 0.1:
            #print('Player ' + self.playerid +': up!')
            return 'd'
        elif diff < -0.1:
            #print('Player ' + self.playerid +': down!')
            return 'u'
        #print('hold position!')
        return 'n'

    def reward_pos(self,err):
        self.rew_diag()
        self.knn.reward(err)
        #self.knn.reward(self.fakediff)
        #print('\a') #Bell
        # Verhaeltnis von Treffern vom Schläger zu Out's: 0..1
        self.hitratio += 1.0/self.timesteps
        if self.hitratio > 1.0:
            self.hitratio = 1.0

        if self.reward_count % self.printcount == 0:
            print(ConCol.OKGREEN + "Player " + str(self.playerid) + ": got positive reward! Hitratio is now: "
                  + str(self.hitratio) + ConCol.ENDC)
        self.newfakediff()

    def reward_neg(self,err):
        self.rew_diag()
        if self.reward_count % self.printcount == 0:
            print('Player ' + str(self.playerid) + ': error is: ' + str(err))
        self.knn.reward(err)
        # Verhaeltnis von Treffern vom Schläger zu Out's: 0..1
        self.hitratio -= 1.0/self.timesteps
        if self.hitratio < 0.0:
            self.hitratio = 0.0

        if self.reward_count % self.printcount == 0:
            print( ConCol.OKBLUE + 'Player ' + str(self.playerid) + ': got negative reward! Hitratio is now: ' + str(self.hitratio) + ConCol.ENDC)
        self.newfakediff()

    def newfakediff(self):
        self.fakediff = numpy.random.normal(0.0,1.0/3.0)*(1.0-self.hitratio)
        # Gauss Normalverteilung von etwa -1 - +1 bei  self.hitratio = 0
        #print('Player ' + self.playerid + ': fakediff is now: ' + str(self.fakediff))

    def rew_diag(self):
        self.reward_count += 1
        print('Rewards: ', self.reward_count)
        print('Hitratio: ', self.hitratio)
        if self.reward_count == 20:
            #print('20')
            self.file.write('hitratio@20: ' + str(self.hitratio) + '\n')
        elif self.reward_count == 50:
            #print('50')
            self.file.write('hitratio@50: ' + str(self.hitratio) + '\n')
        elif self.reward_count == 75:
            #print('75')
            self.file.write('hitratio@75: ' + str(self.hitratio) + '\n')
        elif self.reward_count == 100:
            #print('150')
            self.file.write('hitratio@100: ' + str(self.hitratio) + '\n')
        elif self.reward_count == 150:
            #print('150')
            self.file.write('hitratio@150: ' + str(self.hitratio) + '\n')


