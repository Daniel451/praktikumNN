#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
"""Das Spielfeld für Pong wird simuliert.

Ein anpassbares Spielfeld für Pong mit einer normalerweise 16/9 Größe hat einen Ball, und zwei schläger jeweils
rechts und links am Spielfeldrand. Die Punkte werden für beide Spieler gezählt. Spieler sind immer 0 und 1. Spiler 0
ist der linke Spieler. Zum einfachen Adaptieren an ein folgendes System ist die Schnittstelle mit normierten
Ein- und Ausgaben versehen. Diese Normierung bezieht sich immer von -1 bis +1 """

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

import numpy as np 
import random

class Court:
    """
    Objekt das das Spielfeld darstellt.
    Enthält Funktionen zur Manipulaton von Schlägern und Inspektoren für die Daten:
    - Skaliert für die KNNs
    - Unskaliert für die Visualisierung
    """

    def __init__(self):
        """Initialisiert das Objekt Court. Hierzu zählen Spielfeld, Spieler und Anfangs-
            ballposition.
        :return void
        """
        # Größe des Spielfeldes (16 zu 9 hat sich als gut bestätigt)
        self.x_max = 16.0
        self.y_max = 9.0

        # Ballgeschwindigkeit
        self.speed = 0.5

        # Rauschen auf die Ballposition hinzufügen (Faktor)
        self.outputNoiseMax = 0.0 # Achtung: Noch nie mit Rauschen getestet! Sollte bei 0 bleiben!

        # Soll der Ball aus dem Spielfeld fliegen können oder ewig hin und her springen?
        self.infinite = False

        # Größe des Schlägers von Spieler 0 und 1. (von der Mitte zu einem Ende, d.h hier die halbe
        #   länge eintragen!
        self.batsize = 1.0

        # Im Befehlsmodus kann der Schläger mit den Befehlen 'u' und 'd' bewegt werden. Hier wird die
        #   Sprungweite des Schlägers angegeben.
        self.batstep = 0.3
        
        #### ^^^ Parameter zum ändern ^^^ ####
        
        self.posVec = None # Ortsvektor des Balles (Bezugspunkt ist [0,0] )
        self.dirVec = None # Richtungsvektor des Balles (Einheitsvektor)

        self._bathit = [False, False] # Binärer Speicher, ob der Ball den einen Schläger getroffen hat
        self._out = [False, False] # Binärer Speicher, ob der Ball die Linie übrflogen hat
        self.Points = [0, 0]  # Punktestand
        self.poi = [None, None] # Der "Einschlagpunkt" des Balles auf der Linie, wird erst nach einem Aufprall
                                #  gefüllt. Genutzt um den Error zu berechnen.
        self.bat = [self.y_max/2.0 , self.y_max/2.0] # Schlägerpositionen der Spieler auf ihren Linien.

        self.bouncecount = 0 # Zählt die Schlägertreffer um nach 10 Treffern das Spielfeld zu resetten. 
                             #  ( Da bei gelernten KNNs nicht nur die eine Position gelernt werden soll.)


        self.__initvectors() # Initialisiere das erste Mal den Ortsvektor und Richtungsvektor. (Startvorbereitung)


    def __initvectors(self):
        """Initialisiert Anfangs- und Richtungsballvektoren.
        Irgendwo in der Mitte auf der Y-Achse und mit einem belibigen Startwinkel. Dieser Startwinkel ist maximal von
          -45 Grad bis 45 Grad von der Horizontale au gesehen.
        :return void
        """

        ## Richtungsvektor erzeugen

        # Zufallswinkel im Bogenmaß generieren
        rotationAngle = np.random.uniform(-np.pi / 4, np.pi / 4)

        # aus Zufallswinkel eine Rotationsmatrix generieren
        rotMatrix = np.array([
            [np.cos(rotationAngle), -np.sin(rotationAngle)],
            [np.sin(rotationAngle), np.cos(rotationAngle)]
        ])

        #Rotationsmatrix auf einen Einheitsvektor in horizontaler Richtung anwenden
        self.dirVec = np.dot(rotMatrix, np.array([1, 0]))

        # Zufällig entscheiden, ob der Ball nach links (zu Player 0) oder rechts (zu Player 1) startet.
        if random.random() > 0.5:
            self.dirVec[0] *= -1.0

        ## Ortsvektor erzeugen

        #Start irgendowo auf der Mittellinie
        self.posVec = np.array([self.x_max / 2.0, self.y_max * random.random()])

        # Rücksetzen der Anzahl der Ping-Pongs
        self.bouncecount = 0


    def _incrpoints(self,player):
        """Erhöht den Punktestand für einen Spieler[Player]
        :param player: Int vom Spieler (0 oder 1)
        :return void
        """
        self.Points[player] += 1


    def __sensor_x(self):
        """Gibt die X-Achse mit Rauschen zurück
        :return float, X-Anteil vom Ortsvektor
        """
        return self.posVec[0] + (random.random() - 0.5 ) * self.outputNoiseMax


    def __sensor_y(self):
        """Gibt die Y-Achse mit Rauschen zurück
        :return float, Y-Anteil vom Ortsvektor
        """
        return self.posVec[1] + (random.random() - 0.5 ) * self.outputNoiseMax


    def __sensor_bat(self, player):
        """Gibt die Position des Schlägers von Spieler[Player] mit Rauschen zurück
        :param player: Int vom Spieler (0 oder 1)
        :return float, Schlägerposition von Spieler[Player]
        """
        return self.bat[player] + (random.random() - 0.5 ) * self.outputNoiseMax
        

    def scaled_sensor_x(self):
        """Gibt die X-Achse skaliert von -1 bis +1 mit Rauschen zurück
        :return float, skalierter X-Anteil vom Ortsvektor
        """
        return self.__sensor_x() / (self.x_max/2.0) - 1.0


    def scaled_sensor_y(self):
        """Gibt die Y-Achse skaliert von -1 bis +1 mit Rauschen zurück
        :return float, skalierter Y-Anteil vom Ortsvektor
        """
        return self.__sensor_y() / (self.y_max/2.0) - 1.0


    def scaled_sensor_bat(self, player):
        """Gibt die Position des Schlägers von Spieler[Player] skaliert von -1 bis +1
        mit Rauschen zurück
        :param player: Int vom Spieler (0 oder 1)
        :return float, skalierter Schlägerposition von Spieler[Player]
        """
        return self.__sensor_bat(player) / (self.y_max/2.0) - 1.0

    def scaled_sensor_err(self, player):
        """Gibt den Error von Spieler[Player] skaliert von -1 bis +1 zurück.
        :pre hitbat(player) or out(player)
        :param player: Int vom Spieler (0 oder 1)
        :return float, skalierter Error von Spieler[Player]
        """
        return (self.poi[player] - self.__sensor_bat(player) ) / self.y_max


    def hitbat(self, player):
        """Gibt an, ob der Schläger von Spieler[Player] getroffen wurde oder nicht.
        :param player: Int vom Spieler (0 oder 1)
        :return Bool, Treffer (True) oder kein Treffer (False) vom Schlager von Spieler[Player]
        """
        return self._bathit[player]

    def out(self, player):
        """Gibt an, ob der Ball die Linie von Spieler[Player] überschritten hat oder nicht.
        :param player: Int vom Spieler (0 oder 1)
        :return Bool, Ball hat die Linie überschritten (True) oder nicht überschritten (False) von Spieler[Player]
        """
        return self._out[player]

    def getpoints(self, player):
        """Liefert die Punktanzahl von Spieler[Player]
        :param player: Int vom Spieler (0 oder 1)
        :return int, Punktzahl des Spielers
        """
        return self.Points[player]


    def tick(self):
        """

        :return:
        """
        #########################
        ### Initialisierungen ###
        #########################

        # Setze den Ball auf eine Position weiter. Die Schrittweite wird durch mainloopdelay gesetzt.
        self.posVec += self.dirVec * self.speed

        # Hat der Schläger den Ball getroffen?
        # bathit[0] -> linker Schläger
        # bathit[1] -> rechter Schläger
        self._bathit = [False, False]
        self._out = [False, False]

        ###################
        ### Anweisungen ###
        ###################

        # Falls 10 oder mehr Treffer also jeder mindestens 5x getroffen hat, dann wird abgebrochen
        #  und neu gestartet, damit die aktuelle Endlosschleife unterbrochen wird. Hier würde das KNN
        #  nichts mehr lernen.
        if self.bouncecount > 10: #TODO: Abbruch und neu init sollte besser in der Mitte geschehen!
            self.__initvectors()

        # Abprallen auf der Unterseite bei Y = 0
        if self.posVec[1] < 0:
            self.posVec[1] = self.posVec[1] * -1.0
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # Abprallen auf der Oberseite bei Y = y_max (hier vermutlich 9)
        if self.posVec[1] > self.y_max:
            self.posVec[1] = 2 * self.y_max - self.posVec[1]
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # Prüfe auf Treffer auf der linken Seite (Spieler 0)
        self.__tickBounceLeft()
        
        # Prüfe auf Treffer auf der rechten Seite (Spieler 1)
        self.__tickBounceRight()


    def __tickBounceLeft(self):
        """Checken, ob der Ball links bei Spieler 0 aus dem Spielfeld fliegt oder vom Schläger getroffen wird
        :return: void
        """

        #Wenn der Ortsvektor kleiner ist als 0, dann hat er die Linie von Spieler 0 überschritten, dann...
        if self.posVec[0] < 0:

            #Berechne den theoretischen, genauen Aufprallpunkt (poi: PointOfImpact) auf der Linie von Spieler 0 (Y = 0)

            factor = (0 - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec + (factor * self.dirVec)

            self.poi[0] = poi[1] #Speichere diesen für eine evtl. spätere Nutzung von z.B. scaled_sensor_err(player)

            # Prüfe ob der Ball dann den Schläger getroffen hätte, wenn ja, dann...
            if (poi[1] > self.bat[0] - self.batsize) and (poi[1] < self.bat[0] + self.batsize):
                self._bathit[0] = True            # ... vermerke dies für z.B. hitbat(player)
            else:                                 # wenn jedoch nicht, dann...
                self.Points[1] +=1                # ... Punkte von Spieler 1 (rechts) erhöhen
                self._out[0] = True               # und merken, das der Ball außerhalb des Spielfelds
                                                  #   war, z.B. für out(player)

            # Ball abprallen lassen, falls:
            # -> Das infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[0]:
                self.posVec[0] *= -1.0 # Einfallswinklel = Ausfallswinkel
                self.dirVec[0] *= -1.0

                self.bouncecount += 1 # Treffer vermerken, um bei zu vielen Treffern dieses neu zu starten
            else:
                self.__initvectors() # Kein Treffer, somit das Spiel neu Initialisieren.
                self.bouncecount = 0



    def __tickBounceRight(self):
        """Checken, ob der Ball rechts bei Spieler 1 aus dem Spielfeld fliegt oder vom Schläger getroffen wird
        :return: void
        """
        # Wenn der Ortsvektor größer ist als x_max (hier vermutlich 16), dann hat er die Linie
        # von Spieler 1 überschritten, dann...
        if self.posVec[0] > self.x_max:

            # Berechne den theoretischen, genauen Aufprallpunkt (poi: PointOfImpact) auf der Linie von
            # Spieler (Y = self.x_max)
            factor = (self.x_max - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec + (factor * self.dirVec)

            self.poi[1] = poi[1] #Speichere diesen für eine evtl. spätere Nutzung von z.B. scaled_sensor_err(player)

            # Prüfe ob der Ball dann den Schläger getroffen hätte, wenn ja, dann...
            if poi[1] > self.bat[1] - self.batsize and poi[1] < self.bat[1] + self.batsize:
                self._bathit[1] = True            # ... vermerke dies für z.B. hitbat(player)
            else:                                 # wenn jedoch nicht, dann...
                self.Points[0] +=1                # ... Punkte von Spieler 0 (links) erhöhen
                self._out[1] = True               # und merken, das der Ball außerhalb des Spielfelds
                                                  #  war, z.B. für out(player)

            # Ball abprallen lassen, falls:
            # -> Das infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[1]:
                # 2 Spielfeldlängen - aktuellem X-Betrag ergibt neue X-Position
                self.posVec[0] = 2 * self.x_max - self.posVec[0] # Einfallswinklel = Ausfallswinkel
                self.dirVec[0] *= -1.0

                self.bouncecount += 1 # Treffer vermerken, um bei zu vielen Treffern dieses neu zu starten
            else:
                self.__initvectors() # Kein Treffer, somit das Spiel neu Initialisieren.
                self.bouncecount = 0


    def move(self, player, action):
        """Bewegt den Schläger eines Spielers
        Diese Funktion ist etwas Trickreich, da als "action"-Parameter sowohl ein String als direkter
        up/down-Befehl akzeptiert wird, als auch ein Float der den Schläger direkt setzt.
        :param player Int vom Spieler (0 oder 1), dessen Schläger bewegt werden soll
        :param action str "d" oder "u" (Schläger hoch oder runter bewegen)
        :param action float Schläger auf die entsprechende Position setzen
        :return: void
        """
        # Wenn ein String, dann im Befehls-Mode:
        if type(action) == str:

            # Schläger nach oben bewegen
            if action == 'u' :
                self.bat[player] += self.batstep
                if self.bat[player] > self.y_max: # Korrektur, falls oberer Spielfeldrand erreicht wurde
                    self.bat[player] = self.y_max

            # Schläger nach unten bewegen
            if action == 'd':
                self.bat[player] -= self.batstep
                if self.bat[player] < 0.0: # Korrektur, falls unterer Spielfeldrand erreicht wurde
                    self.bat[player] = 0.0
        # Sonst im Setzen-Mode:
        elif type(action) == float:
            self.bat[player] = action # Der Schläger wird direkt auf die gewünschte Position gesetzt.
            if self.bat[player] < 0.0: # Korrektur, falls unterer Spielfeldrand erreicht wurde
                self.bat[player] = 0.0
            if self.bat[player] > self.y_max: # Korrektur, falls oberer Spielfeldrand erreicht wurde
                self.bat[player] = self.y_max



    def v_getSize(self):
        """
        visu-getter
        :return float Liste [Float: X,Float: Y] der Spielfeldgröße
        """
        return [self.x_max,self.y_max]


    def v_getSpeed(self):
        """
        visu-getter
        :return float Ballgeschwindigkeit
        """
        return self.speed


    def v_getBatSize(self):
        """
        visu-getter
        :return float Schlägerlänge (Größe)
        """
        return self.batsize


    def v_getDirVec(self):
        """
        visu-getter
        :return float Bewegungsvektor
        """
        return self.dirVec


    def v_getPosVec(self):
        """
        visu-getter
        :return float Positionsvektor Liste [Float: X,Float: Y]
        """
        return self.posVec


    def v_getbat(self):
        """
        visu-getter
        :return: Liste [batSpieler0, batSpieler1] -> Position des Schlägermittelpunktes von Spieler 0 / 1
        """
        return self.bat


    def v_getPoint(self):
        """
        visu-getter
        :return: Liste [X,Y] des Punktestundes für Spieler 0 / 1
        """
        return self.Points

