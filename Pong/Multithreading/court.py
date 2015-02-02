#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

"""
Das Pong-Spielfeld wird simuliert.

Court moduliert ein anpassbares Spielfeld für Pong mit einem standardmäßigen Seitenverhältnis von 16:9.
Jenes Spielfeld verfügt über einen Ball und zwei Schläger, jeweils links und rechts am Spielfeldrand,
sowie einen Punktestand für beide Spieler (0 und 1).
Spieler 0 spielt auf der linken Hälfte, Spieler 1 auf der rechten Hälfte.
Zwecks einfacher Adaptierung an Folgesysteme ist die Schnittstelle mit normierten Ein- und Ausgabewerten versehen,
welches alle Daten auf ein Interval [-1.0, 1.0] normiert.
"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

import numpy as np
import random


class court:
    """
    Objekt, dass das Spielfeld darstellt.

    Enthält außerdem Funktionen zur Manipulation von Schlägern und Inspektoren für die Daten:
        - Skalierte Daten für die KNNs
        - Unskalierte Daten für die Visualisierung
    """


    def __init__(self):
        """
        Initialisiert ein court-Objekt.
        Hierzu zählen Spielfeld, Spieler sowie die Startposition des Balles.

        :return void
        """

        ##############################
        ### veränderbare Parameter ###
        ##############################

        # Größe des Spielfeldes (standardmäßig 16 zu 9; hat bei Tests bewährt)
        self.x_max = 16.0
        self.y_max = 9.0

        # Ballgeschwindigkeit
        # (Faktor für den Richtungs-/Bewegungsvektor / die Ballgeschwindigkeit;
        # NeuerOrtsvektor = AlterOrtsvektor + Richtungs-/Bewegungsvektor * Ballgeschwindigkeitsfaktor)
        self.speed = 0.5

        # Rauschen auf die Ballposition hinzufügen (Faktor)
        self.outputNoiseMax = 0.0  # Achtung: Noch nie mit Rauschen getestet! Sollte bei 0 bleiben!

        # Soll der Ball aus dem Spielfeld fliegen können oder ewig hin und her springen?
        # True  -> Ball fliegt ewig hin und her, wird bei einem Tor nicht auf Startposition zurückgesetzt
        # False -> Ball wird bei Tor zurückgesetzt auf die Startposition
        self.infinite = False

        # Größe der Schläger von Spieler 0 und 1
        # (von der Mitte zum Ende, d.h hier die halbe Länge der gewünschten Gesamtlänge eintragen!)
        self.batsize = 1.0

        # Im Befehlsmodus kann der Schläger mit den Befehlen 'u' und 'd' bewegt werden.
        # Hier wird die dazugehörige Sprungweite des Schlägers angegeben.
        self.batstep = 0.3

        ############################################
        ### Initialisierungen (nicht verändern!) ###
        ############################################

        # Ortsvektor des Balles (Bezugspunkt ist [0,0])
        self.posVec = None

        # Richtungs-/Bewegungsvektor des Balles (Einheitsvektor)
        self.dirVec = None

        # Binärer Speicher, ob der Ball den einen Schläger getroffen hat [links, rechts]
        self._bathit = [False, False]

        # Binärer Speicher, ob der Ball die Linie geflogen ist [links, rechts]
        self._out = [False, False]

        # Punktestand [Spieler 0, Spieler 1]
        self.Points = [0, 0]

        # Der "Einschlagspunkt" des Balles auf der (Toraus-)Linie, wird erst nach einem Aufprall
        # mit konkreten Werten belegt und dann zur Fehlerberechnung genutzt (supervised learning).
        self.poi = [None, None]

        # Initiale Schlägerpositionen der Spieler auf ihren Linien.
        # [SchlängerLinks, SchlägerRechts]
        # Positionsänderungen sind somit, wie in Pong üblich, nur auf der Y-Achse möglich.
        self.bat = [self.y_max / 2.0, self.y_max / 2.0]

        # Zählt die Schlägertreffer (Kollisionen des Balles mit einem Schläger).
        # Die KNNs sollen unterschiedliche Winkel lernen (der Winkel wird immer zufallsinitialisiert),
        # bei ausreichender Lerndauer bzw. stark minimiertem Fehler jedoch sind die KNNs manchmal auf
        # einigen Winkeln derart talentiert, dass der Ball nie mehr über die Torlinie gehen würde.
        # Um ein solches "Endlosspiel" zu verhindern, wird der Ball nach 10 Treffern resettet,
        # das Spielfeld also zurückgesetzt mit einer initialen Ballposition auf der Spielfeldmitte und
        # neuem, zufallskalkuliertem Winkel.
        self.bouncecount = 0

        # Startvorbereitung
        # Initialisiert das erste Mal den Ortsvektor und Bewegungs-/Richtungsvektor
        self.__initvectors()


    def __initvectors(self):
        """
        Initialisiert Anfangs- und Richtungsballvektoren.
        Irgendwo in der Mitte auf der Y-Achse und mit einem belibigen Startwinkel.
        Der Startwinkel ist stets größergleich -45 Grad sowie kleinergleich +45 Grad von der Horizontalen aus gesehen.

        :return void
        """

        # Richtungsvektor erzeugen

        # Zufallswinkel im Bogenmaß generieren
        # 2 Pi entsprechen dem vollen Einheitskreis, also 360°
        # [-Pi/4, +Pi/4] entspricht einem Interval von [-45°, +45°]
        # Dieses Interval hat sich bewährt, da zu spitze den Lerneffekt und vor allem die Lerndauer
        # negativ beeinflussen.
        rotationAngle = np.random.uniform(-np.pi / 4, np.pi / 4)

        # Aus dem Zufallswinkel eine entsprechende Rotationsmatrix generieren
        rotMatrix = np.array([
            [np.cos(rotationAngle), -np.sin(rotationAngle)],
            [np.sin(rotationAngle), np.cos(rotationAngle)]
        ])

        # Rotationsmatrix auf einen Einheitsvektor (horizontale Ausrichtung) anwenden
        self.dirVec = np.dot(rotMatrix, np.array([1, 0]))

        # Zufällig entscheiden, ob der Ball nach links (zu Player 0) oder rechts (zu Player 1) startet.
        if random.random() > 0.5:
            self.dirVec[0] *= -1.0  # x-Komponente des Richtungs-/Bewegungsvektors wird an der Y-Achse gespiegelt

        # Ortsvektor erzeugen

        # Start irgendowo auf der Mittellinie
        # (x-Koordinate ist also fixiert auf die Mittellinie, y-Koordinate zufällig)
        self.posVec = np.array([self.x_max / 2.0, self.y_max * random.random()])

        # Rücksetzen der Anzahl der Schlägertreffer (__init__)
        self.bouncecount = 0


    def _incrpoints(self, player):
        """
        Erhöht den Punktestand für einen Spieler[Player]

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return void
        """
        self.Points[player] += 1


    def __sensor_x(self):
        """
        Gibt den X-Anteil des Ortsvektors des Balles mit Rauschen zurück

        :return float, X-Anteil vom Ortsvektor
        """
        return self.posVec[0] + (random.random() - 0.5) * self.outputNoiseMax


    def __sensor_y(self):
        """
        Gibt den Y-Anteil des Ortsvektors des Balles mit Rauschen zurück

        :return float, Y-Anteil vom Ortsvektor
        """
        return self.posVec[1] + (random.random() - 0.5) * self.outputNoiseMax


    def __sensor_bat(self, player):
        """
        Gibt die Position des Schlägers auf der Y-Achse von Spieler[Player] mit Rauschen zurück

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return float, Schlägerposition von Spieler[Player]
        """
        return self.bat[player] + (random.random() - 0.5) * self.outputNoiseMax


    def scaled_sensor_x(self):
        """
        Gibt den X-Anteil des Ortsvektors des Balles skaliert von -1 bis +1 mit Rauschen zurück
        (Rauschen kommt von __sensor_x())

        :return float, skalierter X-Anteil vom Ortsvektor
        """
        return self.__sensor_x() / (self.x_max / 2.0) - 1.0


    def scaled_sensor_y(self):
        """
        Gibt den Y-Anteil des Ortsvektors des Balles skaliert von -1 bis +1 mit Rauschen zurück
        (Rauschen kommt von __sensor_y())

        :return float, skalierter Y-Anteil vom Ortsvektor
        """
        return self.__sensor_y() / (self.y_max / 2.0) - 1.0


    def scaled_sensor_bat(self, player):
        """
        Gibt die Position des Schlägers von Spieler[Player] skaliert von -1 bis +1
        mit Rauschen zurück
        (Rauschen kommt von __sensor_bat())

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return float, skalierte Schlägerposition von Spieler[Player]
        """
        return self.__sensor_bat(player) / (self.y_max / 2.0) - 1.0


    def hitbat(self, player):
        """
        Gibt an, ob der Schläger von Spieler[Player] getroffen wurde oder nicht im aktuellen Tick/Spielzug.

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return Bool, Treffer (True) oder kein Treffer (False) vom Schläger von Spieler[Player]
        """
        return self._bathit[player]


    def scaled_sensor_err(self, player):
        """
        Gibt den Fehler von Spieler[Player] skaliert von -1 bis +1 zurück.

        :pre hitbat(player) or out(player)

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return float, skalierter Error von Spieler[Player]
        """
        return (self.poi[player] - self.__sensor_bat(player) ) / self.y_max


    def out(self, player):
        """
        Gibt an, ob der Ball die Linie von Spieler[Player] überschritten hat oder nicht.

        :param player: Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return Bool, Ball hat die Linie von Spieler[Player] überschritten (True) oder nicht überschritten (False)
        """
        return self._out[player]


    def getpoints(self, player):
        """
        Liefert die Punktanzahl von Spieler[Player]

        :param player: Punktzahl von Spieler 0 oder 1
        :type player: Int (0 oder 1)

        :return int, Punktzahl des Spielers
        """
        return self.Points[player]


    def tick(self):
        """
        Berechnet einen Tick/Spielzug,
        hierbei wird der Ball bewegt, die Überschreitung einer der Torauslinien
        oder die Kollision mit einem Schläger auf False initialisiert, außerdem
        die Ballposition zurückgesetzt, falls die Spieler den Ball zu oft hin und
        her gespielt haben ohne Tor (Endlosspiel verhindern).
        Ebenso wird überprüft, ob der Ball auf eine Bande getroffen ist und seinen
        Bewegungs-/Richtungsvektor ändern muss.
        Zum Schluss wird evaluiert, ob der Ball über die Torauslinie geflogen oder
        ob ein Schläger den Ball getroffen hat.

        :return void
        """

        #########################
        ### Initialisierungen ###
        #########################

        # Setzt den Ball eine Position weiter.
        # Die Schrittweite wird durch den Faktor self.speed gesetzt, der den Einheitsvektor dirVec skaliert
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
        # und neu gestartet, damit die aktuelle Endlosschleife unterbrochen wird. Hier würde das KNN
        # sonst nichts Neues mehr lernen.
        if self.bouncecount > 10:
            self.__initvectors()

        # Abprallen an der Unterseite bei Y = 0
        if self.posVec[1] < 0:
            self.posVec[1] *= -1.0
            self.dirVec[1] *= -1.0
        
        # Abprallen an der Oberseite bei Y = y_max (hier vermutlich 9)
        if self.posVec[1] > self.y_max:
            self.posVec[1] = 2 * self.y_max - self.posVec[1]
            self.dirVec[1] *= -1.0
        
        # Prüfe auf Treffer auf der linken Seite (Spieler 0)
        self.__tickBounceLeft()
        
        # Prüfe auf Treffer auf der rechten Seite (Spieler 1)
        self.__tickBounceRight()


    def __tickBounceLeft(self):
        """
        Checken, ob der Ball links bei Spieler 0 aus dem Spielfeld fliegt oder vom Schläger getroffen wird

        :return: void
        """

        # Wenn der Ortsvektor kleiner ist als 0, dann hat er die Torauslinie von Spieler 0 überschritten
        if self.posVec[0] < 0:

            # Berechne den theoretischen, genauen Aufprallpunkt (poi: PointOfImpact)
            # auf der Linie von Spieler 0 (Y = 0)

            factor = (0 - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec + (factor * self.dirVec)

            self.poi[0] = poi[1]  # Speichere diesen für eine evtl. spätere Nutzung von z.B. scaled_sensor_err(player)

            # Prüfe ob der Ball dann den Schläger getroffen hätte, wenn ja, dann...
            if (poi[1] > self.bat[0] - self.batsize) and (poi[1] < self.bat[0] + self.batsize):
                self._bathit[0] = True  # ... vermerke dies für z.B. hitbat(player)
            else:  # wenn jedoch nicht, dann...
                self.Points[1] += 1  # ... Punkte von Spieler 1 (rechts) erhöhen
                self._out[0] = True  # und merken, das der Ball außerhalb des Spielfelds
                # war, z.B. für out(player)

            # Ball abprallen lassen, falls:
            # -> Infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[0]:
                self.posVec[0] *= -1.0  # Einfallswinklel = Ausfallswinkel
                self.dirVec[0] *= -1.0

                self.bouncecount += 1  # Treffer vermerken, um bei zu vielen Treffern dieses neu zu starten
            else:
                self.__initvectors()  # Kein Treffer, somit das Spiel neu Initialisieren.
                self.bouncecount = 0


    def __tickBounceRight(self):
        """Checken, ob der Ball rechts bei Spieler 1 aus dem Spielfeld fliegt oder vom Schläger getroffen wird
        :return: void
        """
        # Wenn der Ortsvektor größer ist als x_max (hier vermutlich 16), dann hat er die Torauslinie
        # von Spieler 1 überschritten
        if self.posVec[0] > self.x_max:

            # Berechne den theoretischen, genauen Aufprallpunkt (poi: PointOfImpact) auf der Linie von
            # Spieler (Y = self.x_max)
            factor = (self.x_max - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec + (factor * self.dirVec)

            self.poi[1] = poi[1]  # Speichere diesen für eine evtl. spätere Nutzung von z.B. scaled_sensor_err(player)

            # Prüfe ob der Ball dann den Schläger getroffen hätte, wenn ja, dann...
            if poi[1] > self.bat[1] - self.batsize and poi[1] < self.bat[1] + self.batsize:
                self._bathit[1] = True  # ... vermerke dies für z.B. hitbat(player)
            else:  # wenn jedoch nicht, dann...
                self.Points[0] += 1  # ... Punkte von Spieler 0 (links) erhöhen
                self._out[1] = True  # und merken, das der Ball außerhalb des Spielfelds
                # war, z.B. für out(player)

            # Ball abprallen lassen, falls:
            # -> Das infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[1]:
                # 2 Spielfeldlängen - aktuellem X-Betrag ergibt neue X-Position
                self.posVec[0] = 2 * self.x_max - self.posVec[0]  # Einfallswinklel = Ausfallswinkel
                self.dirVec[0] *= -1.0

                self.bouncecount += 1  # Treffer vermerken, um bei zu vielen Treffern dieses neu zu starten
            else:
                self.__initvectors()  # Kein Treffer, somit das Spiel neu Initialisieren.
                self.bouncecount = 0


    def move(self, player, action):
        """
        Bewegt den Schläger eines Spielers
        Diese Funktion ist etwas Trickreich, da als "action"-Parameter sowohl ein String als direkter
        up/down-Befehl akzeptiert wird, als auch ein Float der den Schläger direkt setzt.

        :param player: Spieler 0 oder 1 (dessen Schläger bewegt werden soll)
        :type player: Int

        :param action: Wenn str, dann zwischen "d" oder "u" unterscheiden (Schläger hoch oder runter bewegen)
        :type action: String

        :param action: Wenn float, dann Schläger auf die entsprechende Position setzen
        :type action: float

        :return: void
        """

        # Wenn ein String, dann im Befehls-Modus:
        if type(action) == str:

            # Den Schläger nach oben bewegen
            if action == 'u':
                self.bat[player] += self.batstep
                if self.bat[player] > self.y_max:  # Korrektur, falls der obere Spielfeldrand erreicht wurde
                    self.bat[player] = self.y_max

            # Den Schläger nach unten bewegen
            if action == 'd':
                self.bat[player] -= self.batstep
                if self.bat[player] < 0.0:  # Korrektur, falls der untere Spielfeldrand erreicht wurde
                    self.bat[player] = 0.0

        # Sonst im Setzen-Modus:
        elif type(action) == float:
            self.bat[player] = (action + 1) * self.y_max / 2  # Der Schläger wird direkt auf die gewünschte Position gesetzt
            if self.bat[player] < 0.0:  # Korrektur, falls der untere Spielfeldrand erreicht wurde
                self.bat[player] = 0.0
            if self.bat[player] > self.y_max:  # Korrektur, falls der obere Spielfeldrand erreicht wurde
                self.bat[player] = self.y_max


    def v_getSize(self):
        """
        visu-getter

        :return float Liste [Float: X, Float: Y] der Spielfeldgröße
        """
        return [self.x_max, self.y_max]


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

        :return float Ortsvektor Liste [Float: X,Float: Y]
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