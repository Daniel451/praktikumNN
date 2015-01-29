#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
""" Enthält die Definition zur Kommunikation zwischen den Threads"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

class DataFrame:
    """
    DataFrame stellt ein Objekt dar, das dazu genutzt werden kann, Information zu transportieren.
    Es besteht aus einer "Instruction" als String und kann beliebig viele Daten angehangen haben.
    Daten werden mit einem Key identifiziert.
    """

    def __init__(self,instruction='NULL'):
        """
        Initialisiert das Datenpaket
        :param instruction String welcher das Datenpaket identifiziert
        und meist auch als direkte Befehl zu deuten ist.
        :rtype : void
        """
        self.instruction = instruction # Befehl festlegen
        self.data = {'key':'value'}  # Datenbereich initialisieren (könnte auch mit none funktionieren...)


    def add(self,identifier, value):
        """
        Fügt Daten mit einem Key dem Datenpaket hinzu.

        :param identifier: Schlüsselname für die Daten
        :type identifier: str
        :param value: Daten die angehängt werden sollen
        :type value: object
        :rtype: void
        """
        self.data[identifier] = value

    def getdata(self, identifier):
        """
        läd Daten mit einem Key aus dem Datenpaket
        :param identifier: Key der Daten
        :type identifier: str
        :return: Daten die mit dem Key assoziiert sind
        :rtype: object
        """
        return self.data[identifier]

    def instruction(self):
        """
        gibt den Befehl zurück der das Datenpaket identifiziert
        :return: Befehl
        :rtype: str
        """
        return self.instruction