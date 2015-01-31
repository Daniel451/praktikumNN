#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

"""
Diese Datei stellt den Eintrittspunkt der Kernapplikation zur Verfügung.
Von hier aus wird das Programm gestartet.

Nach einigen imports und Klassendefinitionen startet die Hauptschleife, welche dafür sorgt, dass die KNNs, welche
zuvor gestartet wurden, ihre Daten (Positionsvektor des Balles) erhalten und daraufhin ihre predictions zurückgeben
können.
Weiterhin sorgt diese Hauptschleife ebenfalls dafür, dass die KNNs ihre "Rewards", also den Fehler / das aktuelle
Delta zum Lernen erhalten.

Je nach Implementation kann dies allerdings auch bedeuten, dass das Netz nicht unseren vorgestellten supervised
learning Ansatz verfolgt, sondern z.B. unseren anfangs erdachten reinforcement learning Algorithmus verwendet.
Die Implementation und damit die Strategie des Lernens oder auch andere Techniken können also verwendet werden,
um beim nächsten Ballanflug eine bessere Prediction zu liefern.
"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

from multiprocessing import Process, Pipe
from knnframe import Knnframe
from court import court
from telegramframe import TelegrammFrame

import logging

import sys
import threading
import os.path

import json
import socketserver
import time
import datetime




#Todo: kommentieren! irgendwie...

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

class MyTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

class MyTCPServerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                data = json.loads(self.request.recv(8*1024).decode('UTF-8').strip())

            except Exception as e:
                print("Exception while receiving message: ", e)
                return
            
            try:
                instruction = data['instruction']

                if instruction == 'EXIT':
                    # send some 'ok' back
                    print('Exiting...')
                    self.request.sendall(bytes(json.dumps({'return':'ok'}), 'UTF-8'))
                    self.__shutdown_request = True
                    exitapp()
                    self.__is_shut_down.wait()
                    sys.exit(1)
                    break
                
                elif instruction == 'INIT':
                    self.request.sendall(bytes(json.dumps({
                                                            'return': 'ok',
                                                            'size': court.v_getSize(),
                                                            'batsize': court.v_getBatSize(),
                                                            'p1name': 'John A. Nunez',
                                                            'p2name': 'Cynthia J. Wilson',
                                                            }), 'UTF-8'))
                elif instruction == 'REFRESH':
                    self.request.sendall(bytes(json.dumps({
                                                            'return': 'ok',
                                                            'mainloopdelay': court.v_getSpeed(),
                                                            # aktueller Positionsvektor
                                                            'posvec': court.v_getPosVec().tolist(),
                                                            # aktueller Richtungsvektor
                                                            'dirvec': court.v_getDirVec().tolist(),
                                                            # Schlägerposition
                                                            'bat': court.v_getbat(),
                                                            # Spielstand
                                                            'points': court.v_getPoint(),
                                                            # Skalierte Positionsdaten des Schlägers für KNNs
                                                            'sensorP1_bat': court.scaled_sensor_bat(0),
                                                            'sensorP2_bat': court.scaled_sensor_bat(1),
                                                            # Skalierte Positionsdaten des Balls für KNNs
                                                            'sensor_posX': court.scaled_sensor_x(),
                                                            'sensor_posY': court.scaled_sensor_y(),
                                                            }), 'UTF-8'))
                elif instruction == 'saveConfig':
                    x = TelegrammFrame('saveConfig')
                    connPlayer0.send(x)
                    connPlayer1.send(x)

                elif instruction == 'CHSPEED':
                    self.request.sendall(bytes(json.dumps({'return': 'ok'}), 'UTF-8'))
                    print ('change mainloopdelay to...')
                    changespeed()

                else:
                    self.request.sendall(bytes(json.dumps({'return': 'not ok'}), 'UTF-8'))

            except Exception as e:
                print("No instruction available: ", e)
                return



def startplayer(conn, playerid, loadconfig=None):
    """
    Hauptschleife für die Spieler-Threads.
    Sorgt dafür, dass die Anforderungen von der Hauptschleife korrekt abgearbeitet werden.

    :param conn: Kommunikationstunnel zur Hauptschleife.
    :type conn: multiprocessing.Pipe

    :param playerid: Spieler Identifikationsnummer (wird hauptsächlich für Dateinamen gebraucht)
    :type playerid: int

    :param loadconfig: Dateiname mit Konfiguration vom Spieler, default=Null, d.h. eine neue wird erzeugt.
    :type loadconfig: str

    :return: none
    :rtype: void
    """

    # Es wird ein Framework, quasi ein "Wrapper-Objekt", genutzt um ein KNN zu erstellen.
    # Jenes hat den Vorteil, das beliebige/mehrere Implementationen vonverschiendenen KIs,
    # relativ einfach integriert werden können in die vorhandene Infrastruktur.
    #
    # Das Interface von Knnframe.py muss jedoch immer erhalten bleiben ("Inferface-Pattern").

    player = Knnframe(loadconfig, playerid)

    # Hauptschleife für die Spieler
    while True:

        # Warte auf neue Daten von der Hauptschleife in main
        # if conn.poll(None) unterbricht den Progammfluss und wartet bis wirklich Daten vorhanden sind!
        if conn.poll(None):

            # Lade das Telegramm und werte es aus
            telegramm = conn.recv()

            if telegramm.instruction == 'EXIT':
                # Anforderung zum Beenden erhalten, springe aus der Schleife
                break

            elif telegramm.instruction == 'predictNext':
                # Anforderung eine neue Vorhersage zu treffen erhalten
                # Rufe die entsprechende Funktion auf und sende die Antwort zurück an die Hauptschleife
                conn.send(requestprediction(player, telegramm))

            elif telegramm.instruction == 'reward_pos':
                # Positive Belohnung erhalten (Schläger hat Ball getroffen)
                # Rufe die entsprechende Funktion im Interface des Frameworks für die das KNN auf
                err = float(telegramm.getdata('err'))
                player.reward_pos(err)

            elif telegramm.instruction == 'reward_neg':
                # Negative Belohnung erhalten (Ball hat die Linie überschritten, Schläger hat Ball nicht getroffen)
                # Rufe die entsprechende Funktion im Interface des Frameworks für die das KNN auf
                err = float(telegramm.getdata('err'))
                player.reward_neg(err)

            elif telegramm.instruction == 'saveConfig':
                # Anforderung die Konfiguration zu speichern erhalten

                # Bilde den Pfad und Dateinamen aus Zeit/Datum und Spieler ID
                path = 'save/config_' + str(playerid) + '_'\
                       + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

                # Rufe die entsprechende Funktion im Interface des Frameworks zum Speichern für das KNN auf
                player.saveconfig(path)

            else:
                # Fehlerhafte, unbekannte Anforderung erhalten
                print('Player ' + str(playerid) + ': unknown instruction: ' + telegramm.instruction)

    # Bei einer EXIT - Anforderung wird der Kommunikationstunnel geschlossen und der Thread beendet
    conn.close()


def saveconfig(player, saveconfigtelegramm):
    """
    Weißt Spieler an, die Konfiguration zu speichern.

    :param player: das Frame/Wrapperobjekt des Spielers
    :type player: Knnframe

    :param saveconfigtelegramm: Telegramm mit Daten (filename)
    :type saveconfigtelegramm: TelegrammFrame

    :return: none
    :rtype: void
    """
    player.saveconfig(str(saveconfigtelegramm.getdata('filename')))


def requestprediction(player, requesttelegramm):
    """
    Der Spieler wird aufgefordert eine neue Vorhersage für die aktuelle Situation zu treffen.

    :param player: das Frame/Wrapperobjekt des Spielers
    :type player: Knnframe

    :param requesttelegramm: Telegramm mit den Daten (x,y,mypos)
    :type requesttelegramm: TelegrammFrame

    :return: Antwort mit Aktion vom Spieler (move, kann str (up, down, ... -> diskrete Werte) oder
     float (normiert von -1 .. +1) sein, je nach Implementation
    :rtype: TelegrammFrame
    """

    # Gebe Anforderung weiter und erwarte eine Aktion vom Spieler
    action = player.predict(requesttelegramm.getdata('xpos'), requesttelegramm.getdata('ypos'), requesttelegramm.getdata('mypos'))

    # Baue Aktion zu einem Telegramm zusammen
    returnframe = TelegrammFrame('Return')
    returnframe.add('move',action)

    # Telegramm zurückgeben
    return returnframe


def exitapp():
    """
    Setzt die Beendenanforderung für die Hauptschleife.

    :return: none
    :rtype: void
    """

    # Globale Variable exitrequest benutzen
    global exitrequest
    exitrequest = True


def changespeed():
    """
    Verändert die Geschwindigkeit des Spiels, dies hat keinen Einfluss auf die Spieler an sich,
    sondern hilft nur dem Betrachter des Spieles in der Visualisierung einen Überblick zu behalten.
    Es wird also nicht die Ausführungsgeschwindigkeit der Agenten geändert, sondern nur die Tickrate.

    :return: none
    :rtype: void
    """
    global mainloopdelay
    if mainloopdelay > 0.0:
        mainloopdelay = 0.0
    else:
        mainloopdelay = 0.3


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


# +================================+
# +*********  main start  *********+
# +================================+

if __name__ == '__main__':

    # Für effizientes debugging und logging können Werte und Informationen in eine
    # Datei geschrieben werden.
    file = "log_pong.log"
    createlogfile(file)
    logging.basicConfig(filename=file, level=logging.INFO)
    logging.info('==================')
    logging.info('====  Started  ===')
    logging.info('==================')

    # Die maximale Geschwindigkeit wirkt sich positiv auf die Lerngeschwindigkeit aus,
    # jedoch negativ für die Beobachtung/Visualisierung.
    #
    # Die Main-Schleife kann künstlich verzögert werden, damit das Spiel für einen Beobachter erkennbar bleibt.
    # Dieser Wert wird normalerweise über die Funktion changespeed() von der Visualisierung verändert.
    #
    # Die maximale Lerngeschwindigkeit wird mit 0.0 erreicht.
    # Zum Beobachten / für die Visualisierung eignen sich i.d.R. Werte größer als 0.3.
    # Die Variable mainloopdelay muss ein float größergleich 0.0 sein.
    # Die Einheit der Variable sind Sekunden.

    global mainloopdelay
    mainloopdelay = 0.0

    global exitrequest
    exitrequest = False

    # Erstelle das Spielfeld
    court = court()

    # Erzeuge jeweils einen Kommunikationstunnel zwischen der main-Schleife und den Spielern (MLPs)
    givePlayer0, connPlayer0 = Pipe()  # Tunnel zu Spieler0
    givePlayer1, connPlayer1 = Pipe()  # Tunnel zu Spieler1

    # Bereite die Threads für Spieler 0 und 1 vor.
    # Beim Starten wird die Funktion startplayer(conn,playername, loadconfig = None) aufgerufen.
    # Sie beinhaltet eine eigene Endlosschleife und wartet auf Befehle die durch den Kommunikationstunnel kommen.
    p0 = Process(target=startplayer, args=(givePlayer0, 0))
    p1 = Process(target=startplayer, args=(givePlayer1, 1))

    # Starte die Threads
    p0.start()
    p1.start()


    # Die Visualierung kommuniziert über TCP/IP mit dem Server.
    # Die API wird dafür nachfolgend geöffnet:

    # Adresse für den host
    # Server soll nur für den localhost erreichbar sein
    bindaddress = "localhost"
    # Port für die Verbindung (muss frei sein, darf also nicht durch ein anderes Programm benutzt/gebunden werden)
    bindport = 6769


    # Erstelle GUI-Server mit den bekannten Verbindungsdaten
    guiserver = ThreadedTCPServer((bindaddress, bindport), MyTCPServerHandler)
    ip, port = guiserver.server_address

    # Bereite einen parallelen Thread vor (die Hauptschleife wird dann nicht geblockt)
    server_thread = threading.Thread(target=guiserver.serve_forever)

    # Starte GUI-Server, damit er Verbindungen annehmen kann
    server_thread.daemon = True
    server_thread.start()

    logging.info("Server loop running in thread: " + str(server_thread.name))
    logging.info("listening on: " + str(ip) + ':' + str(port) )
    logging.info("starting main loop... ")


    # Hauptschleife, versorgt Spieler mit Daten und koordiniert den Ablauf
    while True:

        # Das Spiel ist in Spielrunden (Ticks) eingeteilt, jede Runde beginnt damit, dass das Spielfeld
        # die neue Ballposition berechnet.
        # Siehe hierzu die Datei: court.py
        court.tick()


        # Wenn das Spielfeld erkennt, das ein Schläger getroffen wurde, dann...
        if court.hitbat(0) or court.hitbat(1):
            rewardframepos = TelegrammFrame('reward_pos')

            if court.hitbat(0): # ... gib entweder Spieler 0 die positive Belohnung ...
                rewardframepos.add('err', court.scaled_sensor_err(0))
                # (Für eine weitere Verbesserung, also wenn der Schläger nicht ganz mittig getroffen wurde,
                # werden jeweils noch das Delta zwischen Auftreffpunkt und Schlägermittelpunkt mitgegeben)
                connPlayer0.send(rewardframepos)
                logging.info("Player 0 got a positive reward! Error: " + str(court.scaled_sensor_err(0)))
            if court.hitbat(1):   # ... oder Spieler 1 die positive Belohnung.
                rewardframepos.add('err', court.scaled_sensor_err(1))
                # (Für eine weitere Verbesserung, also wenn der Schläger nicht ganz mittig getroffen wurde,
                # werden jeweils noch das Delta zwischen Auftreffpunkt und Schlägermittelpunkt mitgegeben)
                connPlayer1.send(rewardframepos)
                logging.info("Player 1 got a positive reward!" + str(court.scaled_sensor_err(1)))


        # Sollte das Spielfeld erkennen, dass der Ball nicht vom Schlager getroffen, sondern über die
        # (Spielaus)Linie geflogen ist, dann wird eine negative Belohnung gesendet.
        if court.out(0) or court.out(1):
            rewardframeneg = TelegrammFrame('reward_neg')

            if court.out(0):
                rewardframeneg.add('err', court.scaled_sensor_err(0))
                connPlayer0.send(rewardframeneg)
                logging.info("Player 0 got a negative reward!" + str(court.scaled_sensor_err(0)))

            elif court.out(1):
                rewardframeneg.add('err', court.scaled_sensor_err(1))
                connPlayer1.send(rewardframeneg)
                logging.info("Player 1 got a negative reward!" + str(court.scaled_sensor_err(1)))

        # Zusammengefasst, kann man erkennen, das nur Belohnungen gegeben werden, wenn der Ball über der Linie war
        # oder der Schläger den Ball getroffen hat.
        #
        # Die Zeit dazwischen, d.h. wenn sich der Ball irgendwo auf dem Spielfeld bewegt,
        # werden -keine- Rückmeldungen gesendet (Delayed-Reward-Problem, wie in Präsentation vorgestellt).
        # Es ist dann also unbekannt, ob die gegenwärtige Situation positiv oder negativ zu bewerten ist.


        # Bereite die Prediction-Anforderung für Spieler 0 und 1 vor, enthalten sollten sein:
        prednextreqPlayer0 = TelegrammFrame('predictNext')
        prednextreqPlayer1 = TelegrammFrame('predictNext')

        # die aktuelle, skalierte X-Ballposition (-1..+1)
        prednextreqPlayer0.add('xpos', court.scaled_sensor_x())
        prednextreqPlayer1.add('xpos', court.scaled_sensor_x())

        # die aktuelle, skalierte Y-Ballposition (-1..+1)
        prednextreqPlayer0.add('ypos', court.scaled_sensor_y())
        prednextreqPlayer1.add('ypos', court.scaled_sensor_y())

        # die aktuelle, skalierte Schlägerposition (-1..+1) des eigenen Schlägers
        prednextreqPlayer0.add('mypos', court.scaled_sensor_bat(0))
        prednextreqPlayer1.add('mypos', court.scaled_sensor_bat(1))

        # Information an die Agenten/Spieler senden
        connPlayer0.send(prednextreqPlayer0)
        connPlayer1.send(prednextreqPlayer1)


        # Hier wird die Verzögerung des Spielverlaufs zwecks besser Visualisation gesetzt:
        if mainloopdelay > 0:
            time.sleep(mainloopdelay)


        # Wir warten nun auf Aktionsdaten von Spieler 0 und 1:

        # Zwischeninformation:
        # Die Funktion x.poll(None) wartet unendlich lange auf Daten, der
        # Programmfluss wir hier also verzögert!
        # Alternativ wäre eine Endlosschleife möglich,
        # diese würde aber mehr Rechenleistung in Anspruch nehmen.

        if connPlayer0.poll(None):                     # Warten auf Daten von Spieler0, ...
            frame = connPlayer0.recv()                 # ... wenn vorhanden, dann abfragen und ...
            if frame.instruction == 'Return':
                court.move(0, frame.getdata('move'))   # ...anschließend von Spielfeld ausführen lassen.

        if connPlayer1.poll(None):                     # Warten auf Daten von Spieler0, ...
            frame = connPlayer1.recv()                 # ... wenn vorhanden, dann abfragen und ...
            if frame.instruction == 'Return':
                court.move(1, frame.getdata('move'))   # ...anschließend von Spielfeld ausführen lassen.

        # Wiederholen, bis ...
        if exitrequest:  # ...die Beendenanforderung erhalten wurde
            break



    # Teile den Spielern mit, dass das Programm beendet wird und sie
    # aufgefordert sind sich ebenfalls zu beenden.
    frame = TelegrammFrame('EXIT')
    logging.info('exit request to Player0')
    connPlayer0.send(frame)
    logging.info('exit request to Player1')
    connPlayer1.send(frame)

    # Warte auf die Spielerthreads, wenn sie beendet sind, ...
    p0.join()
    p1.join()

    # ...warten wir noch auf den GUI-Server, dieser sollte den Port wieder freigeben.
    guiserver.shutdown()

    # Das Programm beendet sich nun selbst!
    logging.info('==================')
    logging.info('==  Terminated  ==')
    logging.info('==================')