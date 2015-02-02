#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

"""
Die Visu sorgt für die Visualisierung der Applikation, also des Spielfedes und aller seiner Komponenten (Balles,
Schläger, usw.).
Die Visu läuft in einem eigenen Thread unter Benutzung des multiprocessing-Moduls von Python, dies erlaubt,
dass die Visu unabhängig von der Hauptapplikation (main.py) ausgeführt werden kann und umgekehrt.
"""

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

from tkinter import *
from multiprocessing import Process, Pipe
from telegramframe import TelegrammFrame

import socket
import json
import time


def netw_communication(conn):
    """
    Etabliert eine Netzwerkverbindung für die Kommunikation zwischen den Threads.
    Wartet darauf, dass die Hauptapplikation gestartet wird und Daten liefert.

    :param conn: Netzwerkkommuniationsobjekt, welches von Pipe erstellt und später an Process übergeben wird

    :return:
    """

    nc = NetwCon()
    connected = False  # initial besteht keine Verbindung

    # Solange noch keine Verbindung besteht
    while not connected:
        print('Try to connect...')  # Konsolenausgabe
        connected = nc.connect()    # Versuch der Verbindungsherstellung
        if not connected:           # Falls keine Verbindung hergestellt werden konnte
            time.sleep(1)           # Warten und im nächsten Schleifendurchlauf erneut versuchen
    print('connected, lets go...')

    while True:
        if conn.poll(None):  # Es wird auf neue Daten gewartet

            # Neue Daten wurden empfangen und werden verarbeitet
            frame = conn.recv()
            nc.send({'instruction': frame.instruction})

            retval = nc.receive()

            conn.send(retval)
            
            if frame.instruction == 'EXIT':
                break

    # Verbindung schließen
    nc.close()
    conn.close()


class NetwCon:
    """
    Netzwerk-Kommunikationsmodul, stellt die Verbindung her und sendet/empfängt Daten.
    """

    def __init__(self, sock=None):
        """

        :param sock:
        :return:
        """
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host='localhost', port=6769):
        """

        :param host:
        :param port:
        :return:
        """
        try:
            self.sock.connect((host, port))
            return True
        except socket.error as e:
            print('something\'s wrong with %s:%d. Exception type is %s' % (host, port, e))
            print('retrying in 1s')
            #sys.exit(1)
        return False
            
        
    def send(self, data):
        """
        Versucht Daten via JSON-Enkodierung (UTF8 Datenformat) zu packen und zu senden

        :param data: Zu sendende Daten

        :return: void
        """
        try:
            self.sock.send(bytes(json.dumps(data), 'UTF-8'))
        except socket.error as e:
            print ("Error sending data: %s" % e)
            sys.exit(1)
            
    def receive(self):
        """
        Versucht Daten zu empfangen und per JSON-Dekodierung (UTF8 Datenformat) zu interpretieren

        :return: Empfangene Daten
        """
        try:
            result = json.loads(self.sock.recv(8*1024).decode('UTF-8'))
        except socket.error as e:
            print ("Error receiving data: %s" % e)
            sys.exit(1)
        return result
        
    def close(self):
        """
        Schließt die Verbindung

        :return: void
        """
        self.sock.close()


class Application(Frame):
    """
    Visu-Applikationsklasse, welche Daten über Benutzerinteraktionen (z.B. Buttons) über das Netzwerk sendet
    und für die GUI, die grafische Darstellung des aktuellen Applikationszustandes, verantwortlich ist.

    Die Klasse übernimmt neben dem initialen Erstellen der GUI auch das stetige Updaten der Zeichnung.
    """


    def __init__(self, conn, master=None):
        """
        Konstruktur der Klasse, setzt initiale Zustände, zeichnet das Spielfeld, ruft dessen
        Aktualisierung auf.

        :param conn: Netzwerkverbindung

        :param master:

        :return:
        """
        Frame.__init__(self, master)  # Konstruktor der Super-Klasse aufrufen
        self.init = True              # Initialisierungswert
        self.pack()                   # Platzierungseinstellungen: Standard ist Top ohne Whitespaces zwischen Objekten
        self.conn = conn              # Netzwerkkommunikationsobjekt
        self.createWidgets()          # GUI-Objekte (Spielfeld, Ball, ...) zeichnen
        self.factor = 50.0            # Faktor für die einfache Skalierung der GUI-Objekte
        self.points = [0, 0]          # Container für den aktuellen Punktestand
        self.update()                 # GUI updaten (neu zeichnen)


    def quitserver(self):
        """
        Beende den Server, sendet ihm die Beendenanforderung

        :return: none
        :rtype: void
        """

        print("send EXIT signal to Server...")
        conn.send(TelegrammFrame('EXIT'))


    def savedata(self):
        """
        Sendet dem Server die Speicheranforderung

        :return: none
        :rtype: void
        """

        print("send save signal to Server...")
        conn.send(TelegrammFrame('saveConfig'))


    def togglespeed(self):
        """
        Sendet dem Server die Geschwindigkeitsänderungsanforderung

        :return: none
        :rtype: void
        """
        print('send speed toggle signal to Server...')
        conn.send(TelegrammFrame('CHSPEED'))
        if conn.poll(None):  # warte auf neue Daten...
            print(conn.recv()) # Daten sind da...


    def createWidgets(self):
        """
        Erstellt die GUI Elemente im Fenster

        :return:
        """
        self.master.title("Pong mit rekurrentem Netz") # Setzt Fenstertitel

        self.bSAVE = Button(self)                   # Erzeugt den Button: 'Save Config'
        self.bSAVE["text"] = "Save Config"
        self.bSAVE["command"] =  self.savedata
        self.bSAVE["state"] = DISABLED
        self.bSAVE.pack({"side": "left"})

        self.bQUIT = Button(self)                   # Erzeugt den Button: 'Quit Visu only'
        self.bQUIT["text"] = "Quit Visu only"
        self.bQUIT["command"] =  self.quit
        self.bQUIT.pack({"side": "left"})
        
        self.bQGoS = Button(self)                   # Erzeugt den Button: 'Quit'
        self.bQGoS["text"] = "Quit"
        self.bQGoS["command"] = self.quitserver
        self.bQGoS.pack({"side": "left"})

        self.bTogSp = Button(self)              # Erzeugt den Button: 'Toggle Speed'
        self.bTogSp["text"] = "Toggle Speed"
        self.bTogSp["command"] = self.togglespeed
        self.bTogSp.pack({"side": "left"})

        self.bottomframe = Frame(self.master)
        self.bottomframe.pack( side = BOTTOM )

        self.infoboxsize = 23 # Parameter zur Breitenfestlegung der Infoboxen

        # Infobox für Spieler0
        self.p0 = Text(self.bottomframe, height=6, width=self.infoboxsize)
        self.p0.pack({"side": LEFT})
        self.p0.insert(END, "Player 0\n")

        # Erstellt die Zeichenfläche für das Spielfeld
        self.court = Canvas(self.bottomframe, width=900, height=500)
        self.court.pack({"side": LEFT})

        # Infobox für Spieler1
        self.p1 = Text(self.bottomframe, height=6, width=self.infoboxsize)
        self.p1.pack({"side": LEFT})
        self.p1.insert(END, "Player 1\n")

    def update(self):
        """
        Update Funktion für die GUI-Elemente, wird regelmäßig aufgerufen

        :return: none
        """
        if self.init: # Wenn noch nie Initialisiert wurde, dann initialisiere!
            conn.send(TelegrammFrame('INIT')) #Frage init Daten an...
            
            if conn.poll(None):  # warte auf neue Daten...
                data = conn.recv() # Daten sind da...

                # Kopiere Daten in Variablen
                self.size = data['size']
                self.batsize = data['batsize']
                self.p1name = data['p1name']
                self.p2name = data['p2name']
                self.timestamp = time.time()
                
                self.drawCourt()       # Zeichne das Spielfeld neu bzw. initial
                self.init = False      # Initialisierung wurde abgeschlossen
                
        conn.send(TelegrammFrame('REFRESH')) #Frage Refresh Daten an...
        if conn.poll(None):  # warte auf neue Daten...
            data = conn.recv() # Daten sind da...

            # Kopiere Daten in Variablen
            self.speed = data['mainloopdelay']
            self.posvec = data['posvec']
            self.dirvec = data['dirvec']
            self.posX = data['sensor_posX']
            self.posY = data['sensor_posY']
            self.bat = data['bat']
            self.points = data['points']
            self.rewcount = data['rewcount']
            self.hitratio = data['hitratio']


            # Bereite einen String vor, der in die Infobox geschrieben wird (Spieler0)
            info = 'Player 0\n\n'
            info += 'Bat positon'.ljust(18, ".")\
                    + (str(round(self.bat[0], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Points'.ljust(18, ".")\
                    + (str(self.points[0])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Rewards'.ljust(18, ".")\
                    + (str(self.rewcount[0])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Hitratio'.ljust(18, ".")\
                    + (str(round(self.hitratio[0], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'

            self.p0.delete("1.0",END) # lösche Infobox
            self.p0.insert(END, info) # schreibe Infobox mit neuen Informationen

            # Bereite einen String vor, der in die Infobox geschrieben wird (Spieler0)
            info = 'Player 1\n\n'
            info += 'Bat positon'.ljust(18, ".")\
                    + (str(round(self.bat[1], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Points'.ljust(18, ".") \
                    + (str(self.points[1])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Rewards'.ljust(18, ".") \
                    + (str(self.rewcount[1])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Hitratio'.ljust(18, ".") \
                    + (str(round(self.hitratio[1], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'

            self.p1.delete("1.0",END) # lösche Infobox
            self.p1.insert(END, info) # schreibe Infobox mit neuen Informationen

            self.updateCurt()  # Zeiche das Spielfeld neu
            
        self.lastrefresh = self.timestamp
        self.timestamp = time.time()
        self.after(50, self.update) # wieder aufrufen nach 50 ms


    def updateCurt(self):
        """
        Neuzeichnen des Spielfeldes

        :return: none
        """
        r = 5 # radius vom Ball
        # Schläger für Spieler 0 zeichnen
        self.court.coords(self.c_bat0,  10, (self.factor * self.bat[0]) + (self.factor * self.batsize),10,(self.factor * self.bat[0]) - (self.factor * self.batsize) )

        # Schläger für Spieler 1 zeichnen
        self.court.coords(self.c_bat1,  self.factor * self.size[0]+10, (self.factor * self.bat[1]) + (self.factor * self.batsize),self.factor * self.size[0]+10,
            (self.factor * self.bat[1]) - (self.factor * self.batsize) )

        # Ball zeichnen
        self.court.coords(self.c_ball,  self.posvec[0]*self.factor-r + 10, self.posvec[1]*self.factor-r, self.posvec[0]*self.factor+r+ 10, self.posvec[1]*self.factor+r )
                                        
        # Richtungsvektor zeichnen
        self.court.coords(self.c_direction, self.posvec[0]*self.factor+ 10, self.posvec[1]*self.factor, self.posvec[0]*self.factor + self.dirvec[0] * self.factor + 10, self.posvec[1]*self.factor + self.dirvec[1] * self.factor)

        #Punkte von den Spilern zeichnen
        self.court.itemconfig(self.p0points,text=str(self.points[0]))
        self.court.itemconfig(self.p1points,text=str(self.points[1]))


        self.court.update()
        
        
    def drawCourt(self):
        """
        Spielfeld initial zeichnen

        :return:none
        """
        
        # Zeichen das Spielfeld
        self.court.create_rectangle(5,0, self.factor * self.size[0]+15,self.factor * self.size[1],fill="black")

        
        # Vertikale Mittellinie zeichnen
        self.court.create_line(self.factor * self.size[0]/2+10, 0, self.factor * self.size[0]/2+10, self.factor * self.size[1], fill="white", dash=(4, 4))
        
        # Spieler0 Schläger zeichnen
        self.c_bat0 = self.court.create_line(
            10,
            50,
            10,
            70,  
            fill="white",width=8)

        # Spieler1 Schläger zeichnen
        self.c_bat1 = self.court.create_line(
            self.factor * self.size[0]+10,
            70,
            self.factor * self.size[0]+10,
            90,  
            fill="white",width=8)
        
        r = 5
        # Ball zeichen
        self.c_ball = self.court.create_oval(100-r + 10, 100-r, 100+r+ 10, 100+r,fill="white")

        # Richtungsvektor zeichnen
        self.c_direction = self.court.create_line( 100, 100, 
                                110, 110, fill="white",arrowshape=(8,10,3),arrow="last")

        # Spielstände zeichnen
        self.p1points = self.court.create_text(self.factor * self.size[0] / 2  + 10 + 150 ,20,text = str(self.points[1]),fill='white',font = ('ARIAL',30))
        self.p0points = self.court.create_text(self.factor * self.size[0] / 2  - 10 - 150 ,20,text = str(self.points[0]),fill='white',font = ('ARIAL',30))

        root.after(1000,self.updateCurt) # Update nach 1s aufrufen
        

if __name__ == '__main__':

    conn, communication_conn = Pipe() # Kommunikationskanal zwischen Socketserver und GUI öffnen
    nwc = Process(target=netw_communication, args=(communication_conn,)) # Prozess vorbereiten
    nwc.start() # Prozess starten
    
    # Neues Fenster öffnen
    root = Tk()
    app = Application(conn, master=root)


    app.mainloop() # Hauptscheife starten
    root.destroy() # Fenster beenden
    
    nwc.join() # Warten auf beenden des Socketservers
