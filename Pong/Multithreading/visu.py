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


    def QGoS(self):
        """

        :return:
        """
        print("Send EXIT Signal to Server...")
        conn.send(TelegrammFrame('EXIT'))


    def save(self):
        """

        :return:
        """
        print("Send Save Signal to Server...")
        conn.send(TelegrammFrame('saveConfig'))


    def Togglespeed(self):
        """

        :return:
        """
        print('send Speed Toggle')
        conn.send(TelegrammFrame('CHSPEED'))
        if conn.poll(None):  # warte auf neue Daten...
            print(conn.recv()) # Daten sind da...


    def createWidgets(self):
        """

        :return:
        """
        self.master.title("Pong mit rekurrentem Netz")

        self.bSAVE = Button(self)
        self.bSAVE["text"] = "Save Config"
        self.bSAVE["command"] =  self.save
        self.bSAVE["state"] = DISABLED
        self.bSAVE.pack({"side": "left"})

        self.bQUIT = Button(self)
        self.bQUIT["text"] = "Quit visu only"
        self.bQUIT["command"] =  self.quit
        self.bQUIT.pack({"side": "left"})
        
        self.bQGoS = Button(self)
        self.bQGoS["text"] = "Quit"
        self.bQGoS["command"] = self.QGoS
        self.bQGoS.pack({"side": "left"})

        self.togglespeed = Button(self)
        self.togglespeed["text"] = "Toggle Speed"
        self.togglespeed["command"] = self.Togglespeed
        self.togglespeed.pack({"side": "left"})

        self.bottomframe = Frame(self.master)
        self.bottomframe.pack( side = BOTTOM )

        self.infoboxsize = 23

        self.p0 = Text(self.bottomframe, height=6, width=self.infoboxsize)
        self.p0.pack({"side": LEFT})
        self.p0.insert(END, "Player 0\n")

        self.court = Canvas(self.bottomframe, width=900, height=500)
        self.court.pack({"side": LEFT})

        self.p1 = Text(self.bottomframe, height=6, width=self.infoboxsize)
        self.p1.pack({"side": LEFT})
        self.p1.insert(END, "Player 1\n")

    def update(self):
        """

        :return:
        """
        if self.init:
            conn.send(TelegrammFrame('INIT')) #Frage init Daten an...
            
            if conn.poll(None):  # warte auf neue Daten...
                data = conn.recv() # Daten sind da...
                
                self.size = data['size']
                self.batsize = data['batsize']
                self.p1name = data['p1name']
                self.p2name = data['p2name']
                self.timestamp = time.time()
                
                self.drawCourt()
                self.init = False
                
        conn.send(TelegrammFrame('REFRESH')) #Frage init Daten an...
        if conn.poll(None):  # warte auf neue Daten...
            data = conn.recv() # Daten sind da...
            
            self.speed = data['mainloopdelay']
            self.posvec = data['posvec']
            self.dirvec = data['dirvec']
            self.posX = data['sensor_posX']
            self.posY = data['sensor_posY']
            self.bat = data['bat']
            self.points = data['points']
            self.rewcount = data['rewcount']
            self.hitratio = data['hitratio']

            info = 'Player 0\n\n'
            info += 'Bat positon'.ljust(18, ".")\
                    + (str(round(self.bat[0], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Points'.ljust(18, ".")\
                    + (str(self.points[0])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Rewards'.ljust(18, ".")\
                    + (str(self.rewcount[0])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Hitratio'.ljust(18, ".")\
                    + (str(round(self.hitratio[0], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'

            self.p0.delete("1.0",END)
            self.p0.insert(END, info)

            info = 'Player 1\n\n'
            info += 'Bat positon'.ljust(18, ".")\
                    + (str(round(self.bat[1], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Points'.ljust(18, ".") \
                    + (str(self.points[1])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Rewards'.ljust(18, ".") \
                    + (str(self.rewcount[1])).rjust(self.infoboxsize - 18, ".") + '\n'
            info += 'Hitratio'.ljust(18, ".") \
                    + (str(round(self.hitratio[1], 2)).ljust(4, "0")).rjust(self.infoboxsize - 18, ".") + '\n'

            self.p1.delete("1.0",END)
            self.p1.insert(END, info)

            self.updateCurt()
            
        self.lastrefresh = self.timestamp
        self.timestamp = time.time()
        self.after(50, self.update)


    def updateCurt(self):
        """

        :return:
        """
        r = 5
        self.court.coords(self.c_bat1,  10, (self.factor * self.bat[0]) + (self.factor * self.batsize),10,(self.factor * self.bat[0]) - (self.factor * self.batsize) )
        self.court.coords(self.c_bat2,  self.factor * self.size[0]+10, (self.factor * self.bat[1]) + (self.factor * self.batsize),self.factor * self.size[0]+10, 
            (self.factor * self.bat[1]) - (self.factor * self.batsize) )
                                        
        self.court.coords(self.c_ball,  self.posvec[0]*self.factor-r + 10, self.posvec[1]*self.factor-r, self.posvec[0]*self.factor+r+ 10, self.posvec[1]*self.factor+r )
                                        
                                        
        self.court.coords(self.c_direction, self.posvec[0]*self.factor+ 10, self.posvec[1]*self.factor, self.posvec[0]*self.factor + self.dirvec[0] * self.factor + 10, self.posvec[1]*self.factor + self.dirvec[1] * self.factor)

        self.court.itemconfig(self.p0points,text=str(self.points[0]))
        self.court.itemconfig(self.p1points,text=str(self.points[1]))


        self.court.update()
        
        
    def drawCourt(self):
        """

        :return:
        """
        
        # court

        self.court.create_rectangle(5,0, self.factor * self.size[0]+15,self.factor * self.size[1],fill="black")
        
        #create_line(x1, y1, x2, y2, width=1, fill="black")
        
        # vertical middle line
        self.court.create_line(self.factor * self.size[0]/2+10, 0, self.factor * self.size[0]/2+10, self.factor * self.size[1], fill="white", dash=(4, 4))
        
        #Player1 - Bat
        self.c_bat1 = self.court.create_line(
            10,
            50,
            10,
            70,  
            fill="white",width=8)
        #Player2 - Bat
        self.c_bat2 = self.court.create_line(
            self.factor * self.size[0]+10,
            70,
            self.factor * self.size[0]+10,
            90,  
            fill="white",width=8)
        
        r = 5
        self.c_ball = self.court.create_oval(100-r + 10, 100-r, 100+r+ 10, 100+r,fill="white")
        
        self.c_direction = self.court.create_line( 100, 100, 
                                110, 110, fill="white",arrowshape=(8,10,3),arrow="last")

        self.p1points = self.court.create_text(self.factor * self.size[0] / 2  + 10 + 150 ,20,text = str(self.points[1]),fill='white',font = ('ARIAL',30))
        self.p0points = self.court.create_text(self.factor * self.size[0] / 2  - 10 - 150 ,20,text = str(self.points[0]),fill='white',font = ('ARIAL',30))

        root.after(1000,self.updateCurt)
        

if __name__ == '__main__':
    conn, communication_conn = Pipe()
    nwc = Process(target=netw_communication, args=(communication_conn,))
    nwc.start()
    
    
    root = Tk()
    app = Application(conn, master=root)
    
    app.mainloop()
    root.destroy()
    
    nwc.join()
