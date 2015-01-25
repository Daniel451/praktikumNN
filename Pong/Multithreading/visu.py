from tkinter import *
from multiprocessing import Process, Pipe
import socket
import json
from data_frame import DataFrame
import time

def netw_communication(conn):
    nc = NetwCon()
    connected = False
    while not connected:
        print('Try to connect...')
        connected = nc.connect()
        if not connected:
            time.sleep(1)
    print('connected, lets go...')

    
    
    
    while True:
        if conn.poll(None):  # warte auf neue Daten...
            
            frame = conn.recv() # Daten sind da...
            print(frame.instruction)
            nc.send({'instruction' : frame.instruction})

            #if  frame.instruction == 'INIT' or frame.instruction == 'REFRESH':
            retval = nc.receive()
            print(retval)

            conn.send(retval)
            
            if frame.instruction == 'EXIT':
                break
    nc.close()
    conn.close()

class NetwCon:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host='localhost', port=6769):
        try:
            self.sock.connect((host, port))
            return True
        except socket.error as e:
            print('something\'s wrong with %s:%d. Exception type is %s' % (host, port, e))
            print('retrying in 1s')
            #sys.exit(1)
        return False
            
        
    def send(self, data):
        try:
            self.sock.send(bytes(json.dumps(data), 'UTF-8'))
        except socket.error as e:
            print ("Error sending data: %s" % e)
            sys.exit(1)
            
    def receive(self):
        try:
            result = json.loads(self.sock.recv(8*1024).decode('UTF-8'))
        except socket.error as e:
            print ("Error receiving data: %s" % e)
            sys.exit(1)
        return result
        
    def close(self):
        self.sock.close()


class Application(Frame):
    
    def QGoS(self):
        print("Send EXIT Signal to Server...")
        conn.send(DataFrame('EXIT'))


    def save(self):
        print("Send Save Signal to Server...")
        conn.send(DataFrame('saveConfig'))

    def Togglespeed(self):
        print('send Speed Toggle')
        conn.send(DataFrame('CHSPEED'))
        if conn.poll(None):  # warte auf neue Daten...
            print(conn.recv()) # Daten sind da...


    def createWidgets(self):

        self.bSAVE = Button(self)
        self.bSAVE["text"] = "Save Config"
        self.bSAVE["command"] =  self.save
        self.bSAVE.pack({"side": "left"})

        self.bQUIT = Button(self)
        self.bQUIT["text"] = "QUIT client only"
        self.bQUIT["command"] =  self.quit
        self.bQUIT.pack({"side": "left"})
        
        self.bQGoS = Button(self)
        self.bQGoS["text"] = "QUIT game on server",
        self.bQGoS["command"] = self.QGoS
        self.bQGoS.pack({"side": "left"})

        self.togglespeed = Button(self)
        self.togglespeed["text"] = "Toggle Speed",
        self.togglespeed["command"] = self.Togglespeed
        self.togglespeed.pack({"side": "left"})


        
        self.court = Canvas(self.master, width=1000, height=600)
        self.court.pack({"side": "left"})
        
        
    
    def __init__(self, conn, master=None):
        Frame.__init__(self, master)
        self.init = True
        self.pack()
        self.conn = conn
        self.createWidgets()
        self.factor = 50.0
        self.points = [0,0]
        self.update()
        
    def update(self):
        if self.init:
            conn.send(DataFrame('INIT')) #Frage init Daten an...
            
            if conn.poll(None):  # warte auf neue Daten...
                data = conn.recv() # Daten sind da...
                
                self.size = data['size']
                self.batsize = data['batsize']
                self.p1name = data['p1name']
                self.p2name = data['p2name']
                self.timestamp = time.time()
                
                self.drawCurt()
                self.init = False
                
        conn.send(DataFrame('REFRESH')) #Frage init Daten an...
        if conn.poll(None):  # warte auf neue Daten...
            data = conn.recv() # Daten sind da...
            
            self.speed = data['speed']
            self.posvec = data['posvec']
            self.dirvec = data['dirvec']
            self.posX = data['sensor_posX']
            self.posY = data['sensor_posY']
            self.bat = data['bat']
            self.points = data['points']
            
            self.updateCurt()
            
        self.lastrefresh = self.timestamp
        self.timestamp = time.time()
        self.after(300, self.update)
            
    def updateCurt(self):
        r = 5
        self.court.coords(self.c_bat1,  10, (self.factor * self.bat[0]) + (self.factor * self.batsize),10,(self.factor * self.bat[0]) - (self.factor * self.batsize) )
        self.court.coords(self.c_bat2,  self.factor * self.size[0]+10, (self.factor * self.bat[1]) + (self.factor * self.batsize),self.factor * self.size[0]+10, 
            (self.factor * self.bat[1]) - (self.factor * self.batsize) )
                                        
        self.court.coords(self.c_ball,  self.posvec[0]*self.factor-r + 10, self.posvec[1]*self.factor-r, self.posvec[0]*self.factor+r+ 10, self.posvec[1]*self.factor+r )
                                        
                                        
        self.court.coords(self.c_direction, self.posvec[0]*self.factor+ 10, self.posvec[1]*self.factor, self.posvec[0]*self.factor + self.dirvec[0] * self.factor + 10, self.posvec[1]*self.factor + self.dirvec[1] * self.factor)

        self.court.itemconfig(self.p0points,text=str(self.points[0]))
        self.court.itemconfig(self.p1points,text=str(self.points[1]))


        self.court.update()
        
        
    def drawCurt(self):
        
        # court
        self.court.create_rectangle(10,0, self.factor * self.size[0]+10,self.factor * self.size[1],fill="black")
        
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
            fill="white",width=10)
        
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
