from multiprocessing import Process, Pipe
from rec_mlp import mlp
from court import court
import logging

import sys
import threading

import json
import socketserver
import time


from DataFrame import DataFrame 

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

class MyTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

class MyTCPServerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                data = json.loads(self.request.recv(8*1024).decode('UTF-8').strip())
                # process the data, i.e. print it:
                #print(json.dumps(data, sort_keys=True,indent=4, separators=(',', ': ')))
            
            except Exception as e:
                print("Exception while receiving message: ", e)
                return
            
            try:
                instruction = data['instruction']
                #print('got instruction: ' + instruction)
            
                if instruction == 'EXIT':
                    # send some 'ok' back
                    print('Exiting...')
                    self.request.sendall(bytes(json.dumps({'return':'ok'}), 'UTF-8'))
                    self.__shutdown_request = True
                    self.__is_shut_down.wait()
                    sys.exit(1)
                    break
                
                elif instruction == 'INIT':
                    self.request.sendall(bytes(json.dumps({
                                                            'return':'ok',
                                                            'size':court.v_getSize(),
                                                            'batsize':court.v_getBatSize(),
                                                            'p1name':'John A. Nunez',
                                                            'p2name':'Cynthia J. Wilson',
                                                        }), 'UTF-8'))
                elif instruction == 'REFRESH':
                    self.request.sendall(bytes(json.dumps({
                                                            'return':'ok',
                                                            'speed':court.v_getSpeed(),
                                                            'posvec':court.v_getPosVec().tolist(), #echte position
                                                            'dirvec':court.v_getDirVec().tolist(), # Richtungsvector
                                                            'bat':court.v_getbat(),
                                                            'points':court.v_getPoint(),
                                                            'sensorP1_bat':court.sensor_bat(0),   # position wie sie das NN sieht!
                                                            'sensorP2_bat':court.sensor_bat(1),
                                                            'sensor_posX':court.sensor_X(),   # position wie sie das NN sieht!
                                                            'sensor_posY':court.sensor_Y(),
                                                        }), 'UTF-8'))
                
                else:
                    self.request.sendall(bytes(json.dumps({'return':'not ok'}), 'UTF-8'))
                    #print('send information: ' + 'NOT OK!')
                print('send information: ' + instruction)
            except Exception as e:
                print("No instruction available: ", e)
                return





def startPlayer(conn,playername, loadConfig = None):
    player = mlp(loadConfig,playername) #Erstelle ein Neuronal Netzwerk als Spieler evtl lade eine Konfigration... Wenn nicht, erstelle eine Neue
    # init ist hier nun passier!
    
    
    while True:
        if conn.poll(None):  # warte auf neue Daten...
            frame = conn.recv() # Daten sind da...
            
            if frame.instruction == 'EXIT': # Beenden.
                break
            elif frame.instruction == 'predictNext': # 
                conn.send(predictNext(player,frame))
            elif frame.instruction == 'reward': # 
                reward(player,frame)
            elif frame.instruction == 'saveConfig': # 
                reward(player,frame)
            else:
                print('unknown instruction: ' + frame.instruction)
    conn.close()

def saveConfig(player,frame):
    player.SaveConfig(frame.getdata('filename'))
    
def predictNext(player,frame):
    action = player.predict(frame.getdata('xpos'), frame.getdata('ypos'), frame.getdata('mypos'))
    returnframe = DataFrame('Return')
    returnframe.add('move',action)
    return returnframe
    
def reward(player,frame):
    player.reward(frame.getdata('error'))
    

if __name__ == '__main__':
    
    logging.basicConfig(filename='log/pong.log', level=logging.DEBUG)
    logging.info('Started')
    
    court = court()
    
    sendPlayerA, connPlayerA = Pipe()
    sendPlayerB, connPlayerB = Pipe()
    p1 = Process(target=startPlayer, args=(sendPlayerA,0))
    p2 = Process(target=startPlayer, args=(sendPlayerB,1))
    p1.start()
    p2.start()
    
    
    
    
    HOST, PORT = "localhost", 6769

    server = ThreadedTCPServer((HOST, PORT), MyTCPServerHandler)
    ip, port = server.server_address

    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print ("Server loop running in thread: ", server_thread.name)
    
    print('Press a Key to exit')
    while True:
        
        court.tic()
        if court.hitBat(0):
            frame = DataFrame('reward')
            frame.add('error',0)
            connPlayerA.send(frame)
        if court.hitBat(1):
            frame = DataFrame('reward')
            frame.add('error',1)
            connPlayerB.send(frame)
            
        frame = DataFrame('predictNext')
        frame.add('xpos',court.sensor_X)
        frame.add('ypos',court.sensor_Y)
        frame.add('mypos',court.sensor_bat(0))
        connPlayerA.send(frame)
        
        frame = DataFrame('predictNext')
        frame.add('xpos',court.sensor_X)
        frame.add('ypos',court.sensor_Y)
        frame.add('mypos',court.sensor_bat(1))
        connPlayerB.send(frame)
        
        time.sleep(0.3)
        #while True:
        #    time.sleep(0.1)
         #   #if connPlayerA.full() and connPlayerA.full():
          #      frame = connPlayerA.recv() # Daten sind da...
           #     if frame.instruction == 'Return':
            #        court.move(frame.getdata('move'))
             #   frame = connPlayerB.recv() # Daten sind da...
              #  if frame.instruction == 'Return':
               #     court.move(frame.getdata('move'))
            
            
        
    
    
    #while True:
    #    x = input("Do what?: ")
    #    if x == 'EXIT'
    frame = DataFrame('EXIT')
    connPlayerA.send(frame)
    connPlayerB.send(frame)
    #        break
                
                  
    p1.join()
    p2.join()
    server.shutdown()
    logging.info('Finished')
    