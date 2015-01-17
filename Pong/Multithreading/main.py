from multiprocessing import Process, Pipe
from knn_frame import mlp
from court import court
import logging

import sys
import threading
import os.path

import json
import socketserver
import time


from data_frame import DataFrame

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





def startplayer(conn,playername, loadconfig = None):
    player = mlp(loadconfig,playername) #Erstelle ein Neuronal Netzwerk als Spieler evtl lade eine Konfigration... Wenn nicht, erstelle eine Neue
    # init ist hier nun passier!

    
    while True:
        print('Player ' + str(playername) + ' waiting for new instruction...' )
        if conn.poll(None):  # warte auf neue Daten...
            frame = conn.recv() # Daten sind da...
            
            if frame.instruction == 'EXIT': # Beenden.
                print('Player ' + str(playername) + ' Call: ' +  frame.instruction )
                break
            elif frame.instruction == 'predictNext': #
                print('Player ' + str(playername) + ' Call: ' + frame.instruction )
                conn.send(predictnext(player,frame))
            elif frame.instruction == 'reward_pos': #
                print('Player ' + str(playername) + ' Call: ' + frame.instruction )
                reward(player,frame)
            elif frame.instruction == 'saveConfig': #
                path = 'save/config_' + str(playername) + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.time()) + '.pcf'
                print('Player ' + str(playername) + ' Call: '+ frame.instruction + ' in ' + path )
                player.saveconfig(path)
            else:
                print('Player ' + str(playername) + ' unknown instruction: ' + frame.instruction)
    conn.close()

def saveconfig(player,frame):
    player.saveconfig(frame.getdata('filename'))
    
def predictnext(player,frame):


    #action = player.predict(frame.getdata('xpos'), frame.getdata('ypos'), frame.getdata('mypos'))


    action = 'u'
    returnframe = DataFrame('Return')
    returnframe.add('move',action)
    return returnframe
    
def reward(player,frame):
    if frame.getdata('') == 'positiv':
        player.reward_pos(frame.getdata('error'))
    else:
        player.reward_neg(frame.getdata('error'))




if __name__ == '__main__':

    # logging path
    path = "log_pong.log"

    # check if logfile exists
    if not os.path.exists(path):
        file = open(path, "w+")
        file.close()

    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info('Started')
    
    court = court()
    
    sendPlayerA, connPlayer0 = Pipe()
    sendPlayerB, connPlayer1 = Pipe()
    p1 = Process(target=startplayer, args=(sendPlayerA,0))
    p2 = Process(target=startplayer, args=(sendPlayerB,1))
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

        print('new tick!')
        court.tick()
        if court.hitbat(0):
            frame = DataFrame('reward_pos')
            frame.add('error',0)
            connPlayer0.send(frame)
        if court.hitbat(1):
            frame = DataFrame('reward_pos')
            frame.add('error',1)
            connPlayer1.send(frame)

        print('send data to player 0!')
        frame = DataFrame('predictNext')
        frame.add('xpos',court.sensor_X())
        frame.add('ypos',court.sensor_Y())
        frame.add('mypos',court.sensor_bat(0))
        connPlayer0.send(frame)

        print('send data to player 1!')
        frame = DataFrame('predictNext')
        frame.add('xpos',court.sensor_X())
        frame.add('ypos',court.sensor_Y())
        frame.add('mypos',court.sensor_bat(1))
        connPlayer1.send(frame)


        #while True:

        print('wait for player answers!')
        time.sleep(0.3) # muss warten!!
            #if not (connPlayer0.empty() or connPlayer1.empty()):
        if connPlayer0.poll(None): # Daten sind da...
            frame = connPlayer0.recv()
            if frame.instruction == 'Return':
                court.move(0,frame.getdata('move'))



        if connPlayer1.poll(None): # Daten sind da...
            frame = connPlayer1.recv()
            if frame.instruction == 'Return':
                court.move(1,frame.getdata('move'))

            
        
    
    
    #while True:
    #    x = input("Do what?: ")
    #    if x == 'EXIT'
    frame = DataFrame('EXIT')
    connPlayer0.send(frame)
    connPlayer1.send(frame)
    #        break
                
                  
    p1.join()
    p2.join()
    server.shutdown()
    logging.info('Finished')
    