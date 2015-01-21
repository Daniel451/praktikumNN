from multiprocessing import Process, Pipe
from knnframe import knnframe
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
                                                            'sensorP1_bat':court.scaled_sensor_bat(0),   # position wie sie das NN sieht!
                                                            'sensorP2_bat':court.scaled_sensor_bat(1),
                                                            'sensor_posX':court.scaled_sensor_x(),   # position wie sie das NN sieht!
                                                            'sensor_posY':court.scaled_sensor_y(),
                                                        }), 'UTF-8'))
                elif instruction == 'saveConfig':
                    saveconfig = True



                else:
                    self.request.sendall(bytes(json.dumps({'return':'not ok'}), 'UTF-8'))
            except Exception as e:
                print("No instruction available: ", e)
                return





def startplayer(conn,playername, loadconfig = None):
    player = knnframe(loadconfig,playername) #Erstelle ein Neuronal Netzwerk als Spieler evtl lade eine Konfigration... Wenn nicht, erstelle eine Neue
    # init ist hier nun passier!

    
    while True:
        #print('Player ' + str(playername) + ' waiting for new instruction...' )
        if conn.poll(None):  # warte auf neue Daten...
            frame = conn.recv() # Daten sind da...
            
            if frame.instruction == 'EXIT': # Beenden.
                print('Player ' + str(playername) + ' Call: ' +  frame.instruction )
                exit()
                break

            elif frame.instruction == 'predictNext': #
                #print('Player ' + str(playername) + ' Call: ' + frame.instruction )
                conn.send(predictnext(player,frame))

            elif frame.instruction == 'reward_pos': #
                #print('Player ' + str(playername) + ' Call: ' + frame.instruction )
                err = float(frame.getdata('err'))
                player.reward_pos(err)

            elif frame.instruction == 'reward_neg': #
                #print('Player ' + str(playername) + ' Call: ' + frame.instruction )
                err = float(frame.getdata('err'))
                player.reward_neg(err)

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

    action = player.predict(frame.getdata('xpos'), frame.getdata('ypos'), frame.getdata('mypos'))

    returnframe = DataFrame('Return')
    returnframe.add('move',action)

    return returnframe


if __name__ == '__main__':

    # logging path
    path = "log_pong.log"

    # check if logfile exists
    if not os.path.exists(path):
        file = open(path, "w+")
        file.close()

    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info('Started')
    saveconfig = False
    court = court()
    
    sendPlayerA, connPlayer0 = Pipe()
    sendPlayerB, connPlayer1 = Pipe()
    p0 = Process(target=startplayer, args=(sendPlayerA,0))
    p1 = Process(target=startplayer, args=(sendPlayerB,1))
    p0.start()
    p1.start()
    
    
    
    
    HOST, PORT = "localhost", 6769

    server = ThreadedTCPServer((HOST, PORT), MyTCPServerHandler)
    ip, port = server.server_address

    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print ("Server loop running in thread: ", server_thread.name)

    while True:

        #print('new tick!')
        court.tick()
        rewardframepos = DataFrame('reward_pos')

        if court.hitbat(0):
            rewardframepos.add('err', court.scaled_sensor_err(0) )
            connPlayer0.send(rewardframepos)
        if court.hitbat(1):
            rewardframepos.add('err', court.scaled_sensor_err(1) )
            connPlayer1.send(rewardframepos)


        if court.out(0):
            rewardframeneg = DataFrame('reward_neg')
            rewardframeneg.add('err', court.scaled_sensor_err(0) )
            connPlayer0.send(rewardframeneg)
        if court.out(1):
            rewardframeneg = DataFrame('reward_neg')
            rewardframeneg.add('err', court.scaled_sensor_err(1) )
            connPlayer1.send(rewardframeneg)


        prednextreqPlayer0 = DataFrame('predictNext')
        prednextreqPlayer0.add('xpos',court.scaled_sensor_x())
        prednextreqPlayer0.add('ypos',court.scaled_sensor_y())
        prednextreqPlayer0.add('mypos',court.scaled_sensor_bat(0))
        connPlayer0.send(prednextreqPlayer0)
        #print('send data to player 0: ' + str(prednextreqPlayer0))
        #print('Player0 Bat: ' + str(court.scaled_sensor_bat(0)))
        #print('Player0 X: ' + str(court.scaled_sensor_x()))
        #print('Player0 Y: ' + str(court.scaled_sensor_y()))

        prednextreqPlayer1 = DataFrame('predictNext')
        prednextreqPlayer1.add('xpos',court.scaled_sensor_x())
        prednextreqPlayer1.add('ypos',court.scaled_sensor_y())
        prednextreqPlayer1.add('mypos',court.scaled_sensor_bat(1))
        connPlayer1.send(prednextreqPlayer1)
        #print('send data to player 1: ' + str(prednextreqPlayer1))
        #print('Player1 Bat: ' + str(court.scaled_sensor_bat(1)))


        #while True:

        time.sleep(0.1) # muss warten!!

        if connPlayer0.poll(None): # Daten sind da...
            frame = connPlayer0.recv()
            if frame.instruction == 'Return':
                court.move(0,frame.getdata('move'))



        if connPlayer1.poll(None): # Daten sind da...
            frame = connPlayer1.recv()
            if frame.instruction == 'Return':
                court.move(1,frame.getdata('move'))


        if saveconfig:
            x = DataFrame('saveConfig')
            connPlayer0.send(x)
            connPlayer1.send(x)



            
        
    

    frame = DataFrame('EXIT')
    connPlayer0.send(frame)
    connPlayer1.send(frame)
                
                  
    p0.join()
    p1.join()
    server.shutdown()
    logging.info('Finished')
    
