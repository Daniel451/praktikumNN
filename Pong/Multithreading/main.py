#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
""" bla bla bla"""   #TODO: Write summary about this file!

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

from multiprocessing import Process, Pipe
from knnframe import knnframe
from court import Court
import logging

import sys
import threading
import os.path

import json
import socketserver
import time
import datetime


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
                                                            'size':Court.v_getSize(),
                                                            'batsize':Court.v_getBatSize(),
                                                            'p1name':'John A. Nunez',
                                                            'p2name':'Cynthia J. Wilson',
                                                        }), 'UTF-8'))
                elif instruction == 'REFRESH':
                    self.request.sendall(bytes(json.dumps({
                                                            'return':'ok',
                                                            'speed':Court.v_getSpeed(),
                                                            'posvec':Court.v_getPosVec().tolist(), #echte position
                                                            'dirvec':Court.v_getDirVec().tolist(), # Richtungsvector
                                                            'bat':Court.v_getbat(),
                                                            'points':Court.v_getPoint(),
                                                            'sensorP1_bat':Court.scaled_sensor_bat(0),   # position wie sie das NN sieht!
                                                            'sensorP2_bat':Court.scaled_sensor_bat(1),
                                                            'sensor_posX':Court.scaled_sensor_x(),   # position wie sie das NN sieht!
                                                            'sensor_posY':Court.scaled_sensor_y(),
                                                        }), 'UTF-8'))
                elif instruction == 'saveConfig':
                    x = DataFrame('saveConfig')
                    connPlayer0.send(x)
                    connPlayer1.send(x)


                elif instruction == 'CHSPEED':
                    self.request.sendall(bytes(json.dumps({'return':'ok'}), 'UTF-8'))
                    print ('change speed to...')
                    changespeed()





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
                #print('Player ' + str(playername) + ' Call: ' +  frame.instruction )
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

                path = 'save/config_' + str(playername) + '_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') + '.pcf'
                #print('Player ' + str(playername) + ' Call: '+ frame.instruction + ' in ' + path )
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


def changespeed():
    global speed
    if speed > 0.0:
        speed = 0.0
    else:
        speed = 0.3



if __name__ == '__main__':

    # logging path
    path = "log_pong.log"
    global speed
    speed = 0.0

    # check if logfile exists
    if not os.path.exists(path):
        file = open(path, "w+")
        file.close()

    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info('Started')
    saveconfig = False
    court = Court()
    
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

    x = DataFrame('saveConfig')
    connPlayer0.send(x)
    connPlayer1.send(x)
    x = input('sdf')

    while True:

        #print('new tick!')
        Court.tick()
        rewardframepos = DataFrame('reward_pos')

        if Court.hitbat(0):
            rewardframepos.add('err', Court.scaled_sensor_err(0) )
            connPlayer0.send(rewardframepos)
        if Court.hitbat(1):
            rewardframepos.add('err', Court.scaled_sensor_err(1) )
            connPlayer1.send(rewardframepos)


        if Court.out(0):
            rewardframeneg = DataFrame('reward_neg')
            rewardframeneg.add('err', Court.scaled_sensor_err(0) )
            connPlayer0.send(rewardframeneg)
        if Court.out(1):
            rewardframeneg = DataFrame('reward_neg')
            rewardframeneg.add('err', Court.scaled_sensor_err(1) )
            connPlayer1.send(rewardframeneg)


        prednextreqPlayer0 = DataFrame('predictNext')
        prednextreqPlayer0.add('xpos',Court.scaled_sensor_x())
        prednextreqPlayer0.add('ypos',Court.scaled_sensor_y())
        prednextreqPlayer0.add('mypos',Court.scaled_sensor_bat(0))
        connPlayer0.send(prednextreqPlayer0)
        #print('send data to player 0: ' + str(prednextreqPlayer0))
        #print('Player0 Bat: ' + str(court.scaled_sensor_bat(0)))
        #print('Player0 X: ' + str(court.scaled_sensor_x()))
        #print('Player0 Y: ' + str(court.scaled_sensor_y()))

        prednextreqPlayer1 = DataFrame('predictNext')
        prednextreqPlayer1.add('xpos',Court.scaled_sensor_x())
        prednextreqPlayer1.add('ypos',Court.scaled_sensor_y())
        prednextreqPlayer1.add('mypos',Court.scaled_sensor_bat(1))
        connPlayer1.send(prednextreqPlayer1)
        #print('send data to player 1: ' + str(prednextreqPlayer1))
        #print('Player1 Bat: ' + str(court.scaled_sensor_bat(1)))


        #while True:
        if speed > 0:
            time.sleep(speed)

        if connPlayer0.poll(None): # Daten sind da...
            frame = connPlayer0.recv()
            if frame.instruction == 'Return':
                Court.move(0,frame.getdata('move'))



        if connPlayer1.poll(None): # Daten sind da...
            frame = connPlayer1.recv()
            if frame.instruction == 'Return':
                Court.move(1,frame.getdata('move'))






            
        
    

    frame = DataFrame('EXIT')
    connPlayer0.send(frame)
    connPlayer1.send(frame)
                
                  
    p0.join()
    p1.join()
    server.shutdown()
    logging.info('Finished')
    
