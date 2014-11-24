#!/usr/bin/env python3.4
#-*- coding: utf-8 -*-

import numpy as n
from threading import Thread
import time

class action:
    global w
    def __init__(self, x,y):
        self.w = n.random.random((x, y))
        
    def getQ(self):
        return self.w

class World:
    global world
    global up
    global down
    global right
    global left
    
    def __init__(self, x,y):
        self.world = n.zeros((x, y))
        self.up = action(x,y)
        self.down = action(x,y)
        self.right = action(x,y)
        self.left = action(x,y)
        visualisation = Thread(target=visu, args=('BLA',self.up) )
        visualisation.start()

    def setGoal(self, x = -1,y = -1):
        if x == -1 or y == -1:
             x = n.random.randint(0,self.world.shape[0])
             y = n.random.randint(0,self.world.shape[1])
        self.world[x,y] = 1
        
    def debugPrint(self):
        print(self.world)
        
    def visu(threadname,up):
        time.sleep(1)
        print(up.getQ())
        
        


print('start')
x = World(3,3)
x.setGoal()
time.sleep(6)
print('ende')

