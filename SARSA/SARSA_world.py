#-*- coding: utf-8 -*-

# imports
import random
import numpy




class SARSAworld:
    def __init__(self, size_x, size_y):
        self.x = random.randint(0, size_x-1)
        self.y = random.randint(0, size_y-1)
        self.size_x = size_x
        self.size_y = size_y

        self.newRandomStartPosition()


    def newRandomStartPosition(self):
        """
        calculates a new random start position (x,y) for our robot
        """
        self.x = random.randint(0, self.size_x-1)
        self.y = random.randint(0, self.size_y-1)
        
    def doAction(self, action):

        # position world redoAction
        if  action == 0:      #down
            self.x -= 1
        elif action == 1:     #right
            self.y += 1
        elif action == 2:     #up
            self.x += 1
        elif action == 3:     #left
            self.y -= 1
        else:
            print('unknown action')
            # borders remain, movement stops 

        if self.x < 0:
            self.x = 0
        elif self.x >= self.size_x:
            self.x = self.size_x - 1

        if self.y < 0:
            self.y = 0
        elif self.y >= self.size_y:
            self.y = self.size_y - 1


    def get_reward(self):
        if self.x == 2 and self.y == 3:
            return 1.0
        else:
            return 0.0
            

    def get_sensor(self):
        p = numpy.zeros(self.size_x * self.size_y)
        p[self.x * self.size_x + self.y] = 1.0
        return p




