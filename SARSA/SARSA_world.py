#-*- coding: utf-8 -*-

# imports
import random
import numpy
from numpy.random import rand
from numpy.random import random_integers as randInteger




class SARSAworld:
    def __init__(self, size_x, size_y):
        self.x = random.randint(0, size_x-1)
        self.y = random.randint(0, size_y-1)
        self.size_x = size_x
        self.size_y = size_y
        
        self.newRandomStartPosition()
        self.goal_x = self.x
        self.goal_y = self.y

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
        if self.x == self.goal_x and self.y == self.goal_y:
            return 1.0
        else:
            return 0.0
            
    def getMap(self):
        Map = numpy.zeros((self.size_x , self.size_y))
        Map[self.goal_x][self.goal_y] = 0.5
        return Map

    def get_sensor(self):
        p = numpy.zeros(self.size_x * self.size_y)
        p[self.x * self.size_x + self.y] = 1.0
        return p


#Alternative Implementation

class SARSAmaze():
    def __init__(self, size_x, size_y):
        self.x = random.randint(0, size_x-1)
        self.y = random.randint(0, size_y-1)
        self.size_x = size_x
        self.size_y = size_y
        self.mazemap = self.createmaze(size_x,size_y)
        self.newRandomStartPosition()
        self.goal_x = self.x
        self.goal_y = self.y
        self.newRandomStartPosition()
        print(self.mazemap.shape)
        print(self.mazemap[0,0])
        print('Goal_x: ' + str( self.goal_x))
        print('Goal_y: ' + str( self.goal_y))
        print(self.mazemap[self.goal_x,self.goal_y])

    def createmaze(self,width, height, complexity=.75, density=0.25):
        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * (shape[0] // 2 * shape[1] // 2))
        # Build actual maze
        Z = numpy.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = randInteger(0, shape[1] // 2) * 2, randInteger(0, shape[0] // 2) * 2
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[randInteger(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return Z

    def newRandomStartPosition(self):
        """
        calculates a new random start position (x,y) for our robot
        """
        while True:
            self.x = random.randint(0, self.size_x-1)
            self.y = random.randint(0, self.size_y-1)
            if self.mazemap[self.x,self.y] == False:
                break
    
    def getMap(self):
        mazemap = (numpy.copy(self.mazemap)).astype(float)
        mazemap[self.goal_x][self.goal_y] = 0.5
        return mazemap
    
    def doAction(self, action):
        if self.x < 0:
            self.x = 0
        elif self.x >= self.size_x:
            self.x = self.size_x - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= self.size_y:
            self.y = self.size_y - 1
        
        
        if action == 0 and self.mazemap[self.x-1,self.y] == False and self.x > 0:      #left
            self.x -= 1
        elif action == 1 and self.mazemap[self.x,self.y+1] == False and self.y < self.size_y:     #up
            self.y += 1
        elif action == 2 and self.mazemap[self.x+1,self.y] == False and self.x < self.size_x:     #right
            self.x += 1
        elif action == 3 and self.mazemap[self.x,self.y-1] == False and self.y > 0:     #down
            self.y -= 1
        #print('Point: x=' + str(self.x) + ' y= ' + str(self.y))
        
    def get_reward(self):
        if  self.goal_x == self.x and self.goal_y == self.y:
            return 1.0
        else:
            return 0.0
            

    def get_sensor(self):
        p = numpy.zeros(self.size_x * self.size_y)
        p[self.x * self.size_x + self.y] = 1.0
        return p


