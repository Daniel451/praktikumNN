#-*- coding: utf-8 -*-


import random
import time
import numpy
import matplotlib.pyplot as plt 
import matplotlib.pylab 
from numpy.random import rand
from numpy.random import random_integers as randInteger


class maze():
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
            self.x = random.randint(0, size_x-1)
            self.y = random.randint(0, size_y-1)
            if self.mazemap[self.x,self.y] == False:
                break
    
    def getMaze(self):
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


def nextAction (S_from, beta):
    sum = 0.0
    p_i = 0.0
    rnd = numpy.random.random()
    d_r = len (S_from)
    sel = 0

    for i in range (d_r):
        sum += numpy.exp (beta * S_from[i])

    S_target = numpy.zeros (d_r)

    for i in range (d_r):
        p_i += numpy.exp (beta * S_from[i]) / sum

        if  p_i > rnd:
            sel = i
            rnd = 1.1 # out of reach, so the next will not be turned ON

    return sel


def plot(state):

    plt.subplot(3,3,2)
    c = plt.pcolor(w[0].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0,size_x,0,size_y])
    plt.title('Weight up')
    
    plt.subplot(3,3,8)
    d = plt.pcolor(w[1].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0,size_x,0,size_y])
    plt.title('Weight down')
    
    plt.subplot(3,3,4)
    e = plt.pcolor(w[2].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0,size_x,0,size_y])
    plt.title('Weight left')
    
    plt.subplot(3,3,6)
    f = plt.pcolor(w[3].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0,size_x,0,size_y])
    plt.title('Weight right')
    
    plt.subplot(3,3,5)
    f = plt.pcolor(world.getMaze(),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0,size_x,0,size_y])
    plt.title('Maze')


    #for _x in range (0,size_x):
    #    for _y in range (0,size_y):
    #        vektor = numpy.array([w[0][_x] - w[2][_x], w[1][_y] - w[3][_y] ])
    #        point = numpy.array([_x, _y])
    #        start = point + vektor
    #        g.arrow(start[0], start[1],vektor[0], vektor[1])

    plt.draw()
    plt.pause(0.01)


    


size_x = 31 
size_y = 31
size_map = size_x * size_y
size_mot = 4
w = numpy.random.uniform (0.0, 0.0, (size_mot, size_map))
world = maze(size_x, size_y)

beta = 5.0

#use the multiprocessing module to perform the plotting doActionivity in another process (i.e., on another core):
#job_for_another_core = multiprocessing.Process(target=plot,args=())
#job_for_another_core.start()

plot(0)
plt.pause(1)

for iter in range (100000):

    world.newRandomStartPosition()
    I = world.get_sensor()
    h = numpy.dot (w, I)
    doAction = nextAction (h, beta)    # nächste action bestimmen...
    doAction_vec = numpy.zeros (size_mot)
    doAction_vec[doAction] = 1.0
    val = numpy.dot (w[doAction], I)      # Sichere W von vor der nächsten Action 
    r = 0
    duration = 0
    
    while r == 0:
        
        duration += 1
        
        world.doAction(doAction)
        r = world.get_reward()
        SensorVal = world.get_sensor()
        
        h = numpy.dot (w, SensorVal)
        doAction_tic = nextAction (h, beta)
        
        doAction_vec = numpy.zeros (size_mot)
        doAction_vec[doAction] = 1.0
        
        wDotSensorVal = numpy.dot(w[doAction_tic], SensorVal)
        
        if  r == 1.0:  # This is cleaner than defining
            target = r                                  # target as r + 0.9 * wDotSensorVal,
        else:                                           # because weights now converge.
            target = 0.9 * wDotSensorVal                      # gamma = 0.9
        delta = target - val                            # prediction error
        
        w += 0.3 * (target - val) * numpy.outer(doAction_vec, I)
        
        I[0:size_map] = SensorVal[0:size_map]
        val = wDotSensorVal
        doAction = doAction_tic
        #if duration > 100000:
            #break
    
    print('------------- Needed hops: ' + str(duration) + '-------------')
    if iter%10 == 0:
        plot(1)
print('Done')
plot(3)
plt.show()
    


