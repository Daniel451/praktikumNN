#-*- coding: utf-8 -*-


import random
import time
import numpy
import matplotlib.pyplot as plt 
import matplotlib.pylab 
from numpy.random import rand



class world:
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
        self.x = random.randint(0, size_x-1)
        self.y = random.randint(0, size_y-1)
        
    def doAction(self, action):

        # position world redoActionion
        if  numpy.random.random_sample() > 0.0:
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

        if   self.x < 0:
            self.x = 0
        elif self.x >= self.size_x:
            self.x = self.size_x - 1

        if   self.y < 0:
            self.y = 0
        elif self.y >= self.size_y:
            self.y = self.size_y - 1


    def get_reward(self):
        if  self.x == 2 and self.y == 3:
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
    Z = rand(4,6)

    plt.subplot(3,3,2)
    c = plt.pcolor(w[0].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0, size_x, 0, size_y])
    plt.title('Weight up')
    
    plt.subplot(3,3,8)
    d = plt.pcolor(w[1].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0, size_x, 0, size_y])
    plt.title('Weight down')
    
    plt.subplot(3,3,4)
    e = plt.pcolor(w[2].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0, size_x, 0, size_y])
    plt.title('Weight left')
    
    plt.subplot(3,3,6)
    f = plt.pcolor(w[3].reshape(size_x,size_y),cmap=plt.get_cmap('RdYlGn'))
    plt.axis([0, size_x, 0, size_y])
    plt.title('Weight right')
    
    #plt.subplot(3,3,5)
    #g = plt.axes()
    #g.set_ylim([0,size_y])
    #g.set_xlim([0,size_x])
    #plt.title('Best Direction')


    #for _x in range (0,size_x):
    #    for _y in range (0,size_y):
    #        vektor = numpy.array([w[0][_x] - w[2][_x], w[1][_y] - w[3][_y] ])
    #        point = numpy.array([_x, _y])
    #        start = point + vektor
    #        g.arrow(start[0], start[1],vektor[0], vektor[1])

    plt.draw()
    plt.pause(0.01)


    


size_x = 10 
size_y = 10
size_map = size_x * size_y
size_mot = 4
w = numpy.random.uniform (0.0, 0.0, (size_mot, size_map))
world = world(size_x, size_y)

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
    
    print('------------- Needed hops: ' + str(duration) + '-------------')
    if iter%10 == 0:
        plot(1)
print('Done')
plot(3)
plt.show()
    


