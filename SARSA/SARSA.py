


import random
import time
import numpy
import matplotlib.pyplot as plt 
from numpy.random import rand


class world:


    def __init__(self, size_x, size_y):
        """
        :param size_x: column size of the world
        :param size_y: row size of the world
        """

        self.size_x = size_x
        self.size_y = size_y

        self.newRandomStartPosition()

        self.ind = numpy.arange(0, self.size_x * self.size_y)


    def newRandomStartPosition(self):
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
                # Ränder von Welt bleiben Ränder! Bewegung bleit stehen!
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


def plot(init):
    Z = rand(6,10)

    plt.subplot(3,3,2)
    c = plt.pcolor(Z)
    plt.title('Weight up')
    
    
    plt.subplot(3,3,8)
    c = plt.pcolor(Z)
    plt.title('Weight down')
    
    plt.subplot(3,3,4)
    c = plt.pcolor(Z)
    plt.title('Weight left')
    
    plt.subplot(3,3,6)
    c = plt.pcolor(Z)
    plt.title('Weight right')
    
    plt.subplot(3,3,5)
    c = plt.pcolor(Z)
    plt.title('Best Direction')
    
    if init:
        plt.draw()
        plt.pause(1)
    else:
        plt.draw()
    #plt.pause(1)
    #print(w)
    
    return True
    



size_x, size_y = 4, 6
size_map = size_x * size_y
size_mot = 4
w = numpy.random.uniform (0.0, 0.0, (size_mot, size_map))
world = world(size_x, size_y)

beta = 50.0

#use the multiprocessing module to perform the plotting doActionivity in another process (i.e., on another core):
#job_for_another_core = multiprocessing.Process(target=plot,args=())
#job_for_another_core.start()

plot(True)

for iter in range (1000):

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
        
        wDotSensorVal = numpy.dot (w[doAction_tic], SensorVal)
        
        if  r == 1.0:  # This is cleaner than defining
            target = r                                  # target as r + 0.9 * wDotSensorVal,
        else:                                           # because weights now converge.
            target = 0.9 * wDotSensorVal                      # gamma = 0.9
        delta = target - val                            # prediction error
        
        w += 0.5 * delta * numpy.outer (doAction_vec, I)
        
        I[0:size_map] = SensorVal[0:size_map]
        val = wDotSensorVal
        doAction = doAction_tic
    

    print('---------------------------------------')
    print(w)
    plot(False)
    
exit = True

