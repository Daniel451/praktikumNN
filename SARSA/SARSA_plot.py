#-*- coding: utf-8 -*-

# imports
import time
import matplotlib.pyplot as plt 
import numpy as np



class SARSAplot:

    def __init__(self):
        self.plt = plt


    def plot(self, w, size_x, size_y, world=False):
        Z = np.random.rand(4,6)

        self.plt.subplot(3,3,8)
        c = self.plt.pcolor(w[0].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight Down')
        
        self.plt.subplot(3,3,6)
        d = self.plt.pcolor(w[1].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight Right')
        
        self.plt.subplot(3,3,2)
        e = self.plt.pcolor(w[2].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight Up')
        
        self.plt.subplot(3,3,4)
        f = self.plt.pcolor(w[3].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight Left')
        
        if world != False:
            self.plt.subplot(3,3,5)
            g = plt.pcolor(world.getMap(),cmap=plt.get_cmap('RdYlGn'))
            self.plt.axis([0,size_x,0,size_y])
            self.plt.title('Map')
        
        
        self.plt.draw()
        self.plt.pause(0.01)


    def plotPause(self, i):
        self.plt.pause(i)


    def plotShow(self):
        self.plt.show()



