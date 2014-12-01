#-*- coding: utf-8 -*-

# imports
import time
import matplotlib.pyplot as plt 
import numpy as np



class SARSAplot:

    def __init__(self):
        self.plt = plt


    def plot(self, w, size_x, size_y):
        Z = np.random.rand(4,6)

        self.plt.subplot(3,3,2)
        c = self.plt.pcolor(w[0].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight up')
        
        self.plt.subplot(3,3,8)
        d = self.plt.pcolor(w[1].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight down')
        
        self.plt.subplot(3,3,4)
        e = self.plt.pcolor(w[2].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight left')
        
        self.plt.subplot(3,3,6)
        f = self.plt.pcolor(w[3].reshape(size_x,size_y),cmap=self.plt.get_cmap('RdYlGn'))
        self.plt.axis([0, size_x, 0, size_y])
        self.plt.title('Weight right')
        
        self.plt.draw()
        self.plt.pause(0.01)


    def plotPause(self, i):
        self.plt.pause(i)


    def plotShow(self):
        self.plt.show()



