from matplotlib.pylab import *
import numpy


for i in range(0,20):
    plt.pcolor(numpy.random.random((5,5)), cmap=cm.gray)
    plt.draw()
    plt.pause(0.1)

plt.show()
