#-*- coding: utf-8 -*-



from SARSA_world import SARSAworld as SW
from SARSA_action import SARSAaction as SA
from SARSA_plot import SARSAplot as SP
from SARSA_loop import SARSAloop as SL

import numpy



# size of the gridworld
size_x = 10 
size_y = 10

# mapsize
size_map = size_x * size_y
size_motion = 4

w = numpy.random.uniform (0.0, 0.0, (size_motion, size_map))
beta = 5.0


# initialize everything
world = SW(size_x, size_y)
action = SA()
plot = SP()

plot.plot(w, size_x, size_y)
plot.plotPause(1)






print('Done')
plot.plot(w, size_x, size_y)
plot.plotShow()
    


