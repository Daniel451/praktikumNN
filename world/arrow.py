import matplotlib.pyplot as plt
import numpy

up = 0.4
down = 0.6
left = 0.3
right = 0.2

y = 6
x = 4


vektor = numpy.array([up - down, left - right])
point = numpy.array([x, y])

start = point + vektor
end = point - vektor

print('start: ' + str(start))
print('end: ' + str(end))

ax = plt.axes()
ax.set_ylim([0,10])
ax.set_xlim([0,12])
ax.arrow(start[0], start[1],vektor[0], vektor[1])
#ax.arrow(end[0], end[1],1, 1)

plt.show()