__author__ = 'daniel'


from matplotlib.pylab import *

import numpy



arr = numpy.random.random((10, 10))

# Display a random matrix with a specified figure number and a grayscale
# colormap
#plt.matshow(arr, fignum=0, cmap=cm.gray)

#plt.ylabel("rows")
#plt.xlabel("columns")

#plt.colorbar()


im1 = plt.subplot(2,1,1)
im1.imshow(arr, cmap=cm.gray, interpolation="nearest")

im2 = plt.subplot(2,1,2)
im2.imshow(arr, cmap=cm.gist_heat, interpolation="nearest")

plt.ylabel("rows")
plt.xlabel("columns")




plt.show()