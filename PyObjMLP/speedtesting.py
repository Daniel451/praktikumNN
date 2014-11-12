from speedtest import Speedtest as sp 

import numpy as np
import math

timer = sp()

xs = [1,2,3,4,5]
a = np.array(xs)
b = np.array([1])

iterations = 1000000

def myfunc(x):
    math.sin(x)

nmyfunc = np.vectorize(myfunc)

def myfunc2(x):
    for i in range(0,iterations):
        sin(x)

nmyfunc2 = np.vectorize(myfunc2)

timer.record("create 10000 np arrays")

for i in range(0, iterations):
    np.sin(a)

timer.record("rdy creating")

timer.record("conventional adding")

for i in range(0, iterations):
    [math.sin(x) for x in xs]

timer.record("adding rdy")

for i in range(0, iterations):
    map(math.sin, xs)

#timer.record("3")
#
#for i in range(0, iterations):
#    map(np.sin, a)
#
#timer.record("4")
#
#for i in range(0, iterations):
#    nmyfunc(a)

timer.record("5")

nmyfunc2

timer.record("6")

timer.printRecords()
