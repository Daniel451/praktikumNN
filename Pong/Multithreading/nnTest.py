
from NeuralNetwork import NeuralNetwork
from pprint import pprint
import numpy


knn = NeuralNetwork([2,4,3],1)

print('go..')
print('Init')
pprint(knn.debug())
print('..done!')

s_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #Trainingsdaten Input
s_teach = numpy.array([[0], [1], [1], [0]])          #Trainingsdaten Output

a = [0,0,0,0]
sin=numpy.array([[ 0.8 , 0.2 ]])
print(knn.predict(sin))
print(knn.predict(sin))
print(knn.predict(sin))
print(knn.predict(sin))


knn.reward([1.0,0.2,0.3])
print('ende')
exit()


for i in range(0, 2000):
    print(i)
    x = numpy.random.randint(0,4)

    answ=knn.predict(s_in[x])
    err = -1.0*(answ[0] - s_teach[x])

    if answ > 10.0 or answ < -10.0:
        break

    knn.reward(err,0.1)

    a[x] += 1


for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,knn.predict(i))

print(a)

pprint(knn.debug())