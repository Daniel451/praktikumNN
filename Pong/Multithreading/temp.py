import numpy as np

S = np.array([[10.0,20.0]])
W = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])


RS = np.array([[1.0,2.0,3.0]])
RW = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])

B = np.array([ [ 1, 0.2, 0.3 ] ])








print(RS.dot(RW.T) + S.dot(W) + B )

