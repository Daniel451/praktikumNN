import numpy as np


def _tanh(x):  # diese Funktion stellt die Übertragungsfunktion der Neuronen dar. Forwardpropagation
    return np.tanh(x)


def _tanh_deriv(
        x):  # diese Funktion stellt die Ableitung der Übertragungsfunktion dar. Sie ist für die Backpropagation nötig.
    return 1.0 - np.power(np.tanh(x), 2)





l=0
Activation=[np.array([[ 0.2, -0.3, -0.5, 0.4]]   )]

W=[np.array([   [ -1.86,   7.47,  -3.85],
                [ -1.66,  -2.07,   2.38],
                [  6.83,  -6.67,   0.24],
                [ -6.46,  -4.00,  -0.99]]   )]

delta=np.array([[ 0.4 ,  0.5 , 0.6]]   )

print('delta.dot(W[l].T): ',delta.dot(W[l].T))

print('tanhderiv:',_tanh_deriv(Activation[l]))

#delta.dot(W[l].T):  [[ 0.681 -0.271 -0.459 -5.178]]
delta_next = _tanh_deriv(Activation[l]) * delta.dot(W[l].T)

#tanhderiv: [[ 0.96104298  0.91513696  0.78644773  0.85563879]]


print ('result:   ',delta_next)

#result:    [[ 0.65447027 -0.24800212 -0.36097951 -4.43049763]]




