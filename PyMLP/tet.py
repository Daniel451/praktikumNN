import numpy as n


def _tanh_deriv(x):  
    return x   #!! Achtung !!





l = 0



a = n.array( [[0 , 1]] )

W = [n.array([[-0.133301604 ,  0.0008572114],
               [ 0.0777116726, -0.3206972183],
               [-0.0198465845, -0.4098421422]]), n.array([[-0.2522253992,  0.093474046 , -0.1166710151]])]
B = [n.array([[ 0.0848112323],
               [ 0.4111778799],
               [ 0.0436669983]]), n.array([[ 0.1515373456]])]




print (str(a))
print (str(W[l]))
print (str(B[l]))


int_a=n.atleast_2d((a * W[l]).sum(axis=1)) + n.transpose(B[l])

print('ERgebnis:')
print(str(int_a))
#[[111 225 339 453 567]]



print('tanh(0.02) - ' + str(n.tanh(0.02)))
print('tanh(0.4) - ' + str(n.tanh(0.4)))
print('tanh(0.7) - ' + str(n.tanh(0.7)))
 