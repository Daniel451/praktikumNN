from tempfile import TemporaryFile
import numpy as np

n = 10000;

a = np.array([[2,0],[3,0],[3,1],[5,0],[5,1],[5,2]])
b = np.arange(n) * 10
c = np.arange(n) * -0.5

file = TemporaryFile()

file = 'save/xyz.sd'
np.savez(file, a = a, b = b, c = c);
