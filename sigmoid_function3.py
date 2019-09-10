import matplotlib.pyplot as plt
import numpy as np
import math


def sigmoid(a):
    s = 1 / (1 + e**-a)
    return s


def d_sigmoid(a):
    d = sigmoid(a) * (1-sigmoid(a))
    return d


e = math.e
dx = 0.1
x = np.arange(-8, 8, dx)

# sigmoid
y_sig = sigmoid(x)
# sigmoid x lean
y_dsig = (sigmoid(x+dx) - sigmoid(x)) / dx
# sigmoid y lean
# dy_sig = sigmoid(x)*(1-sigmoid(x))
dy_sig = d_sigmoid(x)

plt.plot(x, y_sig, label="sigmoid")
plt.plot(x, y_dsig, label="d_sigmoid")
plt.plot(x, dy_sig, label="dy_sigmoid")
plt.legend()
plt.show()

