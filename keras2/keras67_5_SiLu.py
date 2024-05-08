# SiLu(Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)

def silu(X) :
    return X * (1 / (1 + np.exp(-X)))     # X * sigmoid(X)

# silu = lambda X : X * (1 / 1 + np.exp(-X))

y = silu(X)

plt.plot(X, y)
plt.grid()
plt.show()














