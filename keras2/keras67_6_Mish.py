

import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)

def mish(X) :
    return X * np.tanh(np.log(1 + np.exp(X)))     # X * tanh(softplus(X))
                                                  # softplus(X) = log(1 + (X))
# mish = lambda X :  X * np.tanh(np.log(1 + np.exp(X)))

y = mish(X)

plt.plot(X, y)
plt.grid()
plt.show()





# elu, selu, leakyrelu








