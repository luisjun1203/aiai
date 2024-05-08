import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)

# def relu(X):
    # return np.maximum(0, X)

relu = lambda X : np.maximum(0, X)

y = relu(X)

plt.plot(X, y)
plt.grid()
plt.show()















