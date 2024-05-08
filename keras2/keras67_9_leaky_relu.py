import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)

# def leakyrelu(X):
    # return np.maximum(0.1 * X, X)

leakyrelu = lambda X : np.maximum(0.1 * X, X)

y = leakyrelu(X)

plt.plot(X, y)
plt.grid()
plt.show()









