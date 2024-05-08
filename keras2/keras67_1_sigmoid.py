# 난 정말 시그모이드~~~~


import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(X):
#     return 1 / (1 + np.exp(-X))

sigmoid = lambda X : 1 / (1 + np.exp(-X))

X = np.arange(-5, 5, 0.1)
print(X)
print(len(X))   #100

y = sigmoid(X)
plt.plot(X,y)
plt.grid()
plt.show()






