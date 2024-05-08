import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)


# def selu(X, lambda_=1.0507, alpha=1.67326):
#     return lambda_ * ((X > 0) * X + (X <= 0) * (alpha * (np.exp(X) - 1)))

lambda_ = 1
alpha = 1

selu = lambda X : lambda_ * ((X > 0) * X + (X <= 0) * (alpha * (np.exp(X) - 1)))

y = selu(X)

plt.plot(X, y)
plt.grid()
plt.show()
















