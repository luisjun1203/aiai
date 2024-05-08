import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-5, 5, 0.1)

a = 1
# def elu(X):
    # return (X>0) * X + (X<=0) * (a * (np.exp(X)-1))

elu = lambda X : (X>0) * X + (X<=0) * (a * (np.exp(X)-1))


y = elu(X)

plt.plot(X,y)
plt.grid()
plt.show()












