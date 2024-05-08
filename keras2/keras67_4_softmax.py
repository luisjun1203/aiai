import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1, 5)

# def softmax(X):
    # return np.exp(X) / np.sum(np.exp(X))

softmax = lambda X : np.exp(X) / np.sum(np.exp(X))

y = softmax(X)

ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90)
plt.show()




# plt.plot(X, y)
# plt.grid()
























