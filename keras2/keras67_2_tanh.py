import numpy as np
import matplotlib.pyplot as plt

# -1< tanh < 1
X = np.arange(-5, 5, 0.1)
y = np.tanh(X)

plt.plot(X,y)
plt.grid()
plt.show()







