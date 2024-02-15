from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

(X_train, _), (X_test, _) = mnist.load_data()       # _ : 이 자리에 y가 들어와야 하지만 비워둘거야

print(X_train.shape, X_test.shape)      # (60000, 28, 28) (10000, 28, 28)

# X = np.append(X_train, X_test, axis=0)
X = np.concatenate([X_train, X_test], axis=0)       # list형식으로 바꿔줘야함

print(X.shape)                          # (70000, 28, 28)

####################### [실습] ###################################################
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일때 몇개?

X1= X.reshape(X.shape[0], -1)

print(X1.shape)
# PCA 적용
pca = PCA()
pca.fit(X1)


EVR = pca.explained_variance_ratio_         
# print(EVR)

evr_cumsum = np.cumsum(EVR)

# print(evr_cumsum)

n_components95 = np.sum(evr_cumsum >= 0.95) 
n_components99 = np.sum( evr_cumsum >= 0.99) 
n_components999 = np.sum(evr_cumsum >= 0.999) 
n_components1 = np.sum(evr_cumsum >= 1.0) 
print(n_components95)   # 631
print(n_components99)   # 454
print(n_components999)  # 299  
print(n_components1)    # 72


# print(evr_cumsum)

# print(sum(EVR)) 




