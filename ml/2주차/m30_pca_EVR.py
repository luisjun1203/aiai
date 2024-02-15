
from sklearn.datasets import load_diabetes,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

# print(sk.__version__)       # 1.1.3

datasets = load_breast_cancer()
X = datasets['data']
y = datasets['target']

# print(X.shape, y.shape)     # (150, 4) (150, )

sts = StandardScaler()
X = sts.fit_transform(X)


pca = PCA(n_components=18)   
X = pca.fit_transform(X)
# print(X)                    
# print(X.shape)              # (150, 2)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)


model = RandomForestClassifier()

model.fit(X_train, y_train)

results = model.score(X_test, y_test)

print("model.score : ", results)

# y_predict = model.predict(X_test)


EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 

import matplotlib.pyplot as plt

plt.plot(evr_cumsum)
plt.grid()
plt.show()
# [0.44272026 0.63243208 0.72636371 0.79238506 0.84734274 0.88758796
#  0.9100953  0.92598254 0.93987903 0.95156881 0.961366   0.97007138
#  0.97811663 0.98335029 0.98648812 0.98915022 0.99113018 0.99288414
#  0.9945334  0.99557204 0.99657114 0.99748579 0.99829715 0.99889898
#  0.99941502 0.99968761 0.99991763 0.99997061 0.99999557 1.        ]


# diabetes
# [0.40242108 0.14923197 0.12059663 0.09554764 0.06621814 0.06027171
#  0.05365657 0.0433682  0.007832   0.00085607]
    # 1.0

#  cancer
#  model.score :  0.916083916083916
#  [4.42720256e-01 1.89711820e-01 9.39316326e-02 6.60213492e-02
#  5.49576849e-02 4.02452204e-02 2.25073371e-02 1.58872380e-02
#  1.38964937e-02 1.16897819e-02 9.79718988e-03 8.70537901e-03
#  8.04524987e-03 5.23365745e-03 3.13783217e-03 2.66209337e-03
#  1.97996793e-03 1.75395945e-03 1.64925306e-03 1.03864675e-03
#  9.99096464e-04 9.14646751e-04 8.11361259e-04 6.01833567e-04
#  5.16042379e-04 2.72587995e-04 2.30015463e-04 5.29779290e-05
#  2.49601032e-05 4.43482743e-06]

# print(sum(EVR))     # 1.0
