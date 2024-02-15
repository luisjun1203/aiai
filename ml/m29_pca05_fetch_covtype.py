from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

datasets = fetch_covtype()

X = datasets.data
y = datasets.target


columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "2.35706587e-01 4.81478779e-02 3.32470063e-02 6.10893116e-02\
 5.77017224e-02 1.16636661e-01 4.16720995e-02 4.39262560e-02\
 4.21169150e-02 1.09083355e-01 1.01819864e-02 6.03515862e-03\
 1.17111712e-02 3.68831587e-02 1.04801852e-03 9.03287542e-03\
 2.39440367e-03 1.30093885e-02 5.21761227e-04 2.65013408e-03\
 9.33725677e-06 4.38580427e-05 1.50606890e-04 1.09361901e-02\
 2.81691827e-03 1.06806535e-02 4.08679953e-03 3.32608552e-04\
 2.63036835e-06 8.15618286e-04 1.70397697e-03 2.27480557e-04\
 9.94458692e-04 1.92167430e-03 7.12940272e-04 1.45258061e-02\
 1.03483542e-02 4.11028817e-03 1.45009886e-04 4.43341498e-04\
 6.81962105e-04 1.93598042e-04 5.53663171e-03 3.16130349e-03\
 3.66774796e-03 5.55543689e-03 4.67047734e-03 6.06793266e-04\
 1.55982621e-03 8.50360573e-05 6.42950975e-04 9.86635877e-03\
 1.00973944e-02 5.87008309e-03"


fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
# print(fi_float)
fi_list = pd.Series(fi_float)

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

low_col_list = [X.columns[index] for index in low_idx_list]
if len(low_col_list) > len(X.columns) * 0.25:
    low_col_list = low_col_list[:int(len(X.columns)*0.25)]
# print('low_col_list',low_col_list)
X.drop(low_col_list,axis=1,inplace=True)
print("after X.shape",X.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True, random_state=3, stratify=y)

# print(X_train.shape, X_test.shape)      # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)      # (464809, 7) (116203, 7)


sts = StandardScaler()
sts.fit(X_train)
X_train = sts.transform(X_train)
X_test = sts.transform(X_test)


n_features = X_train.shape[1]
accuracy_results = {}

for n in range(1, n_features + 1):
    pca = PCA(n_components=n)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)  

    model = RandomForestClassifier(random_state=3)
    model.fit(X_train_p, y_train)
    y_predict = model.predict(X_test_p)
    acc = accuracy_score(y_test, y_predict)
    
   
    accuracy_results[n] = acc
    print(f"n_components = {n}, Accuracy: {acc}")


EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 


# n_components = 1, Accuracy: 0.4335669150008606
# n_components = 2, Accuracy: 0.5314007523788449
# n_components = 3, Accuracy: 0.6594723513068279
# n_components = 4, Accuracy: 0.7601730963094097
# n_components = 5, Accuracy: 0.7964741461975363
# n_components = 6, Accuracy: 0.8491800054092596
# n_components = 7, Accuracy: 0.8632735856015342
# n_components = 8, Accuracy: 0.8836222369747486
# n_components = 9, Accuracy: 0.8776671338300017
# n_components = 10, Accuracy: 0.8698728823977773
# n_components = 11, Accuracy: 0.8660421430503307
# n_components = 12, Accuracy: 0.8817142435642104
# n_components = 13, Accuracy: 0.8930048437461582
# n_components = 14, Accuracy: 0.8758378205601043
# n_components = 15, Accuracy: 0.8832927638858155
# n_components = 16, Accuracy: 0.8897248653831968
# n_components = 17, Accuracy: 0.8758476555776843
# n_components = 18, Accuracy: 0.8907821297730569
# n_components = 19, Accuracy: 0.8990337095227557
# n_components = 20, Accuracy: 0.8964569349167711
# n_components = 21, Accuracy: 0.9073738044306754
# n_components = 22, Accuracy: 0.9119913451845295




EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 

# 로스 :  0.6938604116439819
# ACC :  0.701832115650177
# accuracy_score :  0.7018321385850623

# 로스 :  0.5239526033401489
# ACC :  0.7768474221229553
# accuracy_score :  0.7768474135779627



# model.score :  0.5275601780138182
# accuracy_score :  0.5275601780138182



# RandomForestClassifier accuracy: 0.9508
# RandomForestClassifier : [2.35706587e-01 4.81478779e-02 3.32470063e-02 6.10893116e-02
#  5.77017224e-02 1.16636661e-01 4.16720995e-02 4.39262560e-02
#  4.21169150e-02 1.09083355e-01 1.01819864e-02 6.03515862e-03
#  1.17111712e-02 3.68831587e-02 1.04801852e-03 9.03287542e-03
#  2.39440367e-03 1.30093885e-02 5.21761227e-04 2.65013408e-03
#  9.33725677e-06 4.38580427e-05 1.50606890e-04 1.09361901e-02
#  2.81691827e-03 1.06806535e-02 4.08679953e-03 3.32608552e-04
#  2.63036835e-06 8.15618286e-04 1.70397697e-03 2.27480557e-04
#  9.94458692e-04 1.92167430e-03 7.12940272e-04 1.45258061e-02
#  1.03483542e-02 4.11028817e-03 1.45009886e-04 4.43341498e-04
#  6.81962105e-04 1.93598042e-04 5.53663171e-03 3.16130349e-03
#  3.66774796e-03 5.55543689e-03 4.67047734e-03 6.06793266e-04
#  1.55982621e-03 8.50360573e-05 6.42950975e-04 9.86635877e-03
#  1.00973944e-02 5.87008309e-03]


# RandomForestClassifier accuracy: 0.9517
# RandomForestClassifier : [0.24177724 0.04729986 0.03223823 0.06074479 0.05723692 0.11762072
#  0.0408306  0.04305952 0.04114048 0.1107642  0.01036883 0.00501996
#  0.01160774 0.03691716 0.00096006 0.0095996  0.00235982 0.01221417
#  0.00217144 0.0108647  0.00254159 0.01088519 0.00389085 0.00076003
#  0.00176801 0.00099266 0.00190196 0.00072315 0.01588769 0.01107157
#  0.00400235 0.00515232 0.00319365 0.00387939 0.00568215 0.00440035
#  0.00161041 0.00065391 0.01051029 0.01034752 0.0053489 ]

