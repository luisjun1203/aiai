import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score

from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score


datasets = load_boston()

X = datasets.data
y = datasets.target

columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "1.81160617e-02 3.40742692e-04 7.66145096e-03 2.90278482e-04\
 2.31001277e-02 4.02977165e-01 9.80089494e-03 9.38535032e-02\
 1.78711502e-03 1.23622337e-02 3.12324897e-02 6.32992967e-03\
 3.92148007e-01"


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




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

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

    model = RandomForestRegressor(random_state=3)
    model.fit(X_train_p, y_train)
    y_predict = model.predict(X_test_p)
    r2 = r2_score(y_test, y_predict)
    
   
    accuracy_results[n] = r2
    print(f"n_components = {n}, Accuracy: {r2}")


EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 

#  0.871346

# n_components = 1, Accuracy: 0.3347750095677938
# n_components = 2, Accuracy: 0.7846487791000327
# n_components = 3, Accuracy: 0.7921333490830424
# n_components = 4, Accuracy: 0.8036790678233969
# n_components = 5, Accuracy: 0.8259970606924636
# n_components = 6, Accuracy: 0.825577036791315
# n_components = 7, Accuracy: 0.821868341502593
# n_components = 8, Accuracy: 0.8137634825479113
# n_components = 9, Accuracy: 0.8128032878924619
# n_components = 10, Accuracy: 0.8120685819809483


# random_state=42 epochs=1000, batch_size=15, test_size=0.15
# 로스 :  14.557331085205078
# R2스코어 :  0.7770829847720614

# random_state=42 epochs=10000, batch_size=15,test_size=0.15
# 로스 :  14.244630813598633
# R2스코어 :  0.7818713632982545

# random_state=42, epochs=1000, batch_size=15
# 로스 :  13.158198356628418
# R2스코어 :  0.7985079556506842

# random_state=20, epochs=1500, batch_size=15
# 로스 :  15.55542278289795
# R2스코어 :  0.80151274445764

# random_state=20, epochs=1500, batch_size=15
# loss = 'mse'
# R2스코어 :  0.793961027822985
# loss = 'mae'
# R2스코어 :  0.7919380058581664

# GradientBoostingRegressor accuracy: 0.8529
# GradientBoostingRegressor : [1.81160617e-02 3.40742692e-04 7.66145096e-03 2.90278482e-04
#  2.31001277e-02 4.02977165e-01 9.80089494e-03 9.38535032e-02
#  1.78711502e-03 1.23622337e-02 3.12324897e-02 6.32992967e-03
#  3.92148007e-01]


# GradientBoostingRegressor accuracy: 0.8542
# GradientBoostingRegressor : [0.01876913 0.00818736 0.02358514 0.40347126 0.00810269 0.09198756
#  0.01560411 0.03034294 0.00644294 0.39350687]
