# https://dacon.io/competitions/open/235610/mysubmission


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")


lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)

y = train_csv['quality']

lae.fit(y)
y = lae.transform(y)


# fi_str = "0.07470172 0.09809676 0.07729709 0.08594441 0.08591067 0.08888839\
#  0.09251841 0.10319924 0.08149437 0.08424209 0.12449336 0.00321348"


# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# # print(fi_float)
# fi_list = pd.Series(fi_float)

# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# # print('low_idx_list',low_idx_list)

# low_col_list = [X.columns[index] for index in low_idx_list]
# if len(low_col_list) > len(X.columns) * 0.25:
#     low_col_list = low_col_list[:int(len(X.columns)*0.25)]
# # print('low_col_list',low_col_list)
# X.drop(low_col_list,axis=1,inplace=True)
# print("after X.shape",X.shape)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42, stratify=y)       #9266, 781

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
#    0.93675385 ,0.98754721
# n_components = 1, Accuracy: 0.4793087767166894
# n_components = 2, Accuracy: 0.5306957708049114
# n_components = 3, Accuracy: 0.5625284220100045
# n_components = 4, Accuracy: 0.5879945429740792
# n_components = 5, Accuracy: 0.6089131423374261
# n_components = 6, Accuracy: 0.6120964074579355
# n_components = 7, Accuracy: 0.6202819463392452
# n_components = 8, Accuracy: 0.629831741700773
# n_components = 9, Accuracy: 0.6230104592996817  
# n_components = 10, Accuracy: 0.629831741700773
# n_components = 11, Accuracy: 0.6230104592996817
# n_components = 12, Accuracy: 0.6252842201000455

# y_submit = model.predict(test_csv)  
# y_predict = model.predict(X_test) 

# # y_test = np.argmax(y_test, axis=1)
# # y_predict = np.argmax(y_predict, axis=1)
# # y_submit = np.argmax(y_submit, axis=1)+3

# submission_csv['quality'] = y_submit



# submission_csv.to_csv(path + "submission_0214_1_.csv", index=False)

# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


# model.score :  0.4961346066393815
# accuracy_score :  0.4961346066393815



# RandomForestClassifier accuracy: 0.6485
# RandomForestClassifier : [0.07470172 0.09809676 0.07729709 0.08594441 0.08591067 0.08888839
#  0.09251841 0.10319924 0.08149437 0.08424209 0.12449336 0.00321348]


# RandomForestClassifier accuracy: 0.6385
# RandomForestClassifier : [0.11462832 0.10169926 0.10058473 0.10115564 0.11206451 0.12409431
#  0.10165279 0.10246075 0.14165969]






