# https://dacon.io/competitions/open/235610/mysubmission



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['species',], axis=1)
y = train_csv['species']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=3, stratify=y)

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

# n_components = 1, Accuracy: 0.875
# n_components = 2, Accuracy: 0.8958333333333334
# n_components = 3, Accuracy: 0.8958333333333334
# n_components = 4, Accuracy: 0.8958333333333334

EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 
# 0.95530892



# y_test = np.argmax(y_test, axis=1)
# y_submit = np.argmax(y_submit, axis=1)
# submission_csv['species'] = y_submit       
                    

# submission_csv.to_csv(path + "submission_02_14_1_.csv", index=False)
# print(submission_csv)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)
    
# model.score :  0.9583333333333334
# accuracy_score :  0.9583333333333334


# DecisionTreeClassifier accuracy: 0.9167
# DecisionTreeClassifier : [0.0208454  0.         0.43160394 0.54755067]
# RandomForestClassifier accuracy: 0.8958
# RandomForestClassifier : [0.09871251 0.03986515 0.50675715 0.35466518]
# GradientBoostingClassifier accuracy: 0.9167
# GradientBoostingClassifier : [0.01890886 0.00964932 0.72181912 0.2496227 ]
# XGBClassifier accuracy: 0.9167
# XGBClassifier : [0.01613282 0.04970136 0.8050349  0.12913094]




# GradientBoostingClassifier accuracy: 0.9167
# GradientBoostingClassifier : [0.01168437 0.01026084 0.68757583 0.29047896]
# XGBClassifier accuracy: 0.9167
# XGBClassifier : [0.01613282 0.04970136 0.8050349  0.12913094]



