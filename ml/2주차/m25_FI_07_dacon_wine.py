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


fi_str = "0.07470172 0.09809676 0.07729709 0.08594441 0.08591067 0.08888839\
 0.09251841 0.10319924 0.08149437 0.08424209 0.12449336 0.00321348"


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




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42, stratify=y)       #9266, 781

models = [
DecisionTreeClassifier(),
RandomForestClassifier(),
GradientBoostingClassifier(),
XGBClassifier()
]


for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(model.__class__.__name__, ":", model.feature_importances_)
    # y_submit = model.predict(test_csv)  
    # y_predict = model.predict(X_test) 

  

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






