import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# def save_code_to_file(filename=None):
# if filename is None:
#     # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
#     filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
# else:
#     filename = filename + ".txt"
# with open(__file__, "r") as file:
#     code = file.read()

# with open(filename, "w") as file:
#     file.write(code)


path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
  
  
 

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

  
train_csv['대출기간'] = pd.Categorical(train_csv['대출기간']).codes
test_csv['대출기간'] = pd.Categorical(test_csv['대출기간']).codes

test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')

train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'

test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'

lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])

lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

lae.fit(y)
y = lae.transform(y)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train) 
# X_test = mms.transform(X_test)
# test_csv = mms.transform(test_csv)

# models = [
# # DecisionTreeClassifier(),
# # RandomForestClassifier(),
# # GradientBoostingClassifier(),
# XGBClassifier()
# ]
model = XGBClassifier()

model.fit(X_train, y_train)

initial_predictions = model.predict(X_test)
initial_accuracy = f1_score(y_test, initial_predictions, average='macro')
print(f"초기 모델 정확도: {initial_accuracy:.4f}")

# 특성 중요도 기반으로 하위 20% 컬럼 제거
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)
bottom_20_percent = int(len(indices) * 0.2)
columns_to_drop = X.columns[indices[:bottom_20_percent]]
X_dropped = X.drop(columns=columns_to_drop)


X_train_dropped, X_test_dropped, y_train, y_test = train_test_split(X_dropped, y, test_size=0.3, random_state=42)

# 수정된 데이터셋으로 모델 재학습
model.fit(X_train_dropped, y_train)

# 수정된 모델 성능 평가
new_predictions = model.predict(X_test_dropped)
new_accuracy = f1_score(y_test, new_predictions, average='macro')
print(f"수정된 모델 정확도: {new_accuracy:.4f}")


# for model in models:
#     model_name = model.__class__.__name__
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     f1 = f1_score(y_test, predictions, average='macro')
#     print(f"{model_name} accuracy: {f1:.4f}")
#     print(model.__class__.__name__, ":", model.feature_importances_)


# y_predict = model.predict(X_test_dropped) 

# y_submit = model.predict(test_csv)  

# y_submit = pd.DataFrame(y_submit)
# submission_csv['대출등급'] = y_submit
# # print(y_submit)

# fs = f1_score(y_test, y_predict, average='macro')
# print("f1_score : ", fs)

# submission_csv.to_csv(path + "submisson_02_14_1_.csv", index=False)

# model.score :  0.42464520595361716
# f1_score :  0.22385580661899354



# XGBClassifier accuracy: 0.7673
# XGBClassifier : [0.04751153 0.40013993 0.01194792 0.01605869 0.03354851 0.0166168
#  0.01359611 0.02594885 0.01885613 0.18896157 0.20461066 0.01134649
#  0.01085671]
# f1_score :  0.7673481921460661


# 초기 모델 정확도: 0.7839
# 수정된 모델 정확도: 0.7710








