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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

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
    f1 = f1_score(y_test, y_predict, average='macro')
    
   
    accuracy_results[n] = f1
    print(f"n_components = {n}, f1_score : {f1}")


EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 

# 1.

# n_components = 1, f1_score : 0.15601455028426287
# n_components = 2, f1_score : 0.177220840551155
# n_components = 3, f1_score : 0.2717883423287208
# n_components = 4, f1_score : 0.30590499549165184
# n_components = 5, f1_score : 0.318125978391879
# n_components = 6, f1_score : 0.3252249624374514
# n_components = 7, f1_score : 0.3240676589420047
# n_components = 8, f1_score : 0.31703048843553455
# n_components = 9, f1_score : 0.31791587465713084
# n_components = 10, f1_score : 0.31967833265500417
# n_components = 11, f1_score : 0.3144517405830056
# n_components = 12, f1_score : 0.3166206245484099
# n_components = 13, f1_score : 0.4775602968302303

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








