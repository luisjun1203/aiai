from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler, LabelEncoder
from sklearn.utils import all_estimators
import warnings
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from imblearn.over_sampling import SMOTE


warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

lae = LabelEncoder()
y= lae.fit_transform(y)

# def remove_outlier(dataset:pd.DataFrame):
#     for label in dataset:
#         data = dataset[label]
#         q1 = data.quantile(0.25)
#         q3 = data.quantile(0.75)
#         iqr = q3-q1
#         upbound    = q3 + iqr*1.5
#         underbound = q1 - iqr*1.5
#         dataset.loc[dataset[label] < underbound, label] = underbound
#         dataset.loc[dataset[label] > upbound, label] = upbound
        
#     return dataset

# # print(train_csv.head(10))
# # print(test_csv.head(10))

# X = remove_outlier(X)
# # print(train_csv.shape,x.shape,sep='\n')
# # print(train_csv.max(),train_csv.min())
# # print(x.max(),x.min())

# X = X.astype(np.float32)
# y = y.astype(np.float32)



# print(X)
# print(y)        # [4 4 1 ... 2 2 2]
# print(X.shape)  # (581012, 54)
# print(y.shape)  # (581012,)
# # print(X.value_counts())
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))


# y = y.copy()


# for i, v in enumerate(y):
#     if v <=4:
#         y[i] = 0
#     elif v==5:
#         y[i]=1
#     # elif v==6:
#     #     y[i]=2
#     # elif v==7:
#     #     y[i]=3    
#     # elif v==8:
#     #     y[i]=4
#     else:
#         y[i]=2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 300,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}


# 2. 모델 구성

# smote = SMOTE(k_neighbors=5)
# X_train, y_train = smote.fit_resample(X_train, y_train)

model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1,
        eval_metric ='mlogloss' 
        )


# 4. 평가

result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print("acc_score ", acc)
print("f1_score", f1)

#before
# 최종점수 :  0.8504341540235623
# acc_score  0.8504341540235623
# f1_score 0.8176013685201714


# after
# 최종점수 :  0.7972083336919012
# acc_score  0.7972083336919012
# f1_score 0.7555326434971265