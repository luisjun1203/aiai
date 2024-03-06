import numpy as np
import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score





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
# print(X)
# print(X.shape)
y = train_csv['quality']-3
# print(y.shape)


# y = pd.get_dummies(y)

# print(y)
# print(y.shape)      #(5497, 7)      # 3,4,5,6,7,8,9

# print(X)

# print(X.shape)          # (5497, 12)
# print(y.shape)          #(5497, 7)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=777)      

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6,
# 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 1000,
# 'num_class': 30, 'objective': 'multi:softmax', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}

parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 13,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.8,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.8,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    'objective' : ['multi:softmax']
}




model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)


model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='mlogloss' 
          )


# 4. 평가, 예측
result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score ", acc)


# 최종점수 :  0.55
# acc_score  0.55

# 최종점수 :  0.5490909090909091
# acc_score  0.5490909090909091

# 최종점수 :  0.6790909090909091
# acc_score  0.6790909090909091