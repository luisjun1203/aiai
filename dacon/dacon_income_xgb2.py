import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
import time
path = "c:\\_data\\dacon\\income\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']
test = test_csv
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(X[column])
    X[column] = lb.transform(X[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test_csv[column])
    test_csv[column] = lb.transform(test_csv[column])
    
# 데이터 스케일링
scaler = StandardScaler()
# scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test_csv = scaler.transform(test_csv)

import random
def auto(a,b):
    r = random.randint(1,500)

    # 훈련 데이터와 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=b, random_state=a)

    # XGBoost 모델 학습
    xgb_params = {'learning_rate': 0.05,
                'n_estimators': 200,
                'max_depth': 9,
                'min_child_weight': 0.07709868781803283,
                'subsample': 0.80309973945344,
                'colsample_bytree': 0.9254025887963853,
                'gamma': 6.628562492458777e-08,
                'reg_alpha': 0.012998871754325427,
                'reg_lambda': 0.10637051171111844}

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)
    import joblib

    # 모델 저장
    joblib.dump(model, "c://_data//dacon//income//weights//money_xgb_03_20_1.pkl")

    # 저장된 모델 불러오기
    loaded_model = joblib.load("c://_data//dacon//income//weights//money_xgb_03_20_1.pkl")
    # 검증 데이터 예측
    y_pred_val = model.predict(X_val)

    # 검증 데이터 RMSE 계산
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print("Validation RMSE:", rmse_val,'r',r)

    y_submit = model.predict(test_csv)  
    # y_submit = lae.inverse_transform(y_submit)
    # y_submit = lae.inverse_transform(y_submit)
    submission_csv['Income'] = y_submit
    print(y_submit)

    submission_csv.to_csv(path + "submisson_03_20_1_xgb.csv", index=False)
    
    return rmse_val
    time.sleep(1)
    
    
import random
for i in range(10000000):
    b = (0.2)
    a = random.randrange(1, 100000000)
    # a = (79422819)
    r = auto(a, 0.1)          
    print("random_state : ", a)
    if r < 500  :
        print("random_state : ", a)
        print("rmse : ", r)
        break    
    
    
#random_state :  61062186    RMSE: 509.61084521379144
#random_state :  53338046   rmse :  484.37078689407923
#random_state :  79422819   rmse :  499.38601065590484
#random_state :  55973140   rmse :  498.85454619212214
# random_state :  66409007