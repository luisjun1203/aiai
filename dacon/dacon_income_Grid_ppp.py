import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
import random

import time
path = "c:\\_data\\dacon\\income\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

test_csv = test_csv.fillna('Child 18+ never marr Not in a subfamily')

train_csv.loc[train_csv['Industry_Status']=='Armed Forces', 'Industry_Status'] = 'Retail'
test_csv.loc[test_csv['Industry_Status']=='Armed Forces', 'Industry_Status'] = 'Retail'

test_csv.loc[test_csv['Birth_Country']=='Holand-Netherlands', 'Birth_Country'] = 'Unknown'

test_csv.loc[test_csv['Birth_Country (Father)']=='Panama', 'Birth_Country (Father)'] = 'Unknown'



lae = LabelEncoder()
lae.fit(train_csv['Gender'])
train_csv['Gender'] = lae.transform(train_csv['Gender'])
test_csv['Gender'] = lae.transform(test_csv['Gender'])

lae.fit(train_csv['Education_Status'])
train_csv['Education_Status'] = lae.transform(train_csv['Education_Status'])
test_csv['Education_Status'] = lae.transform(test_csv['Education_Status'])

lae.fit(train_csv['Employment_Status'])
train_csv['Employment_Status'] = lae.transform(train_csv['Employment_Status'])
test_csv['Employment_Status'] = lae.transform(test_csv['Employment_Status'])

lae.fit(train_csv['Industry_Status'])
train_csv['Industry_Status'] = lae.transform(train_csv['Industry_Status'])
test_csv['Industry_Status'] = lae.transform(test_csv['Industry_Status'])

lae.fit(train_csv['Occupation_Status'])
train_csv['Occupation_Status'] = lae.transform(train_csv['Occupation_Status'])
test_csv['Occupation_Status'] = lae.transform(test_csv['Occupation_Status'])

lae.fit(train_csv['Race'])
train_csv['Race'] = lae.transform(train_csv['Race'])
test_csv['Race'] = lae.transform(test_csv['Race'])

lae.fit(train_csv['Hispanic_Origin'])
train_csv['Hispanic_Origin'] = lae.transform(train_csv['Hispanic_Origin'])
test_csv['Hispanic_Origin'] = lae.transform(test_csv['Hispanic_Origin'])

lae.fit(train_csv['Martial_Status'])
train_csv['Martial_Status'] = lae.transform(train_csv['Martial_Status'])
test_csv['Martial_Status'] = lae.transform(test_csv['Martial_Status'])

lae.fit(train_csv['Citizenship'])
train_csv['Citizenship'] = lae.transform(train_csv['Citizenship'])
test_csv['Citizenship'] = lae.transform(test_csv['Citizenship'])  
  
lae.fit(train_csv['Birth_Country'])
train_csv['Birth_Country'] = lae.transform(train_csv['Birth_Country'])
test_csv['Birth_Country'] = lae.transform(test_csv['Birth_Country'])  

lae.fit(train_csv['Birth_Country (Mother)'])
train_csv['Birth_Country (Mother)'] = lae.transform(train_csv['Birth_Country (Mother)'])
test_csv['Birth_Country (Mother)'] = lae.transform(test_csv['Birth_Country (Mother)'])

lae.fit(train_csv['Birth_Country (Father)'])
train_csv['Birth_Country (Father)'] = lae.transform(train_csv['Birth_Country (Father)'])
test_csv['Birth_Country (Father)'] = lae.transform(test_csv['Birth_Country (Father)'])

lae.fit(train_csv['Tax_Status'])
train_csv['Tax_Status'] = lae.transform(train_csv['Tax_Status'])
test_csv['Tax_Status'] = lae.transform(test_csv['Tax_Status'])

lae.fit(train_csv['Household_Summary'])
train_csv['Household_Summary'] = lae.transform(train_csv['Household_Summary'])
test_csv['Household_Summary'] = lae.transform(test_csv['Household_Summary'])

lae.fit(train_csv['Income_Status'])
train_csv['Income_Status'] = lae.transform(train_csv['Income_Status'])
test_csv['Income_Status'] = lae.transform(test_csv['Income_Status'])


def classify_data(item):
    if 'Householder' in item or 'Spouse of householder' in item:
        return 'Householder_and_Spouse'
    elif 'Nonfamily householder' in item or 'Secondary individual' in item:
        return 'Nonfamily_Householder'
    elif 'Child' in item:
        return 'Child'
    elif 'Grandchild' in item:
        return 'Grandchild'
    elif 'Other Rel' in item or 'Other Relative' in item:
        return 'Other_Relative'
    elif 'In group quarters' in item or 'Responsible Person of unrelated subfamily' in item or 'Spouse of Responsible Person of unrelated subfamily' in item:
        return 'Group_Quarters'
    else:
        return 'Unknown' 

classified_data = [classify_data(item) for item in train_csv['Household_Status']]
classified_data2 = [classify_data(item) for item in test_csv['Household_Status']]

train_csv['Household_Status'] = lae.fit_transform(classified_data)
test_csv['Household_Status'] = lae.fit_transform(classified_data2)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

train_csv = train_csv.drop(['Gains', 'Losses', 'Dividends'], axis=1)
test_csv = test_csv.drop(['Gains', 'Losses', 'Dividends'], axis=1)

  
X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']

test = test_csv
# lb = LabelEncoder()

# 라벨 인코딩할 열 목록
# columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status',
#                      'Occupation_Status','Race','Hispanic_Origin','Martial_Status',
#                      'Household_Status','Household_Summary','Citizenship','Birth_Country',
#                      'Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# # 데이터프레임 x의 열에 대해 라벨 인코딩 수행
# for column in columns_to_encode:
#     lb.fit(X[column])
#     X[column] = lb.transform(X[column])

# # 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
# for column in columns_to_encode:
#     lb.fit(test_csv[column])
#     test_csv[column] = lb.transform(test_csv[column])
    
# 데이터 스케일링
scaler = StandardScaler()
# scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test_csv = scaler.transform(test_csv)


r = random.randint(1,500)

# 훈련 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=53338046)

# XGBoost 모델 학습
# xgb_params = {'learning_rate': 0.05,
#             'n_estimators': 200,
#             'max_depth': 9,
#             'min_child_weight': 0.07709868781803283,
#             'subsample': 0.80309973945344,
#             'colsample_bytree': 0.9254025887963853,
#             'gamma': 6.628562492458777e-08,
#             'reg_alpha': 0.012998871754325427,
#             'reg_lambda': 0.10637051171111844}

# model = xgb.XGBRegressor(**xgb_params)

parameters = {
'n_estimators': [100, 300, 500],  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
'learning_rate': [0.01, 0.05, 0.1],  # 학습률/ 디폴트 0.3/0~1/
'max_depth': [6, 9, 12],  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
'min_child_weight':  [0, 1, 5],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
'gamma': [0, 1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
'subsample': [0.5, 1],  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
'colsample_bytree': [0.5, 1],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
'colsample_bylevel': [0.5, 1], #  디폴트 1 / 0~1
'colsample_bynode': [0.5, 1], #  디폴트 1 / 0~1
'reg_alpha' : [0],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
'reg_lambda' :   [1],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
'objective': ['reg:squarederror'],  # 학습 태스크 파라미터
# 'num_class': [30],
'verbosity' : [1] 
}
#2. 모델 구성
model = GridSearchCV(XGBRegressor(), param_grid=parameters, cv=kfold, verbose=1,
                # refit = True,     # default
                    n_jobs=-1)



model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)], early_stopping_rounds=50,
          verbose=100)
import joblib

# 모델 저장
joblib.dump(model, "c://_data//dacon//income//weights//money_xgb_03_27_3.pkl")

# 저장된 모델 불러오기
loaded_model = joblib.load("c://_data//dacon//income//weights//money_xgb_03_27_3.pkl")
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

submission_csv.to_csv(path + "submisson_03_27_3_xgb.csv", index=False)

# return rmse_val
# time.sleep(1)

    
# import random
# for i in range(10000000):
#     b = (0.3)
#     a = random.randrange(1, 100000000)
#     # a = (79422819)
#     r = auto(a, 0.3)          
#     print("random_state : ", a)
#     if r < 500  :
#         print("random_state : ", a)
#         print("rmse : ", r)
#         break    
    
    
#random_state :  61062186    RMSE: 509.61084521379144
#random_state :  53338046   rmse :  484.37078689407923
#random_state :  79422819   rmse :  499.38601065590484
#random_state :  55973140   rmse :  498.85454619212214
# random_state :  66409007




# random_state :  53338046 , test_size = 0.2  Validation RMSE: 548.2939853261468 r 165     -> 542.507924671
# random_state :  53338046 , test_size = 0.15 Validation RMSE: 509.19259917148514 r 454     -> 543.4868875554
# 전처리 -> random_state :  53338046 , test_size = 0.15 



