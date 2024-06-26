import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
import random
from lightgbm import LGBMRegressor
import tensorflow as tf
tf.random.set_seed(3)
np.random.seed(3)
random.seed(3)

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

train_csv.loc[train_csv['Martial_Status']=='Married (Armed Force Spouse)', 'Martial_Status'] = 'Married'
test_csv.loc[test_csv['Martial_Status']=='Married (Armed Force Spouse)', 'Martial_Status'] = 'Married'

train_csv.loc[train_csv['Martial_Status']=='Married (Spouse Absent)', 'Martial_Status'] = 'Separated'
test_csv.loc[test_csv['Martial_Status']=='Married (Spouse Absent)', 'Martial_Status'] = 'Separated'

train_csv.loc[train_csv['Occupation_Status']=='Armed Forces', 'Occupation_Status'] = 'Unknown'
test_csv.loc[test_csv['Occupation_Status']=='Armed Forces', 'Occupation_Status'] = 'Unknown'


train_csv.loc[train_csv['Employment_Status']=='Seeking Part-Time', 'Employment_Status'] = 'Not Working'
test_csv.loc[test_csv['Employment_Status']=='Seeking Part-Time', 'Employment_Status'] = 'Not Working'

train_csv.loc[train_csv['Employment_Status']=='Seeking Full-Time', 'Employment_Status'] = 'Not Working'
test_csv.loc[test_csv['Employment_Status']=='Seeking Full-Time', 'Employment_Status'] = 'Not Working'

train_csv.loc[train_csv['Employment_Status']=='Part-Time (Usually Part-Time)', 'Employment_Status'] = 'Choice Part-Time'
test_csv.loc[test_csv['Employment_Status']=='Part-Time (Usually Part-Time)', 'Employment_Status'] = 'Choice Part-Time'

train_csv.loc[train_csv['Employment_Status']=='Part-Time (Usually Full-Time)', 'Employment_Status'] = 'Full-Time'
test_csv.loc[test_csv['Employment_Status']=='Part-Time (Usually Full-Time)', 'Employment_Status'] = 'Full-Time'


education_levels = {
    'Children': ['Children','Middle (7-8)','Elementary (1-4)', 'Elementary (5-6)', 'Kindergarten'],


    'High School': ['High Freshman', 'High Sophomore', 'High Junior', 'High Senior', 'High graduate'],


    'Degree': ['College', 'Associates degree (Academic)', 'Associates degree (Vocational)', 'Bachelors degree',
               'Masters degree', 'Doctorate degree', 'Professional degree']
}

def get_group(item):
    for level, items in education_levels.items():
        if item in items:
            return level
    return 'Unknown'

train_csv['Education_Status'] = train_csv['Education_Status'].apply(get_group)
test_csv['Education_Status'] = test_csv['Education_Status'].apply(get_group)




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




# print(train_csv[['Education_Status']])






# print(np.unique(train_csv['Education_Status'], return_counts=True))       
# print(np.unique(test_csv['Education_Status'], return_counts=True))

# print(np.unique(train_csv['Household_Status'], return_counts=True))       
# # print(np.unique(test_csv['Household_Status'], return_counts=True))

n_splits= 7
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


r = random.randint(400,500)

# 훈련 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=53338046)

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
'n_estimators': [300],  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
'learning_rate': [0.01],  # 학습률/ 디폴트 0.3/0~1/
'max_depth': [12],  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
'min_child_weight':  [0, 1, 5],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
'gamma': [0, 1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
'subsample': [0, 0.5, 1],  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
'colsample_bytree': [0, 0.5, 1],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
'colsample_bylevel': [0, 0.5, 1], #  디폴트 1 / 0~1
'colsample_bynode': [0, 0.5, 1], #  디폴트 1 / 0~1
'reg_alpha' : [0],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
'reg_lambda' :   [1],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
'objective': ['reg:squarederror'],  # 학습 태스크 파라미터
# 'num_class': [30],
'verbosity' : [2] 
}

# parameters = {
#     'n_estimators': [300],  # 부스팅 라운드 수
#     'learning_rate': [0.01, 0.1, 0.05],  # 학습률
#     'max_depth': [12],  # 트리의 최대 깊이
#     'min_child_samples':  [0, 1],  # 자식 노드가 가지고 있어야 할 최소 데이터 개수
#     'num_leaves': [62],  # 하나의 트리가 가질 수 있는 최대 리프의 수
#     'bagging_fraction': [0.5, 0.8],  # 데이터 샘플링 비율
#     'feature_fraction': [0.5, 0.8],  # 특성 샘플링 비율
#     'reg_alpha' : [0],  # L1 규제
#     'reg_lambda' :   [1],  # L2 규제
#     'objective': ['regression'],  # 학습 태스크 파라미터
#     'verbosity': [1] 
# }

#2. 모델 구성
model = GridSearchCV(XGBRegressor(), param_grid=parameters, cv=kfold, verbose=2,
                # refit = True,     # default
                    n_jobs=-1)

# model = GridSearchCV(LGBMRegressor(), param_grid=parameters, cv=kfold, verbose=1, n_jobs=-1)

model.fit(X_train, y_train,
        #   eval_set=[(X_val, y_val)], early_stopping_rounds=50,0
          verbose=2)

# model.fit(
#     X_train, y_train,
#     # eval_set=[(X_val, y_val)],
#     # early_stopping_rounds=50,
#     # verbose=1
# )

import joblib



# 모델 저장
# joblib.dump(model, "c://_data//dacon//income//weights//money_LGBM_03_29_5.pkl")
joblib.dump(model, "c://_data//dacon//income//weights//money_XGB_03_29_5.pkl")

# 저장된 모델 불러오기
# loaded_model = joblib.load("c://_data//dacon//income//weights//money_LGBM_03_29_5.pkl")
loaded_model = joblib.load("c://_data//dacon//income//weights//money_XGB_03_29_5.pkl")

# 검증 데이터 예측
y_pred_val = model.predict(X_val)
best_params = model.best_params_

# 검증 데이터 RMSE 계산
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val,'r',r)

y_submit = model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)
# y_submit = lae.inverse_transform(y_submit)
submission_csv['Income'] = y_submit
print(y_submit)
print('최적 파라미터 : ',best_params)
# submission_csv.to_csv(path + "submisson_03_29_5_LGBM.csv", index=False)
submission_csv.to_csv(path + "submisson_03_29_5_XGB.csv", index=False)


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




# random_state :  53338046 , test_size = 0.2            Validation RMSE: 548.2939853261468 r 165                                            -> 542.507924671
# random_state :  53338046 , test_size = 0.15           Validation RMSE: 509.19259917148514 r 454                                           -> 543.4868875554
# 전처리 -> random_state :  53338046 , test_size = 0.15 , kfold random state = 123, n_splits= 5  Validation RMSE: 507.5864672233961 r 468    -> 542.1382314334
# 전처리 -> random_state :  53338046 , test_size = 0.15 , kfold random state = 713, n_splits= 7  Validation RMSE: 507.67958456052907 r 470
# 전처리 -> random_state :  53338046 , test_size = 0.15 , kfold random state = 123, n_splits= 7  Validation RMSE: 507.54949182279205 r 449    -> 541.1157531034


# Validation RMSE: 509.66099446119154 r 434
# 최적 파라미터 :  {'bagging_fraction': 0.5, 'feature_fraction': 0.5, 
#             'learning_rate': 0.01, 'max_depth': 12, 'min_child_samples': 5,
#             'n_estimators': 300, 'num_leaves': 62, 'objective': 'regression',
#             'reg_alpha': 0, 'reg_lambda': 1, 'verbosity': 1}


# Validation RMSE: 549.2987012817131 r 474
# [ 64.83928   35.98204  428.5391   ... 538.7156    29.824112 625.8827  ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5,
#             'gamma': 0, 'learning_rate': 0.01, 'max_depth': 12, 'min_child_weight': 5,
#             'n_estimators': 300, 'objective': 'reg:squarederror', 
#             'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5, 'verbosity': 0}