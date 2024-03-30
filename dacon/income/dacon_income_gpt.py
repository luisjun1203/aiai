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

lae = LabelEncoder()



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



for col in ['Gains', 'Losses', 'Dividends']:
    threshold = train_csv[col].quantile(0.99)
    train_csv[col] = train_csv[col].apply(lambda x: x if x <= threshold else threshold)
    test_csv[col] = test_csv[col].apply(lambda x: x if x <= threshold else threshold)

# Birth_Country 컬럼만 유지하고 나머지는 제거
train_csv.drop(['Birth_Country (Father)', 'Birth_Country (Mother)'], axis=1, inplace=True)
test_csv.drop(['Birth_Country (Father)', 'Birth_Country (Mother)'], axis=1, inplace=True)

def age_group(age):
    if age < 18:
        return 'Youth'
    elif age < 65:
        return 'Adult'
    else:
        return 'Senior'

train_csv['Age_Group'] = train_csv['Age'].apply(age_group)
test_csv['Age_Group'] = test_csv['Age'].apply(age_group)

train_csv['Age_Group'] = lae.fit_transform(train_csv['Age_Group'])
test_csv['Age_Group'] = lae.fit_transform(test_csv['Age_Group'])

# print(train_csv.head())


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



n_splits= 7
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# train_csv = train_csv.drop(['Gains', 'Losses', 'Dividends'], axis=1)
# test_csv = test_csv.drop(['Gains', 'Losses', 'Dividends'], axis=1)


X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']

test = test_csv

scaler = StandardScaler()
# scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test_csv = scaler.transform(test_csv)


r = random.randint(1,500)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=53338046)


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
model = GridSearchCV(XGBRegressor(), param_grid=parameters, cv=kfold, verbose=2,
                # refit = True,     # default
                    n_jobs=-1)


model.fit(X_train, y_train,
        #   eval_set=[(X_val, y_val)], early_stopping_rounds=50,0
          verbose=2)


import joblib



# 모델 저장
joblib.dump(model, "c://_data//dacon//income//weights//money_XGB_03_30_2.pkl")

loaded_model = joblib.load("c://_data//dacon//income//weights//money_XGB_03_30_2.pkl")

y_pred_val = model.predict(X_val)
best_params = model.best_params_

rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val,'r',r)

y_submit = model.predict(test_csv)  
submission_csv['Income'] = y_submit
print(y_submit)
print('최적 파라미터 : ',best_params)
submission_csv.to_csv(path + "submisson_03_30_2_XGB.csv", index=False)

# 99
# Validation RMSE: 549.7134128155112 r 430
# [ 54.16168   32.584335 430.25546  ... 486.685     26.965923 661.0956  ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 0.5, 'colsample_bytree': 0.5,
# 'gamma': 1, 'learning_rate': 0.01, 'max_depth': 12, 'min_child_weight': 5, 'n_estimators': 300, 'objective': 
# 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5, 'verbosity': 2}

# 75
# Validation RMSE: 551.2922966781219 r 430
# [ 25.408865  28.349987 421.38376  ... 423.93857   27.019299 663.0905  ]
# 최적 파라미터 :  {'colsample_bylevel': 0.5, 'colsample_bynode': 0.5,
# 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 12,
# 'min_child_weight': 5, 'n_estimators': 300, 'objective': 
# 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5, 'verbosity': 2}

#95
# Validation RMSE: 549.9060589399403 r 430
# [ 32.48497   69.11335  419.05032  ... 466.5743    26.951122 633.8182  ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 0.5, 'colsample_bytree': 0.5,
# 'gamma': 1, 'learning_rate': 0.01, 'max_depth': 12, 'min_child_weight': 5, 'n_estimators': 300, 'objective': 
# 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5, 'verbosity': 2}


#90
# Validation RMSE: 551.6303865607747 r 430
# [ 60.637875  55.13174  424.15762  ... 598.8777    26.983376 683.41516 ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 0.5, 'colsample_bytree': 0.5,
# 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 12, 'min_child_weight': 0, 'n_estimators': 300, 'objective': 
# 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1, 'verbosity': 2}

#97
# Validation RMSE: 550.5479238142116 r 430
# [ 35.109196  58.394943 417.7776   ... 480.23856   26.665348 688.6879  ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 0.5, 'colsample_bytree': 0.5, 
# 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 12, 'min_child_weight': 5, 'n_estimators': 300, 'objective': 
# 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5, 'verbosity': 2}



# submisson_03_30_2_XGB.csv
# Validation RMSE: 549.0087712353022 r 122
# [  3.0830472   7.406462  429.5641    ... 373.574       2.815016
#  625.61096  ]
# 최적 파라미터 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0,
# 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 0, 'n_estimators': 500, 'objective': 'reg:squarederror',
# 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1, 'verbosity': 1}






