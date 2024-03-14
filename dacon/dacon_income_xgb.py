
import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model, load_model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, GRU, Dropout, Input, concatenate
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random as rn
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
tf.random.set_seed(3)
np.random.seed(3)

path = "c:\\_data\\dacon\\income\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv.shape)  # (20000, 22)
# print(test_csv.shape)   # (10000, 21)

# print(train_csv.info())

#   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   Age                     20000 non-null  int64
#  1   Gender                  20000 non-null  object
#  2   Education_Status        20000 non-null  object
#  3   Employment_Status       20000 non-null  object
#  4   Working_Week (Yearly)   20000 non-null  int64
#  5   Industry_Status         20000 non-null  object
#  6   Occupation_Status       20000 non-null  object
#  7   Race                    20000 non-null  object
#  8   Hispanic_Origin         20000 non-null  object
#  9   Martial_Status          20000 non-null  object
#  10  Household_Status        20000 non-null  object
#  11  Household_Summary       20000 non-null  object
#  12  Citizenship             20000 non-null  object
#  13  Birth_Country           20000 non-null  object
#  14  Birth_Country (Father)  20000 non-null  object
#  15  Birth_Country (Mother)  20000 non-null  object
#  16  Tax_Status              20000 non-null  object
#  17  Gains                   20000 non-null  int64
#  18  Losses                  20000 non-null  int64
#  19  Dividends               20000 non-null  int64
#  20  Income_Status           20000 non-null  object
#  21  Income                  20000 non-null  int64
# dtypes: int64(6), object(16)
# memory usage: 3.5+ MB
# None

# print(test_csv.info())
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   Age                     10000 non-null  int64
#  1   Gender                  10000 non-null  object
#  2   Education_Status        10000 non-null  object
#  3   Employment_Status       10000 non-null  object
#  4   Working_Week (Yearly)   10000 non-null  int64
#  5   Industry_Status         10000 non-null  object
#  6   Occupation_Status       10000 non-null  object
#  7   Race                    10000 non-null  object
#  8   Hispanic_Origin         10000 non-null  object
#  9   Martial_Status          10000 non-null  object
#  10  Household_Status        9999 non-null   object
#  11  Household_Summary       10000 non-null  object
#  12  Citizenship             10000 non-null  object
#  13  Birth_Country           10000 non-null  object
#  14  Birth_Country (Father)  10000 non-null  object
#  15  Birth_Country (Mother)  10000 non-null  object
#  16  Tax_Status              10000 non-null  object
#  17  Gains                   10000 non-null  int64
#  18  Losses                  10000 non-null  int64
#  19  Dividends               10000 non-null  int64
#  20  Income_Status           10000 non-null  object
# dtypes: int64(5), object(16)
# memory usage: 1.7+ MB
# None


# print(train_csv.isnull().sum())
# Household_Status 결측치 하나있음 
test_csv = test_csv.fillna('Child 18+ never marr Not in a subfamily')
# print(test_csv.isna().sum())

# print(np.unique(train_csv['Household_Status'], return_counts=True))
# print(np.unique(test_csv['Household_Status'], return_counts=True))


# print(np.unique(train_csv['Income_Status'], return_counts=True))
# (array(['Over Median', 'Under Median', 'Unknown'], dtype=object), array([  737, 13237,  6026], dtype=int64))

# print(np.unique(test_csv['Income_Status'], return_counts=True))
# (array(['Over Median', 'Under Median', 'Unknown'], dtype=object), array([ 404, 6642, 2954], dtype=int64))

# print(np.unique(train_csv['Birth_Country (Father)'], return_counts=True))       # 'Panama' o
# print(np.unique(test_csv['Birth_Country (Father)'], return_counts=True))        # 'Panama' x

# print(np.unique(train_csv['Birth_Country'], return_counts=True))            # 'Holand-Netherlands' o
# print(np.unique(test_csv['Birth_Country'], return_counts=True))             # 'Holand-Netherlands' x


# print(np.unique(train_csv['Household_Summary'], return_counts=True))        # 유난히 적은 애들이 있음
# print(np.unique(test_csv['Household_Summary'], return_counts=True))         # # 유난히 적은 애들이 있음

# print(np.unique(train_csv['Occupation_Status'], return_counts=True))
# print(np.unique(test_csv['Occupation_Status'], return_counts=True))

# (array(['Admin Support (include Clerical)', 'Armed Forces',
    #    'Craft & Repair', 'Farming & Forestry & Fishing',
    #    'Handlers/Cleaners', 'Machine Operators & Inspectors',
    #    'Management', 'Private Household Services', 'Professional',
    #    'Protective Services', 'Sales', 'Services',
    #    'Technicians & Support', 'Transportation', 'Unknown'], dtype=object), array([2709,    1, 1869,  296,  837, 1383, 1111,  105, 1488,  260, 1692,
    #    2313,  558,  690, 4688], dtype=int64))
    
# print(np.unique(train_csv['Industry_Status'], return_counts=True))   # 'Armed Forces' 1개,      
# print(np.unique(test_csv['Industry_Status'], return_counts=True))    # 'Armed Forces' 1개, 

# print(np.unique(train_csv['Gender'], return_counts=True))   # (array(['F', 'M'], dtype=object), array([10472,  9528], dtype=int64))         
# print(np.unique(test_csv['Gender'], return_counts=True))    # (array(['F', 'M'], dtype=object), array([5206, 4794], dtype=int64))

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
 
print(np.unique(train_csv['Household_Status'], return_counts=True))       
print(np.unique(test_csv['Household_Status'], return_counts=True))


  
X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']

# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=3)


splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=3)

parameters = {
    'XGB__n_estimators': [300],  # 부스팅 라운드의 수
    'XGB__learning_rate': [0.01],  # 학습률
    'XGB__max_depth': [9],  # 트리의 최대 깊이
    'XGB__min_child_weight': [1],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
    'XGB__gamma': [0.6],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
    'XGB__subsample': [0.6],  # 각 트리마다의 관측 데이터 샘플링 비율
    'XGB__colsample_bytree': [0.8],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
    # 'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
    # 'XGB__num_class': [26],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
    'XGB__verbosity' : [1],
    'XGB__reg_alpha' : [0.7],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'XGB__reg_lambda' : [1],
     
}

pipe = Pipeline([('RBS',RobustScaler()),
                 ('XGB', XGBRegressor(random_state=3,
                                    #    early_stopping_rounds = 50
                                       ))])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,   
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=10,
                    )


model.fit(X_train,y_train, 
        #   eval_test = [(X_test, y_test)]
          )



print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(X_test,y_test)) 

y_predict=model.predict(X_test)
# y_predict = lae.inverse_transform(y_predict)

print("r2.score:", r2_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(X_test)

print("best_acc.score:",r2_score(y_test,y_pred_best))

y_submit = model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
# y_submit = lae.inverse_transform(y_submit)
submission_csv['Income'] = y_submit
print(y_submit)

submission_csv.to_csv(path + "submisson_03_13_1_xgb.csv", index=False)


def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)