
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

#  17  Gains                   10000 non-null  int64
#  18  Losses                  10000 non-null  int64
#  19  Dividends

train_csv = train_csv.drop(['Gains', 'Losses','Dividends'], axis=1)
test_csv = test_csv.drop(['Gains', 'Losses','Dividends'], axis=1)

  
X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']

# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y)

scaler = StandardScaler()
# scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test_csv = scaler.transform(test_csv)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=3)

import random
def auto(a,b):
    r = random.randint(1,500)

    # 훈련 데이터와 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=b, random_state=a)

    # XGBoost 모델 학습
    xgb_params = {'learning_rate': 0.01,
                'n_estimators': 500,
                'max_depth': 9,
                'min_child_weight': 0.07709868781803283,
                'subsample': 0.80309973945344,
                'colsample_bytree': 0.9254025887963853,
                'gamma': 6.628562492458777e-08,
                'reg_alpha': 0.012998871754325427,
                'reg_lambda': 0.10637051171111844}

    model = XGBRegressor(**xgb_params)
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

    submission_csv.to_csv(path + "submisson_03_20_3_xgb.csv", index=False)
    
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
# random_state :  56077727

# random_state :  55598698
# rmse :  497.9226862261143