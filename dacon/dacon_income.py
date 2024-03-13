
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
test_csv = test_csv.fillna('Householder')
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

train_csv.loc[train_csv['Industry_Status']=='', ''] = ''

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

print(np.unique(train_csv['Employment_Status'], return_counts=True))            
print(np.unique(test_csv['Employment_Status'], return_counts=True)) 
'''
lae.fit(train_csv['Industry_Status'])
train_csv['Industry_Status'] = lae.transform(train_csv['Industry_Status'])
test_csv['Industry_Status'] = lae.transform(test_csv['Industry_Status'])

lae.fit(train_csv['Occupation_Status'])
train_csv['Occupation_Status'] = lae.transform(train_csv['Occupation_Status'])
test_csv['Occupation_Status'] = lae.transform(test_csv['Occupation_Status'])

lae.fit(train_csv['Race'])
train_csv['Race'] = lae.transform(train_csv['Race'])
test_csv['Race'] = lae.transform(test_csv['Race'])

lae.fit(train_csv['Hispanic_Origin '])
train_csv['Hispanic_Origin '] = lae.transform(train_csv['Hispanic_Origin '])
test_csv['Hispanic_Origin '] = lae.transform(test_csv['Hispanic_Origin '])

lae.fit(train_csv['Martial_Status '])
train_csv['Martial_Status '] = lae.transform(train_csv['Martial_Status '])
test_csv['Martial_Status '] = lae.transform(test_csv['Martial_Status '])

  
X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.105, shuffle=True, random_state=698423134)
'''