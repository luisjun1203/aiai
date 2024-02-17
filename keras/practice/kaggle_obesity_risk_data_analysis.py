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



path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv.describe())
#                 Age        Height        Weight          FCVC           NCP          CH2O           FAF           TUE
# count  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000  20758.000000
# mean      23.841804      1.700245     87.887768      2.445908      2.761332      2.029418      0.981747      0.616756
# std        5.688072      0.087312     26.379443      0.533218      0.705375      0.608467      0.838302      0.602113
# min       14.000000      1.450000     39.000000      1.000000      1.000000      1.000000      0.000000      0.000000
# 25%       20.000000      1.631856     66.000000      2.000000      3.000000      1.792022      0.008013      0.000000
# 50%       22.815416      1.700000     84.064875      2.393837      3.000000      2.000000      1.000000      0.573887
# 75%       26.000000      1.762887    111.600553      3.000000      3.000000      2.549617      1.587406      1.000000
# max       61.000000      1.975663    165.057269      3.000000      4.000000      3.000000      3.000000      2.000000

# print(np.unique(train_csv['NCP'], return_counts=True))



# print(np.unique(train_csv['NCP'], return_counts=True))


# (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
#        'Walking'], dtype=object), array([ 3534,    32,    38, 16687,   467], dtype=int64))
########################### 교통수단 ###################################################

# train_csv.loc[train_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
# train_csv.loc[train_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

# test_csv.loc[test_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
# test_csv.loc[test_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

# print(np.unique(train_csv['MTRANS'], return_counts=True))
# print(np.unique(test_csv['MTRANS'], return_counts=True))

train_csv['Exercise_Score'] = train_csv['FAF'] - train_csv['TUE'] + train_csv['FCVC']
test_csv['Exercise_Score'] = test_csv['FAF'] - test_csv['TUE'] + test_csv['FCVC']

# print(train_csv['Exercise_Score'])


def classify_diet(caec, calc, favc, family_history):
    if family_history == 'yes':
        return 'Moderate'
    elif caec == 'Always' and calc == 'Frequently' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Frequently' and calc == 'Always' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Sometimes' and calc == 'Frequently'and favc == 'yes':
        return 'Moderate'
    elif caec == 'Sometimes' and calc == 'Always'and favc == 'yes':
        return 'Moderate'
    else:
        return 'Healthy'
    
train_csv['Diet_Class'] = train_csv.apply(lambda row: classify_diet(row['CAEC'], row['CALC'], row['FAVC'], row['family_history_with_overweight']), axis=1)  
print(np.unique(train_csv['Diet_Class'], return_counts=True))
# print(np.unique(train_csv['family_history_with_overweight'], return_counts=True))
print(train_csv['Diet_Class'])
    

