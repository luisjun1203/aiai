
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
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier, VotingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

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
 
print(np.unique(train_csv['Household_Status'], return_counts=True))       
print(np.unique(test_csv['Household_Status'], return_counts=True))


  
X = train_csv.drop(['Income'], axis=1)
y = train_csv['Income']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713)


sts = StandardScaler()
sts.fit(X_train)
X_train = sts.transform(X_train)
X_test = sts.transform(X_test)


xgb_model = XGBRegressor(verbosity=1, subsample=0.6, n_estimators=200, 
                         min_child_weight=1, max_depth=9, learning_rate=0.05, 
                         gamma=0.5, colsample_bytree=0.6)

lgb_model = LGBMRegressor(verbosity=2, subsample=0.8, reg_lambda=0.1,
                          reg_alpha=0.1, num_leaves=31, n_estimators=200,
                          min_child_weight=0.001, min_child_samples=20, max_depth=0,
                          learning_rate=0.05, colsample_bytree=0.6, boosting_type='gbdt')

cat_model = CatBoostRegressor(iterations=300, learning_rate=0.05, max_depth=0)





# voting_model = HalvingGridSearchCV(VotingRegressor(estimators=[('xgb', xgb_model),
#                                             ('lgb', lgb_model)],
                                                
#                                             # ('cat', cat_model)],
#                                             # EarlyStopping= 100,
                                # ), param_grid=('xgb', xgb_model), ('lgb', lgb_model))
voting_model = VotingRegressor(estimators=[('xgb', xgb_model),
                                            ('lgb', lgb_model),
                                            ('cat', cat_model)],
                                            # EarlyStopping= 100,
                                )

voting_model.fit(X_train, y_train,
                #  eval_set = [(X_test, y_test)]
                 )


# accuracy = voting_model.score(X_test, y_test)
# y_submit = voting_model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)

# y_predict = voting_model.predict(X_test)



accuracy = voting_model.score(X_test, y_test)
y_submit = voting_model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)

y_predict = voting_model.predict(X_test)
# y_predict = lae.inverse_transform(y_predict)
r2 = r2_score(y_test, y_predict)

y_submit = pd.DataFrame(y_submit)
submission_csv['Income'] = y_submit
print(y_submit)
print("Voting Ensemble r2:", r2)

submission_csv.to_csv(path + "submisson_03_13_1_voting.csv", index=False)


def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)