from sklearn.datasets import load_digits
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score,f1_score, roc_auc_score
import time
import pickle
import joblib


#1.데이터
X,y = load_digits(return_X_y=True)

print(X.shape)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3, stratify=y)



mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# path = "c://_data//_save//_pickle_test//"
# model = pickle.load(open(path + "m39_pickle1_save.dat", 'rb'))

# path =  "c://_data//_save//_joblib_test//"
# model = joblib.load(path + "m40_joblib1_save.dat")


path =  "c://_data//_save//"
model = XGBClassifier()
model.load_model(path + "m41_XGB_save_model.dat")



# model = XGBClassifier()
# path = "c:\\_data\\_save\\_pickle_test\\"
# model = pickle.load(open(path + "m39_pickle1_save.dat", 'rb'))

# model.set_params(
#     **parameters,
#     early_stopping_rounds = 100                
                 
                #  )

# 3. 훈련
# start = time.time()

# model.fit(X_train, y_train, 
#           eval_set = [(X_train, y_train),(X_test, y_test)],
#           verbose = 1,  # true 디폴트 1 / false 디폴트 0 / verbose = n (과정을 n의배수로 보여줌)
#        #   eval_metric = 'rmse',     # 회귀 디폴트
#         #   eval_metric = 'mae',     # rmsle, mape, mphe....등등
#         #   eval_metric = 'logloss',     # 이진분류 디폴트, ACC
#           # eval_metric = 'error'    #  이진분류
#           eval_metric = 'mlogloss'     #  다중분류 디폴트, ACC
#           #  eval_metric = 'auc'       # 이진, 다중 둘다 가능
        
#           )


results = model.score(X_test, y_test)
print("최종점수 : ", results)
y_predict = model.predict(X_test)
# 최종점수 :  0.9694444444444444

############################### pickle ###################################################

# path = "C\\_data\\_save\\_pickle_test\\"
# # path = "C\\_data\\_save\\_joblib_test\\"
# pickle.dump(model, open(path + "m39_pickle1_save.dat", 'wb'))
















