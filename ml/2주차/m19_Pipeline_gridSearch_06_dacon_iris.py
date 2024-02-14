import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline

path = "c://_data//dacon//iris//"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

x=train_csv.drop(['species'],axis=1)
y=train_csv['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


splits = 3
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=28)

parameters = [
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]}
]


print('==============하빙그리드서치 시작==========================')
pipe = Pipeline([('MM', MinMaxScaler()),
                 ('RF', RandomForestClassifier())])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state= 66,
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=40)


start_time = time.time()
model.fit(x_train,y_train)
end_time=time.time()

print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(x_test,y_test)) 

y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(x_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time:",round(end_time-start_time,2),"s")
# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

# ==============하빙그리드서치 시작==========================
# n_iterations: 3
# n_required_iterations: 6
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 90
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 20
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 30
# n_resources: 40
# Fitting 3 folds for each of 30 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 15
# n_resources: 80
# Fitting 3 folds for each of 15 candidates, totalling 45 fits
# 최적의 매개변수: RandomForestClassifier(min_samples_leaf=3)
# 최적의 파라미터: {'min_samples_leaf': 3, 'min_samples_split': 2}
# best_score: 0.9487179487179488
# model.score: 0.9333333333333333
# acc.score: 0.9333333333333333
# best_acc.score: 0.9333333333333333
# time: 9.98 s
# PS C:\Study> 