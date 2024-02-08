import numpy as np
from sklearn .datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
X, y = load_iris(return_X_y=True)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2, stratify=y)

########### randomforest#########################

parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,10,12],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

 #2. 모델 구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                     n_jobs=-1)



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
# results = model.score(X_test, y_test)
# print(results)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# print(pd.DataFrame(model.cv_results_).T)


# best_score :  0.9916666666666668
# model.score :  0.8666666666666667
# accuracy_score :  0.8666666666666667
# 최적튠 ACC :  0.8666666666666667
# 걸린시간 :  2.77 초












