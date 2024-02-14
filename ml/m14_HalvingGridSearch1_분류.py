import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time

best_score = [0, 999, 999]
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_splits = 5
StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)




parameters = [
    {"C":[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},
    {"C":[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {"C":[1,10,100,1000], 'kernel':['sigmoid'],'gamma':[0.001, 0.0001], 'degree':[3,4]}]
print("===============================하빙 그리드 시작=====================")
# model = GridSearchCV(SVC(), parameters, cv=StratifiedKFold, verbose=1, n_jobs=10)
model = HalvingGridSearchCV(SVC(), parameters, cv=StratifiedKFold, verbose=1, n_jobs=-3, random_state=5, factor=2)

starttime = time.time()
model.fit(X_train, y_train)
endtime = time.time()
print("best 매개변수 : ", model.best_estimator_)
print("best parameters : ", model.best_params_)

print("best score : ", model.best_score_)
print("model.score : ", model.score(X_test, y_test))

y_predict = model.predict(X_test)
print("best acc_score : ", accuracy_score(y_test, y_predict))
y_predict_best = model.best_estimator_.predict(X_test)
print("best tuned acc : ", accuracy_score(y_test, y_predict_best))
print("time : {} 초".format(round(endtime - starttime, 2)))

import pandas as pd
# print(pd.DataFrame(model.cv_results_))





# ===============================하빙 그리드 시작=====================
# n_iterations: 4
# n_required_iterations: 6
# n_possible_iterations: 4
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 34
# n_resources: 100
# Fitting 5 folds for each of 34 candidates, totalling 170 fits
# ----------
# iter: 1
# n_candidates: 17
# n_resources: 200
# Fitting 5 folds for each of 17 candidates, totalling 85 fits
# ----------
# iter: 2
# n_candidates: 9
# n_resources: 400
# Fitting 5 folds for each of 9 candidates, totalling 45 fits
# ----------
# iter: 3
# n_candidates: 5
# n_resources: 800
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# best 매개변수 :  SVC(C=100, gamma=0.001)
# best parameters :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# best score :  0.9824292452830189
# model.score :  0.9916666666666667
# best acc_score :  0.9916666666666667
# best tuned acc :  0.9916666666666667
# time : 1.59 초







