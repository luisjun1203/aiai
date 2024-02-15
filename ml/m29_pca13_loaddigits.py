import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score


# 1 데이터
datasets = load_digits()

X = datasets.data
y = datasets.target

# # print(x)
# print(y)    # [0 1 2 ... 8 9 8]
# print(x.shape)  # (1797, 64)    # 64니까 8 * 8
# print(y.shape)  # (1797,)
# print(pd.value_counts(y, sort=False))

columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "0.00000000e+00 1.83855446e-03 2.33757656e-02 8.67034393e-03\
 9.64377303e-03 2.07726810e-02 7.13562576e-03 5.31971108e-04\
 8.65639576e-05 1.26127426e-02 2.56022222e-02 7.98058618e-03\
 1.63152935e-02 2.70933572e-02 4.66384806e-03 5.08067316e-04\
 9.94238314e-05 8.69925590e-03 2.32710779e-02 2.38209034e-02\
 2.88596389e-02 4.90956246e-02 7.54752515e-03 2.47508072e-04\
 1.03141732e-05 1.28844709e-02 3.99315304e-02 2.67499010e-02\
 3.25556188e-02 2.10092752e-02 3.38606704e-02 3.79818141e-05\
 0.00000000e+00 2.77463551e-02 2.94216424e-02 1.94453994e-02\
 3.74844155e-02 1.94164216e-02 2.47016798e-02 0.00000000e+00\
 4.14522363e-05 1.03303265e-02 3.72918073e-02 4.52343993e-02\
 2.20861381e-02 1.95523134e-02 2.03782608e-02 1.03715482e-04\
 9.91660633e-05 2.86286955e-03 1.84185516e-02 2.21683197e-02\
 1.39057193e-02 2.25943794e-02 2.39911740e-02 2.03154015e-03\
 3.46083491e-05 1.86332884e-03 2.11349188e-02 1.05644922e-02\
 2.24734080e-02 2.95248382e-02 1.61449165e-02 3.44132608e-03"


fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

low_col_list = [X.columns[index] for index in low_idx_list]
if len(low_col_list) > len(X.columns) * 0.25:
    low_col_list = low_col_list[:int(len(X.columns)*0.25)]
print('low_col_list',low_col_list)
X.drop(low_col_list,axis=1,inplace=True)
print("after X.shape",X.shape)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {'RF__n_estimators':[100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]},
#     {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
#     {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
#     {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
#     {'RF__min_samples_split':[2,3,5,10]},
#     {'RF__min_samples_split':[2,3,5,10]}
# ]



# 그리드로 돌렸을때 5 * 42 횟수.

# 2 모델
# model = RandomForestClassifier(C=1, kernel='linear', degree=3)
# print('==============하빙그리드서치 시작==========================')
# pipe = Pipeline([('MM', MinMaxScaler()),
#                  ('RF', RandomForestClassifier())])

# model = HalvingGridSearchCV(pipe, parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      , random_state= 66,
#                     # n_iter=10 # 디폴트 10
#                     factor=2,
#                     min_resources=40)

# model = RandomizedSearchCV(SVC(), 
#                      parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      )

sts = StandardScaler()
sts.fit(X_train)
X_train = sts.transform(X_train)
X_test = sts.transform(X_test)

n_features = X_train.shape[1]
accuracy_results = {}

for n in range(1, n_features + 1):
    pca = PCA(n_components=n)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)  

    model = RandomForestClassifier(random_state=3)
    model.fit(X_train_p, y_train)
    y_predict = model.predict(X_test_p)
    acc = accuracy_score(y_test, y_predict)
    
   
    accuracy_results[n] = acc
    print(f"n_components = {n}, Accuracy: {acc}")


# start_time = time.time()
# model.fit(x_train, y_train)
# end_time = time.time()

# print('최적의 매개변수 : ', model.best_estimator_)
# # 최적의 매개변수 :  SVC(C=1, kernel='linear')
# print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
# print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# # best_score :  0.975
# print('model_score : ', model.score(x_test, y_test))    # 
# # model_score :  0.9666666666666667


# y_predict = model.predict(x_test)
# print('accuracy_score', accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
#             # SVC(C-1, kernel='linear').predicict(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

# print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

# 그리드서치
# 최적 튠 ACC :  0.9916666666666667
# 걸린신간 :  2.51 초


# 랜덤 서치
# 최적 튠 ACC :  0.9916666666666667
# 걸린신간 :  1.05 초

    
#     RandomForestClassifier accuracy: 0.9944
# RandomForestClassifier : [0.00000000e+00 1.83855446e-03 2.33757656e-02 8.67034393e-03
#  9.64377303e-03 2.07726810e-02 7.13562576e-03 5.31971108e-04
#  8.65639576e-05 1.26127426e-02 2.56022222e-02 7.98058618e-03
#  1.63152935e-02 2.70933572e-02 4.66384806e-03 5.08067316e-04
#  9.94238314e-05 8.69925590e-03 2.32710779e-02 2.38209034e-02
#  2.88596389e-02 4.90956246e-02 7.54752515e-03 2.47508072e-04
#  1.03141732e-05 1.28844709e-02 3.99315304e-02 2.67499010e-02
#  3.25556188e-02 2.10092752e-02 3.38606704e-02 3.79818141e-05
#  0.00000000e+00 2.77463551e-02 2.94216424e-02 1.94453994e-02
#  3.74844155e-02 1.94164216e-02 2.47016798e-02 0.00000000e+00
#  4.14522363e-05 1.03303265e-02 3.72918073e-02 4.52343993e-02
#  2.20861381e-02 1.95523134e-02 2.03782608e-02 1.03715482e-04
#  9.91660633e-05 2.86286955e-03 1.84185516e-02 2.21683197e-02
#  1.39057193e-02 2.25943794e-02 2.39911740e-02 2.03154015e-03
#  3.46083491e-05 1.86332884e-03 2.11349188e-02 1.05644922e-02
#  2.24734080e-02 2.95248382e-02 1.61449165e-02 3.44132608e-03]


# RandomForestClassifier accuracy: 0.9861