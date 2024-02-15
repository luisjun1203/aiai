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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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



sts = StandardScaler()
X = sts.fit_transform(X)



n_features = len(np.unique(y))
    # print(n_features)

accuracy_results = {}

for n in range(1, n_features):
    lda = LinearDiscriminantAnalysis(n_components=n)
    X1 = lda.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.4, shuffle=False, random_state=3)


    model = RandomForestClassifier(random_state=3)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    
   
    accuracy_results[n] = acc
    print(f"n_components = {n}, Accuracy: {acc}")

EVR = lda.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 

# n_components = 1, Accuracy: 0.3852573018080668
# n_components = 2, Accuracy: 0.6675938803894298
# n_components = 3, Accuracy: 0.8205841446453408
# n_components = 4, Accuracy: 0.8734353268428373
# n_components = 5, Accuracy: 0.9012517385257302
# n_components = 6, Accuracy: 0.9221140472878998
# n_components = 7, Accuracy: 0.9193324061196105
# n_components = 8, Accuracy: 0.9304589707927677
# n_components = 9, Accuracy: 0.9388038942976356
# [0.29073265 0.182297   0.16743667 0.11687783 0.08385942 0.06484256
#  0.04311588 0.02968272 0.02115528]
# [0.29073265 0.47302965 0.64046632 0.75734415 0.84120357 0.90604612
#  0.949162   0.97884472 1.        ]
# 1.0



# 0.98515656


# n_components = 1, Accuracy: 0.33611111111111114
# n_components = 2, Accuracy: 0.5722222222222222
# n_components = 3, Accuracy: 0.775
# n_components = 4, Accuracy: 0.8722222222222222
# n_components = 5, Accuracy: 0.8916666666666667
# n_components = 6, Accuracy: 0.9166666666666666
# n_components = 7, Accuracy: 0.9277777777777778
# n_components = 8, Accuracy: 0.9361111111111111
# n_components = 9, Accuracy: 0.9472222222222222
# n_components = 10, Accuracy: 0.9527777777777777
# n_components = 11, Accuracy: 0.9666666666666667
# n_components = 12, Accuracy: 0.9611111111111111
# n_components = 13, Accuracy: 0.9694444444444444
# n_components = 14, Accuracy: 0.9638888888888889
# n_components = 15, Accuracy: 0.9722222222222222
# n_components = 16, Accuracy: 0.9722222222222222
# n_components = 17, Accuracy: 0.9694444444444444
# n_components = 18, Accuracy: 0.9722222222222222
# n_components = 19, Accuracy: 0.975
# n_components = 20, Accuracy: 0.9694444444444444
# n_components = 21, Accuracy: 0.9722222222222222
# n_components = 22, Accuracy: 0.9666666666666667
# n_components = 23, Accuracy: 0.9777777777777777
# n_components = 24, Accuracy: 0.975
# n_components = 25, Accuracy: 0.9777777777777777
# n_components = 26, Accuracy: 0.9694444444444444
# n_components = 27, Accuracy: 0.9722222222222222
# n_components = 28, Accuracy: 0.9777777777777777
# n_components = 29, Accuracy: 0.9777777777777777
# n_components = 30, Accuracy: 0.9694444444444444
# n_components = 31, Accuracy: 0.975
# n_components = 32, Accuracy: 0.9805555555555555
# n_components = 33, Accuracy: 0.9638888888888889
# n_components = 34, Accuracy: 0.9694444444444444
# n_components = 35, Accuracy: 0.975
# n_components = 36, Accuracy: 0.9777777777777777
# n_components = 37, Accuracy: 0.9777777777777777
# n_components = 38, Accuracy: 0.9777777777777777
# n_components = 39, Accuracy: 0.9805555555555555
# n_components = 40, Accuracy: 0.9833333333333333
# n_components = 41, Accuracy: 0.975
# n_components = 42, Accuracy: 0.975
# n_components = 43, Accuracy: 0.9722222222222222
# n_components = 44, Accuracy: 0.9722222222222222
# n_components = 45, Accuracy: 0.9777777777777777
# n_components = 46, Accuracy: 0.9666666666666667
# n_components = 47, Accuracy: 0.9777777777777777
# n_components = 48, Accuracy: 0.975


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