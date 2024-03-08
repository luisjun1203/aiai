from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler, LabelEncoder
from sklearn.utils import all_estimators
import warnings
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor

warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

lae = LabelEncoder()
y= lae.fit_transform(y)

n_splits= 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)   


parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}


# 2. 모델 구성
importance = [] #  모델이 특정 특성 수로 훈련되었을 때의 정확도(accuracy)를 포함

for i in range(X.shape[1], 0, -1):  # X.shape[1] : 열의수(특성), (시작, 끝, step)


    model = XGBClassifier()
    model.set_params(early_stopping_rounds = 10, **parameters)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='mlogloss' 
          )

# start_time = time.time()
    # model.fit(X_train, y_train)
# end_time = time.time()
# print("최적의 매개변수 : ", model.best_estimator_)

# print("최적의 파라미터 : ", model.best_params_)

# print('best_score : ', model.best_score_)
# print('model.score : ', model.score(X_test, y_test))

# y_predict = model.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

# y_pred_best = model.best_estimator_.predict(X_test)
# print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))

# print("걸린시간 : ", round(end_time - start_time, 2), "초")

    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    
    importance.append((i, acc))
    
    

    if i > 1:  
        
        fI = model.feature_importances_
        least_important = np.argmin(fI)     # fI 배열 내에서 가장 낮은 중요도 값을 가진 특성의 인덱스를 찾는다.
        X_train = np.delete(X_train, least_important, axis=1)
        X_test = np.delete(X_test, least_important, axis=1)
   
for n_features, acc in importance:
    print(f"특성: {n_features}, Accuracy: {acc:.4f}")   

best_feature_count, best_accuracy = max(importance, key=lambda x: x[1]) # max함수가 importance 리스트 X에 대해 X[1]값인 acc를 반환

print(f"가장 좋았던 정확도: {best_accuracy:.4f}, 해당할 때의 특성 수: {best_feature_count}") 




# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 1000, 'num_class': 30, 'objective': 'multi:softmax', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.8789466638618499
# model.score :  0.8838938418643076
# accuracy_score :  0.8838938418643076
# 최적튠 ACC :  0.8838938418643076
# 걸린시간 :  2476.08 초

# 특성: 54, Accuracy: 0.9592
# 특성: 53, Accuracy: 0.9583
# 특성: 52, Accuracy: 0.9581
# 특성: 51, Accuracy: 0.9594
# 특성: 50, Accuracy: 0.9600
# 특성: 49, Accuracy: 0.9588
# 특성: 48, Accuracy: 0.9567
# 특성: 47, Accuracy: 0.9570
# 특성: 46, Accuracy: 0.9537
# 특성: 45, Accuracy: 0.9546
# 특성: 44, Accuracy: 0.9551
# 특성: 43, Accuracy: 0.9490
# 특성: 42, Accuracy: 0.9501
# 특성: 41, Accuracy: 0.9461
# 특성: 40, Accuracy: 0.9475
# 특성: 39, Accuracy: 0.9414
# 특성: 38, Accuracy: 0.9425
# 특성: 37, Accuracy: 0.9445
# 특성: 36, Accuracy: 0.9240
# 특성: 35, Accuracy: 0.9256
# 특성: 34, Accuracy: 0.8937
# 특성: 33, Accuracy: 0.8851
# 특성: 32, Accuracy: 0.8864
# 특성: 31, Accuracy: 0.8877
# 특성: 30, Accuracy: 0.8911
# 특성: 29, Accuracy: 0.8927
# 특성: 28, Accuracy: 0.8819
# 특성: 27, Accuracy: 0.8842
# 특성: 26, Accuracy: 0.8838
# 특성: 25, Accuracy: 0.8812
# 특성: 24, Accuracy: 0.8660
# 특성: 23, Accuracy: 0.8659
# 특성: 22, Accuracy: 0.8659
# 특성: 21, Accuracy: 0.8660
# 특성: 20, Accuracy: 0.8671
# 특성: 19, Accuracy: 0.8419
# 특성: 18, Accuracy: 0.7727
# 특성: 17, Accuracy: 0.7732
# 특성: 16, Accuracy: 0.7722
# 특성: 15, Accuracy: 0.7145
# 특성: 14, Accuracy: 0.7138
# 특성: 13, Accuracy: 0.7108
# 특성: 12, Accuracy: 0.7104
# 특성: 11, Accuracy: 0.7027
# 특성: 10, Accuracy: 0.7013
# 특성: 9, Accuracy: 0.6966
# 특성: 8, Accuracy: 0.6928
# 특성: 7, Accuracy: 0.6864
# 특성: 6, Accuracy: 0.6866
# 특성: 5, Accuracy: 0.6832
# 특성: 4, Accuracy: 0.6817
# 특성: 3, Accuracy: 0.6792
# 특성: 2, Accuracy: 0.5197
# 특성: 1, Accuracy: 0.5197
# 가장 좋았던 정확도: 0.9600, 해당할 때의 특성 수: 50





