import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


# 1. 데이터
X,y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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



# 3. 훈련
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='logloss' 
          )


# 4. 평가, 예측
# result = model.score(X_test, y_test)
# print("최종점수 : ", result)

    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    # print("acc_score ", acc)
    importance.append((i, acc))
    
    # print(f"특성 개수: {i}, 정확도: {acc:.4f}")
##################################################################
# print(model.feature_importances_)

# print(len(model.feature_importances_))
# for문을 사용해서 피처가 약한놈부터 하나씩 제거
# 30, 29, 28, 27, ...
# for i in range(len(model.feature_importances_)):
    if i > 1:  
        
        fI = model.feature_importances_
        least_important = np.argmin(fI)     # fI 배열 내에서 가장 낮은 중요도 값을 가진 특성의 인덱스를 찾는다.
        X_train = np.delete(X_train, least_important, axis=1)
        X_test = np.delete(X_test, least_important, axis=1)
   
for n_features, acc in importance:
    print(f"특성: {n_features}, Accuracy: {acc:.4f}")   

best_feature_count, best_accuracy = max(importance, key=lambda x: x[1]) # max함수가 importance 리스트 X에 대해 X[1]값인 acc를 반환

print(f"가장 좋았던 정확도: {best_accuracy:.4f}, 해당할 때의 특성 수: {best_feature_count}")    

# 특성: 30, Accuracy: 0.9386
# 특성: 29, Accuracy: 0.9386
# 특성: 28, Accuracy: 0.9386
# 특성: 27, Accuracy: 0.9386
# 특성: 26, Accuracy: 0.9561
# 특성: 25, Accuracy: 0.9386
# 특성: 24, Accuracy: 0.9386
# 특성: 23, Accuracy: 0.9561
# 특성: 22, Accuracy: 0.9561
# 특성: 21, Accuracy: 0.9298
# 특성: 20, Accuracy: 0.9474
# 특성: 19, Accuracy: 0.9474
# 특성: 18, Accuracy: 0.9386
# 특성: 17, Accuracy: 0.9386
# 특성: 16, Accuracy: 0.9561
# 특성: 15, Accuracy: 0.9561
# 특성: 14, Accuracy: 0.9474
# 특성: 13, Accuracy: 0.9474
# 특성: 12, Accuracy: 0.9474
# 특성: 11, Accuracy: 0.9474
# 특성: 10, Accuracy: 0.9474
# 특성: 9, Accuracy: 0.9474
# 특성: 8, Accuracy: 0.9386
# 특성: 7, Accuracy: 0.9386
# 특성: 6, Accuracy: 0.9386
# 특성: 5, Accuracy: 0.9211
# 특성: 4, Accuracy: 0.9211
# 특성: 3, Accuracy: 0.9211
# 특성: 2, Accuracy: 0.9211
# 특성: 1, Accuracy: 0.9035
# 가장 좋았던 정확도: 0.9561, 해당할 때의 특성 수: 26
    



