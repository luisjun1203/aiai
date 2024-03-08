import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error




# 1. 데이터
X,y = load_digits(return_X_y=True)

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
            eval_metric ='mlogloss' 
          )


# 4. 평가, 예측


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
    
    
# 특성: 64, Accuracy: 0.9694
# 특성: 63, Accuracy: 0.9667
# 특성: 62, Accuracy: 0.9667
# 특성: 61, Accuracy: 0.9694
# 특성: 60, Accuracy: 0.9694
# 특성: 59, Accuracy: 0.9639
# 특성: 58, Accuracy: 0.9667
# 특성: 57, Accuracy: 0.9722
# 특성: 56, Accuracy: 0.9722
# 특성: 55, Accuracy: 0.9722
# 특성: 54, Accuracy: 0.9694
# 특성: 53, Accuracy: 0.9722
# 특성: 52, Accuracy: 0.9722
# 특성: 51, Accuracy: 0.9694
# 특성: 50, Accuracy: 0.9694
# 특성: 49, Accuracy: 0.9722
# 특성: 48, Accuracy: 0.9667
# 특성: 47, Accuracy: 0.9694
# 특성: 46, Accuracy: 0.9667
# 특성: 45, Accuracy: 0.9694
# 특성: 44, Accuracy: 0.9639
# 특성: 43, Accuracy: 0.9750
# 특성: 42, Accuracy: 0.9667
# 특성: 41, Accuracy: 0.9694
# 특성: 40, Accuracy: 0.9750
# 특성: 39, Accuracy: 0.9694
# 특성: 38, Accuracy: 0.9750
# 특성: 37, Accuracy: 0.9750
# 특성: 36, Accuracy: 0.9639
# 특성: 35, Accuracy: 0.9583
# 특성: 34, Accuracy: 0.9639
# 특성: 33, Accuracy: 0.9583
# 특성: 32, Accuracy: 0.9611
# 특성: 31, Accuracy: 0.9556
# 특성: 30, Accuracy: 0.9611
# 특성: 29, Accuracy: 0.9667
# 특성: 28, Accuracy: 0.9556
# 특성: 27, Accuracy: 0.9556
# 특성: 26, Accuracy: 0.9611
# 특성: 25, Accuracy: 0.9583
# 특성: 24, Accuracy: 0.9611
# 특성: 23, Accuracy: 0.9694
# 특성: 22, Accuracy: 0.9583
# 특성: 21, Accuracy: 0.9556
# 특성: 20, Accuracy: 0.9611
# 특성: 19, Accuracy: 0.9528
# 특성: 18, Accuracy: 0.9444
# 특성: 17, Accuracy: 0.9583
# 특성: 16, Accuracy: 0.9528
# 특성: 15, Accuracy: 0.9472
# 특성: 14, Accuracy: 0.9444
# 특성: 13, Accuracy: 0.9306
# 특성: 12, Accuracy: 0.9194
# 특성: 11, Accuracy: 0.9167
# 특성: 10, Accuracy: 0.9056
# 특성: 9, Accuracy: 0.9139
# 특성: 8, Accuracy: 0.8667
# 특성: 7, Accuracy: 0.8000
# 특성: 6, Accuracy: 0.7472
# 특성: 5, Accuracy: 0.7278
# 특성: 4, Accuracy: 0.6333
# 특성: 3, Accuracy: 0.5389
# 특성: 2, Accuracy: 0.3722
# 특성: 1, Accuracy: 0.2389
# 가장 좋았던 정확도: 0.9750, 해당할 때의 특성 수: 43







