# xgboost와 그리드서치,  랜덤서치, halving 등을 사용
from keras. models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV, HalvingRandomSearchCV
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

(X_train, y_train), (X_test, y_test) = mnist.load_data()       # _ : 이 자리에 y가 들어와야 하지만 비워둘거야

print(X_train.shape, X_test.shape)      # (60000, 28, 28) (10000, 28, 28)

# X = np.append(X_train, X_test, axis=0)
# X = np.concatenate([X_train, X_test], axis=0).reshape(-1, 28*28)       # list형식으로 바꿔줘야함
a = 3
splits = 3
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=a)



X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)



n_components_list = [154, 331, 486, 713, 784]
results = []
for n_components in n_components_list:
    print(f"n_components: {n_components}")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    start = time.time()
    parameters = {
        'XGB__n_estimators': [100],  # 부스팅 라운드의 수
        'XGB__learning_rate': [0.1],  # 학습률
        'XGB__max_depth': [9],  # 트리의 최대 깊이
        'XGB__min_child_weight': [1],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
        'XGB__gamma': [1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
        'XGB__subsample': [0.8],  # 각 트리마다의 관측 데이터 샘플링 비율
        'XGB__colsample_bytree': [0.8],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
        'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
        'XGB__num_class': [10],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
        'XGB__verbosity' : [0]
    }
  

    pipe = Pipeline([('SS', StandardScaler()),
                    ('XGB', XGBClassifier(random_state=a, n_jobs = -1, tree_method = 'gpu_hist', predictor = 'gpu_predictor',  gpu_id=0))])

    model = HalvingRandomSearchCV(pipe, parameters,
                        cv = kfold,
                        verbose=1,
                        refit=True,
                        n_jobs=-1,   
                        # n_iter=10 # 디폴트 10
                        factor=2,
                        min_resources=70
                        )

    model.fit(X_train_pca, y_train)

    end = time.time()
    print("최적의 매개변수:",model.best_estimator_)
    print("최적의 파라미터:",model.best_params_)
    print("best_score:",model.best_score_) 
    print("model.score:", model.score(X_test_pca,y_test)) 

    # y_predict=model.predict(X_test_pca)
    # print("acc.score:", accuracy_score(y_test,y_predict))
    # y_pred_best=model.best_estimator_.predict(X_test_pca)

    # print("best_acc.score:",accuracy_score(y_test,y_pred_best))
    print("걸린시간 : ", round(end - start, 3),"초")
    result = model.score(X_test_pca, y_test)
    results.append((result, round(end - start, 3)))
    
for idx, i in enumerate(results):
    print("PCA = " ,n_components_list[idx] )
    print("acc : ", i[0])
    print("걸리시간 : ", i[1], "초")

# 최적의 파라미터: {'XGB__verbosity': 0, 'XGB__subsample': 0.8, 'XGB__objective': 'multi:softmax',
#            'XGB__num_class': 10, 'XGB__n_estimators': 100, 'XGB__min_child_weight': 1,
#            'XGB__max_depth': 9, 'XGB__learning_rate': 0.1, 'XGB__gamma': 1, 'XGB__colsample_bytree': 0.8}





# PCA =  154
# acc :  0.9149
# 걸리시간 :  237.859 초
# PCA =  331
# acc :  0.8757
# 걸리시간 :  248.633 초
# PCA =  486
# acc :  0.9564
# 걸리시간 :  282.334 초
# PCA =  713
# acc :  0.8775
# 걸리시간 :  271.734 초
# PCA =  784
# acc :  0.9565
# 걸리시간 :  271.367 초
# 최적의 파라미터: {'XGB__verbosity': 0, 'XGB__subsample': 0.8, 'XGB__objective': 'multi:softmax',
#            'XGB__num_class': 10, 'XGB__n_estimators': 100, 'XGB__min_child_weight': 1,
#            'XGB__max_depth': 9, 'XGB__learning_rate': 0.1, 'XGB__gamma': 1, 'XGB__colsample_bytree': 0.8}



# PCA =  154
# acc :  0.9591
# 걸리시간 :  12.933 초
# PCA =  331
# acc :  0.9559
# 걸리시간 :  20.636 초
# PCA =  486
# acc :  0.9582
# 걸리시간 :  26.711 초
# PCA =  713
# acc :  0.9567
# 걸리시간 :  34.763 초
# PCA =  784
# acc :  0.9554
# 걸리시간 :  35.688 초



