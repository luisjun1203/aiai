import numpy as np
from sklearn .datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
X, y = load_iris(return_X_y=True)


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55

for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        scores = cross_val_score(model, X_train, y_train, cv=kfold)
        print("==================", name, "=======================")
        print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))
        # 4. 평가, 예측
        y_predict = cross_val_predict(model, X_test, y_test ,cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print("cross_val_accuracy : ", acc)
        
    except:
        print(name, ' :은 바보멍충이!!')
        # continue


# ACC :  [0.95833333 0.95833333 1.         0.91666667 1.        ]
#  평균 ACC :  0.9667
# [2 1 1 0 2 2 2 2 1 1 1 2 1 1 0 2 2 0 0 0 1 0 0 0 2 1 0 0 1 2]
# [2 1 1 0 2 2 2 2 1 1 2 1 1 1 0 2 2 0 0 0 1 0 0 0 2 1 0 0 1 2]
# accracy score :  0.9333333333333333


# ACC :  [0.96666667 1.         0.96666667 1.         0.93333333]
#  평균 ACC :  0.9733


# ================== AdaBoostClassifier =======================
# ACC :  [0.95833333 0.95833333 0.95833333 1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.8333333333333334
# ================== BaggingClassifier =======================
# ACC :  [0.95833333 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.9833
# cross_val_accuracy :  0.8666666666666667
# ================== BernoulliNB =======================
# ACC :  [0.375      0.41666667 0.375      0.41666667 0.375     ]
#  평균 ACC :  0.3917
# cross_val_accuracy :  0.3333333333333333
# ================== CalibratedClassifierCV =======================
# ACC :  [0.83333333 0.83333333 0.91666667 1.         0.95833333]
#  평균 ACC :  0.9083
# cross_val_accuracy :  0.7666666666666667
# ================== CategoricalNB =======================
# ACC :  [0.33333333 0.375             nan        nan        nan]
#  평균 ACC :  nan
# CategoricalNB  :은 바보멍충이!!
# ClassifierChain  :은 바보멍충이!!
# ================== ComplementNB =======================
# ACC :  [0.66666667 0.66666667 0.625      0.66666667 0.66666667]
#  평균 ACC :  0.6583
# ComplementNB  :은 바보멍충이!!
# ================== DecisionTreeClassifier =======================
# ACC :  [0.95833333 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.9833
# cross_val_accuracy :  0.8333333333333334
# ================== DummyClassifier =======================
# ACC :  [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
#  평균 ACC :  0.3333
# cross_val_accuracy :  0.3333333333333333
# ================== ExtraTreeClassifier =======================
# ACC :  [0.95833333 0.91666667 0.91666667 0.875      0.91666667]
#  평균 ACC :  0.9167
# cross_val_accuracy :  0.9333333333333333
# ================== ExtraTreesClassifier =======================
# ACC :  [0.91666667 0.95833333 1.         0.95833333 1.        ]
#  평균 ACC :  0.9667
# cross_val_accuracy :  0.9
# ================== GaussianNB =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.8666666666666667
# ================== GaussianProcessClassifier =======================
# ACC :  [0.91666667 0.91666667 0.91666667 1.         1.        ]
#  평균 ACC :  0.95
# cross_val_accuracy :  0.8
# ================== GradientBoostingClassifier =======================
# ACC :  [0.95833333 0.95833333 1.         1.         0.95833333]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.8333333333333334
# ================== HistGradientBoostingClassifier =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.3333333333333333
# ================== KNeighborsClassifier =======================
# ACC :  [0.95833333 0.95833333 1.         0.95833333 1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.9
# ================== LabelPropagation =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.9
# ================== LabelSpreading =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.9
# ================== LinearDiscriminantAnalysis =======================
# ACC :  [0.95833333 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.9833
# cross_val_accuracy :  0.9333333333333333
# ================== LinearSVC =======================
# ACC :  [0.875      0.91666667 0.91666667 1.         0.95833333]
#  평균 ACC :  0.9333
# cross_val_accuracy :  0.7666666666666667
# ================== LogisticRegression =======================
# ACC :  [0.91666667 0.91666667 0.91666667 1.         1.        ]
#  평균 ACC :  0.95
# cross_val_accuracy :  0.7666666666666667
# ================== LogisticRegressionCV =======================
# ACC :  [0.91666667 0.95833333 1.         0.91666667 1.        ]
#  평균 ACC :  0.9583
# cross_val_accuracy :  0.9
# ================== MLPClassifier =======================
# ACC :  [0.83333333 0.95833333 0.91666667 1.         1.        ]
#  평균 ACC :  0.9417
# cross_val_accuracy :  0.8666666666666667
# MultiOutputClassifier  :은 바보멍충이!!
# ================== MultinomialNB =======================
# ACC :  [0.91666667 0.79166667 0.66666667 0.79166667 0.625     ]
#  평균 ACC :  0.7583
# MultinomialNB  :은 바보멍충이!!
# ================== NearestCentroid =======================
# ACC :  [0.91666667 0.875      0.91666667 1.         1.        ]
#  평균 ACC :  0.9417
# cross_val_accuracy :  0.8666666666666667
# ================== NuSVC =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.9
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# ================== PassiveAggressiveClassifier =======================
# ACC :  [0.875      0.83333333 0.91666667 0.91666667 0.95833333]
#  평균 ACC :  0.9
# cross_val_accuracy :  0.8333333333333334
# ================== Perceptron =======================
# ACC :  [0.58333333 0.70833333 0.91666667 0.70833333 0.83333333] 
#  평균 ACC :  0.75
# cross_val_accuracy :  0.8333333333333334
# ================== QuadraticDiscriminantAnalysis =======================
# ACC :  [0.95833333 0.95833333 1.         0.91666667 1.        ]
#  평균 ACC :  0.9667
# cross_val_accuracy :  0.8
# ================== RadiusNeighborsClassifier =======================
# ACC :  [0.54166667 0.5        0.54166667 0.58333333 0.5       ]
#  평균 ACC :  0.5333
# cross_val_accuracy :  0.5333333333333333
# ================== RandomForestClassifier =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.8666666666666667
# ================== RidgeClassifier =======================
# ACC :  [0.75       0.79166667 0.875      0.875      0.95833333]
#  평균 ACC :  0.85
# cross_val_accuracy :  0.7
# ================== RidgeClassifierCV =======================
# ACC :  [0.83333333 0.79166667 0.91666667 0.91666667 0.83333333]
#  평균 ACC :  0.8583
# cross_val_accuracy :  0.7333333333333333
# ================== SGDClassifier =======================
# ACC :  [0.79166667 0.79166667 1.         1.         0.95833333]
#  평균 ACC :  0.9083
# cross_val_accuracy :  0.6666666666666666
# ================== SVC =======================
# ACC :  [0.91666667 0.95833333 1.         1.         1.        ]
#  평균 ACC :  0.975
# cross_val_accuracy :  0.9
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!



































































































































