import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings ('ignore')
#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

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


# ================== AdaBoostClassifier =======================
# ACC :  [0.97101449 0.94117647 0.89705882 0.98529412 0.95588235]
#  평균 ACC :  0.9501
# cross_val_accuracy :  0.9649122807017544
# ================== BaggingClassifier =======================
# ACC :  [0.97101449 0.94117647 0.91176471 0.94117647 0.95588235]
#  평균 ACC :  0.9442
# cross_val_accuracy :  0.956140350877193
# ================== BernoulliNB =======================
# ACC :  [0.49275362 0.47058824 0.54411765 0.45588235 0.47058824]
#  평균 ACC :  0.4868
# cross_val_accuracy :  0.7587719298245614
# ================== CalibratedClassifierCV =======================
# ACC :  [1.         0.95588235 0.95588235 0.97058824 0.95588235]
#  평균 ACC :  0.9676
# cross_val_accuracy :  0.9692982456140351
# ================== CategoricalNB =======================
# ACC :  [nan nan nan nan nan]
#  평균 ACC :  nan
# CategoricalNB  :은 바보멍충이!!
# ClassifierChain  :은 바보멍충이!!
# ================== ComplementNB =======================
# ACC :  [0.84057971 0.85294118 0.79411765 0.86764706 0.82352941]
#  평균 ACC :  0.8358
# ComplementNB  :은 바보멍충이!!
# ================== DecisionTreeClassifier =======================
# ACC :  [0.92753623 0.91176471 0.89705882 0.92647059 0.94117647]
#  평균 ACC :  0.9208
# cross_val_accuracy :  0.9473684210526315
# ================== DummyClassifier =======================
# ACC :  [0.53623188 0.52941176 0.54411765 0.54411765 0.54411765]
#  평균 ACC :  0.5396
# cross_val_accuracy :  0.7587719298245614
# ================== ExtraTreeClassifier =======================
# ACC :  [0.89855072 0.94117647 0.86764706 0.88235294 0.89705882]
#  평균 ACC :  0.8974
# cross_val_accuracy :  0.9298245614035088
# ================== ExtraTreesClassifier =======================
# ACC :  [1.         0.94117647 0.95588235 0.98529412 0.94117647]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9649122807017544
# ================== GaussianNB =======================
# ACC :  [0.94202899 0.91176471 0.94117647 0.94117647 0.95588235]
#  평균 ACC :  0.9384
# cross_val_accuracy :  0.956140350877193
# ================== GaussianProcessClassifier =======================
# ACC :  [0.98550725 0.92647059 0.95588235 0.98529412 0.97058824]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9649122807017544
# ================== GradientBoostingClassifier =======================
# ACC :  [0.98550725 0.91176471 0.89705882 0.95588235 0.94117647]
#  평균 ACC :  0.9383
# cross_val_accuracy :  0.9517543859649122
# ================== HistGradientBoostingClassifier =======================
# ACC :  [1.         0.92647059 0.91176471 0.95588235 0.94117647]
#  평균 ACC :  0.9471
# cross_val_accuracy :  0.9649122807017544
# ================== KNeighborsClassifier =======================
# ACC :  [1.         0.91176471 0.95588235 0.97058824 0.97058824]
#  평균 ACC :  0.9618
# cross_val_accuracy :  0.9736842105263158
# ================== LabelPropagation =======================
# ACC :  [0.98550725 0.92647059 0.95588235 0.97058824 0.97058824]
#  평균 ACC :  0.9618
# cross_val_accuracy :  0.9649122807017544
# ================== LabelSpreading =======================
# ACC :  [0.98550725 0.92647059 0.95588235 0.97058824 0.97058824]
#  평균 ACC :  0.9618
# cross_val_accuracy :  0.9649122807017544
# ================== LinearDiscriminantAnalysis =======================
# ACC :  [0.95652174 0.95588235 0.94117647 0.94117647 0.95588235]
#  평균 ACC :  0.9501
# cross_val_accuracy :  0.9692982456140351
# ================== LinearSVC =======================
# ACC :  [1.         0.97058824 0.94117647 0.98529412 0.95588235]
#  평균 ACC :  0.9706
# cross_val_accuracy :  0.9824561403508771
# ================== LogisticRegression =======================
# ACC :  [0.98550725 0.92647059 0.95588235 0.98529412 0.97058824]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9692982456140351
# ================== LogisticRegressionCV =======================
# ACC :  [0.98550725 0.94117647 0.95588235 0.98529412 0.95588235]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9824561403508771
# ================== MLPClassifier =======================
# ACC :  [0.98550725 0.94117647 0.92647059 0.98529412 0.95588235]
#  평균 ACC :  0.9589
# cross_val_accuracy :  0.956140350877193
# MultiOutputClassifier  :은 바보멍충이!!
# ================== MultinomialNB =======================
# ACC :  [0.91304348 0.89705882 0.85294118 0.82352941 0.83823529]
#  평균 ACC :  0.865
# MultinomialNB  :은 바보멍충이!!
# ================== NearestCentroid =======================
# ACC :  [0.92753623 0.91176471 0.92647059 0.94117647 0.91176471]
#  평균 ACC :  0.9237
# cross_val_accuracy :  0.9736842105263158
# ================== NuSVC =======================
# ACC :  [0.95652174 0.91176471 0.95588235 0.95588235 0.95588235]
#  평균 ACC :  0.9472
# NuSVC  :은 바보멍충이!!
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# ================== PassiveAggressiveClassifier =======================
# ACC :  [0.98550725 0.94117647 0.94117647 0.95588235 0.95588235]
#  평균 ACC :  0.9559
# cross_val_accuracy :  0.9605263157894737
# ================== Perceptron =======================
# ACC :  [0.95652174 0.92647059 0.92647059 0.97058824 0.95588235]
#  평균 ACC :  0.9472
# cross_val_accuracy :  0.9736842105263158
# ================== QuadraticDiscriminantAnalysis =======================
# ACC :  [0.95652174 0.95588235 0.89705882 0.97058824 0.95588235]
#  평균 ACC :  0.9472
# cross_val_accuracy :  0.956140350877193
# ================== RadiusNeighborsClassifier =======================
# ACC :  [0.86956522 0.92647059        nan        nan        nan]
#  평균 ACC :  nan
# RadiusNeighborsClassifier  :은 바보멍충이!!
# ================== RandomForestClassifier =======================
# ACC :  [0.98550725 0.94117647 0.95588235 0.95588235 0.94117647]
#  평균 ACC :  0.9559
# cross_val_accuracy :  0.9517543859649122
# ================== RidgeClassifier =======================
# ACC :  [0.95652174 0.95588235 0.94117647 0.97058824 0.97058824]
#  평균 ACC :  0.959
# cross_val_accuracy :  0.9692982456140351
# ================== RidgeClassifierCV =======================
# ACC :  [0.98550725 0.95588235 0.94117647 0.97058824 0.97058824]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9736842105263158
# ================== SGDClassifier =======================
# ACC :  [0.98550725 0.94117647 0.94117647 0.98529412 0.92647059]
#  평균 ACC :  0.9559
# cross_val_accuracy :  0.9780701754385965
# ================== SVC =======================
# ACC :  [0.98550725 0.95588235 0.95588235 0.95588235 0.97058824]
#  평균 ACC :  0.9647
# cross_val_accuracy :  0.9780701754385965
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!