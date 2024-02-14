# keras 18_01 복사
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')

# 1.데이터

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# 2. 모델구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55

for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name, ' :은 바보멍충이!!')
        # continue


# LinearSVC accuracy: 0.9333
# Perceptron accuracy: 0.6667
# LogisticRegression accuracy: 0.9667
# KNeighborsClassifier accuracy: 0.9667
# DecisionTreeClassifier accuracy: 0.9667
# RandomForestClassifier accuracy: 0.9667


# AdaBoostClassifier 의 정답률은 :  1.0
# BaggingClassifier 의 정답률은 :  0.9666666666666667
# BernoulliNB 의 정답률은 :  0.3333333333333333
# CalibratedClassifierCV 의 정답률은 :  0.9
# CategoricalNB 의 정답률은 :  0.9666666666666667



# AdaBoostClassifier 의 정답률은 :  1.0
# BaggingClassifier 의 정답률은 :  0.9666666666666667
# BernoulliNB 의 정답률은 :  0.3333333333333333
# CalibratedClassifierCV 의 정답률은 :  0.9
# CategoricalNB 의 정답률은 :  0.9666666666666667
################################################################################## ClassifierChain  :은 바보멍충이!!
# ComplementNB 의 정답률은 :  0.6666666666666666
# DecisionTreeClassifier 의 정답률은 :  0.9666666666666667
# DummyClassifier 의 정답률은 :  0.3333333333333333
# ExtraTreeClassifier 의 정답률은 :  0.9666666666666667
# ExtraTreesClassifier 의 정답률은 :  0.9666666666666667
# GaussianNB 의 정답률은 :  0.9666666666666667
# GaussianProcessClassifier 의 정답률은 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률은 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률은 :  0.9666666666666667
# KNeighborsClassifier 의 정답률은 :  0.9666666666666667
# LabelPropagation 의 정답률은 :  0.9666666666666667
# LabelSpreading 의 정답률은 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률은 :  0.9666666666666667
# LinearSVC 의 정답률은 :  0.9333333333333333
# LogisticRegression 의 정답률은 :  0.9666666666666667
# LogisticRegressionCV 의 정답률은 :  0.9666666666666667
# MLPClassifier 의 정답률은 :  0.9333333333333333
######################################################## MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB 의 정답률은 :  0.9333333333333333
# NearestCentroid 의 정답률은 :  0.9666666666666667
# NuSVC 의 정답률은 :  0.9666666666666667
####################################################### OneVsOneClassifier  :은 바보멍충이!!
################################################### OneVsRestClassifier  :은 바보멍충이!!
################################################# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.8333333333333334
# Perceptron 의 정답률은 :  0.6666666666666666
# QuadraticDiscriminantAnalysis 의 정답률은 :  0.9666666666666667
# RadiusNeighborsClassifier 의 정답률은 :  0.9666666666666667
# RandomForestClassifier 의 정답률은 :  0.9666666666666667
# RidgeClassifier 의 정답률은 :  0.8
# RidgeClassifierCV 의 정답률은 :  0.8
# SGDClassifier 의 정답률은 :  0.9333333333333333
# SVC 의 정답률은 :  0.9666666666666667
####################################################### StackingClassifier  :은 바보멍충이!!
############################################# ###########VotingClassifier  :은 바보멍충이!!



