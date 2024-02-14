# keras 18_01 복사
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

class CustomXGBClassfier(XGBClassifier):        # XGBClassfier을 상속받겠다.
    def __str__(self):
        return 'XGBClassifier()'                

aaa = CustomXGBClassfier()
# aaa는 인스턴스

# 1.데이터

# X, y = load_iris(return_X_y=True)
# print(X.shape, y.shape) #(150, 4) (150,)

datasets = load_iris()
X = datasets.data
y = datasets['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# 2. 모델구성

model1 = DecisionTreeClassifier(random_state=3)
model2 = RandomForestClassifier(random_state=3)
model3 = GradientBoostingClassifier(random_state=3)
model4 = aaa

models = [ model1, model2, model3, model4]
models = [DecisionTreeClassifier(),
          RandomForestClassifier(),
           GradientBoostingClassifier(),
           aaa    
]

# for model in models:
#     model_name = model.__class__.__name__
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"{model_name} accuracy: {accuracy:.4f}")
#     print(model, ":", model.feature_importances_)




# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     colors = plt.cm.inferno(np.linspace(0, 1, n_features))           # plt.cm 에서 색상종류 가져옴(plasma, inferno, magma, cividis, viridis 등등)
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center', color = colors)
#     plt.yticks(np.arange(n_features), datasets.feature_names)       # y축의 눈금
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)                                        # y축의 범위
#     plt.title(model)
  
def train_and_get_feature_importances(model, X_train, X_test, y_train, y_test, datasets):
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    return model.feature_importances_

feature_importances = []

for model in models:
    feature_importances.append(train_and_get_feature_importances(model, X_train, X_test, y_train, y_test, load_iris()))

n_features = X.shape[1]
colors = plt.cm.inferno(np.linspace(0, 1, n_features))

plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    plt.subplot(2, 2, i+1)
    plt.barh(np.arange(n_features), feature_importances[i], align='center', color=colors)
    plt.yticks(np.arange(n_features), load_iris().feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

plt.tight_layout()
plt.show()
    
# plt.figure(figsize=(15,9))     
# plt.subplot(2, 2, 1)
# plot_feature_importances_dataset(model1)

# plt.subplot(2, 2, 2)
# plot_feature_importances_dataset(model2)

# plt.subplot(2, 2, 3)
# plot_feature_importances_dataset(model3)

# plt.subplot(2, 2, 4)
# plot_feature_importances_dataset(model4)



# DecisionTreeClassifier accuracy: 0.9667
# DecisionTreeClassifier : [0.         0.01666667 0.57407537 0.40925796]
# RandomForestClassifier accuracy: 0.9667
# RandomForestClassifier : [0.09643216 0.02975015 0.42475327 0.44906443]
# GradientBoostingClassifier accuracy: 0.9667
# GradientBoostingClassifier : [0.00986647 0.01429541 0.61321158 0.36262654]
# XGBClassifier accuracy: 0.9667
# XGBClassifier : [0.0122154  0.02133699 0.73724246 0.22920507]


