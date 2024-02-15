#train_test_split 전 스케일링 후 PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# print(sk.__version__)       # 1.1.3

datasets = load_iris()
X = datasets['data']
y = datasets['target']

# print(X.shape, y.shape)     # (150, 4) (150, )

sts = StandardScaler()
X = sts.fit_transform(X)


pca = PCA(n_components=1)   
X = pca.fit_transform(X)
# print(X)                    
# print(X.shape)              # (150, 2)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, stratify=y)


model = RandomForestClassifier()

model.fit(X_train, y_train)

results = model.score(X_test, y_test)

print("model.score : ", results)

# y_predict = model.predict(X_test)


# n_components=3
# model.score :  0.8947368421052632


# n_components=2
# model.score :  0.8947368421052632


# n_components=1
# model.score :  0.9210526315789473






