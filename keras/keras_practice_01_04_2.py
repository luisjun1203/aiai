import warnings                                         #   경고 무시
warnings.filterwarnings('ignore')                       #   경고 무시
from keras.models import Sequential             
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score                    
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. 데이터

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

# print(X.shape)      #(20640, 8)
# print(y.shape)      #(20640,  )

# print(datasets.feature_names)   # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)



# 2.모델구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(X_train, y_train, epochs=50, batch_size=100)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
result = model.predict([X])
print("R2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 3), "초")


plt.scatter(X[:,:1], y)
plt.plot(X[:,:1], result, color='pink')
plt.show()
















