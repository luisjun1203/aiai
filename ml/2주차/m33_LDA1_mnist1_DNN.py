# m31_1에서 뽑은 4가지 결과로 4가지 모델을 맹그러

# 1. 70000, 154
# 2. 70000, 331
# 3. 70000, 486
# 4. 70000, 713
# 5. 70000, 784

# 시간과 성능을 체크한다.


from keras. models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


(X_train, y_train), (X_test, y_test) = mnist.load_data()       # _ : 이 자리에 y가 들어와야 하지만 비워둘거야

print(X_train.shape, X_test.shape)      # (60000, 28, 28) (10000, 28, 28)

# X = np.append(X_train, X_test, axis=0)
# X = np.concatenate([X_train, X_test], axis=0).reshape(-1, 28*28)       # list형식으로 바꿔줘야함

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

sts = StandardScaler()
X_train = sts.fit_transform(X_train)
X_test = sts.transform(X_test)

# n_components_list = [154, 331, 486, 713, 784]
# results = []

lda = LinearDiscriminantAnalysis()
X_train_pca = lda.fit_transform(X_train, y_train)
X_test_pca = lda.transform(X_test)


# PCA 적용

# print(X_train_pca.shape, X_test_pca.shape)                          # (70000, 28*28)

print(X_train_pca.shape)

model = Sequential()
model.add(Dense(19, input_shape = (9,)  ,activation='relu'))
model.add(Dense(97,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(10, activation='softmax'))

start = time.time()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train_pca, y_train, epochs=100, batch_size=5000, verbose=1)
end = time.time()
result = model.evaluate(X_test_pca, y_test)


print("acc : ", result[1])
print("걸린시간 : ", round(end-start, 3),"초")


# acc :  0.9075000286102295
# 걸린시간 :  2.354 초


# results.append((result[1], round(end - start, 3)))
    
# for idx, i in enumerate(results):
#     print("PCA = " ,n_components_list[idx] )
#     print("acc : ", i[0])
#     print("걸리시간 : ", i[1], "초")

# PCA =  154
# acc :  0.9591000080108643
# 걸리시간 :  5.448 초
# PCA =  331
# acc :  0.9488999843597412
# 걸리시간 :  11.523 초
# PCA =  486
# acc :  0.9462000131607056
# 걸리시간 :  20.162 초
# PCA =  713
# acc :  0.9361000061035156
# 걸리시간 :  27.685 초
# PCA =  784
# acc :  0.9373000264167786
# 걸리시간 :  35.629 초


# PCA =  154
# acc :  0.9541000127792358
# 걸리시간 :  3.039 초
# PCA =  331
# acc :  0.9503999948501587
# 걸리시간 :  3.288 초
# PCA =  486
# acc :  0.949999988079071
# 걸리시간 :  4.158 초
# PCA =  713
# acc :  0.9423999786376953
# 걸리시간 :  5.443 초
# PCA =  784
# acc :  0.9337000250816345
# 걸리시간 :  5.88 초
