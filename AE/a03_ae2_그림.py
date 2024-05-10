# PCA : 주성분 분석(차원축소)
# 비지도학습에서 y는X다.
import numpy as np
from keras.datasets import mnist
np.random.seed(3)
import tensorflow as tf
tf.random.set_seed(3)


#1. 데이터
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(60000, 28*28).astype('float32')/255.
X_test = X_test.reshape(10000, 28*28).astype('float32')/255.


# print(np.max(X_train), np.min(X_test))    # 1.0 0.0


###노이즈 추가###                               # 평균0, 표준편차 0.1인 정규분포
X_train_noised = X_train + np.random.normal(0, 0.1, size=X_train.shape)
X_test_noised = X_test + np.random.normal(0, 0.1, size=X_test.shape)
# 이렇게 노이즈를 추가하게되면 스케일링해준 범위에 변동이 생기기때문에 1보다 큰값을 가진 놈들은 1로 설정

# print(X_train_noised.shape, X_test_noised.shape)    # (60000, 784) (10000, 784)
# print(np.max(X_train_noised), np.min(X_train_noised))    # 1.51010011258835 -0.5412108531082793
# print(np.max(X_test_noised), np.min(X_test_noised))    # 1.4547846753820015 -0.4760749951002331


X_train_noised = np.clip(X_train_noised, a_min=0, a_max=1)
X_test_noised = np.clip(X_test_noised, a_min=0, a_max=1)
# print(np.max(X_train_noised), np.min(X_train_noised))    # 1.0 0.0
# print(np.max(X_test_noised), np.min(X_test_noised))    # 1.0 0.0


# 2.모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

hidden_size = 713   # PCA 1.0일때 성능
# hidden_size = 486   # PCA 0.999일때 성능
# hidden_size = 331   # PCA 0.99일때 성능
# hidden_size = 154   # PCA 0.95일때 성능


model = autoencoder(hidden_layer_size=hidden_size)


model.summary()
# Total params: 101,200

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse', )
model.compile(optimizer='adam', loss='binary_crossentropy', )

model.fit(X_train_noised, X_train, epochs=30, batch_size=128, validation_split=0.2)

# 4.평가, 예측
decoded_imgs = model.predict(X_test_noised)
# evaluate는 지표를 신뢰하기 힘듬



# 그림 그리자 (생각없이 쳐라)
import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
          plt.subplots(3, 5, figsize=(20, 7))
          
          
# 이미지를 다섯개를 무작위로 고른다.
random_images = random.sample(range(decoded_imgs.shape[0]),5)          

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(X_test[random_images[i]].reshape(28, 28), cmap='gray')
    if  i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(X_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if  i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 풀력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if  i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
