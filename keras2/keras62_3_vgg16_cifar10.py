import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)   # 2.9.0
from keras.datasets import cifar10
from keras.applications import VGG16
from sklearn.preprocessing import OneHotEncoder
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


(X_train, y_train), (X_test, y_test) = cifar10.load_data()



ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg16.trainable = False # 가중치 동결

# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(10, activation='softmax'))

# model.summary()


# ============================================================
# Total params: 14,777,098
# Trainable params: 62,410
# Non-trainable params: 14,714,688
# ============================================================


model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='swish'))   
model.add(Conv2D(97, (4, 4), activation='swish'))                         
model.add(Conv2D(210, (3, 3), activation='swish'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(312, activation='swish'))
model.add(Dropout(0.3))  
model.add(Dense(48, activation='swish'))
model.add(Dense(10, activation='softmax'))


# model.trainable = False
model. summary()

strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1, batch_size=1000,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()


results = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)


print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("accuracy_score : ", acc)

# vgg16
# loss 1.1603782176971436
# acc 0.6115999817848206
# 걸리시간 :  819.376 초

# cnn
# loss 0.731319785118103
# acc 0.7548999786376953
# 걸리시간 :  -114445.298 초
# accuracy_score :  0.7549