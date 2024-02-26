import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0
from keras.datasets import cifar10
from keras.applications import VGG16


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()
# ============================================================
# Total params: 14,777,098
# Trainable params: 62,410
# Non-trainable params: 14,714,688
# ============================================================

