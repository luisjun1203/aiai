import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

from keras.applications import VGG16

# model = VGG16()
# 디폴트 :  include_top = True, input_shape = (224, 224, 3)
#   ===================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
#   ====================================================================

model = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(32, 32, 3))
model.summary()
#   ====================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
#   ====================================================================

# model.compile(loss='categorical_cossentropy', optimizer='adam')

# model.fit()

##################### include_top = False ###############################
#1. FC layer 날려
#2. inpt_shape 내가 하고싶은걸로 해 !!!!!!!



































