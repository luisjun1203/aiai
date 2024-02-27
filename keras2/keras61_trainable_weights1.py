import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

# 1 . 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


# 2 . 모델

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

# print(model.weights)    # 가중치의 초기값
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]]
# kernel : 가중치
print("=================================================================")
# print(model.trainable_weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.1723156 ,  0.5125139 ],
#        [ 0.41434443, -0.8537577 ],
#        [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[ 1.1585606],
#        [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
# =================================================================
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.1723156 ,  0.5125139 ],
#        [ 0.41434443, -0.8537577 ],
#        [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[ 1.1585606],
#        [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
print("=================================================================")
# print(model.weights)                  # 6    
# print(len(model.trainable_weights))   # 6
################################################################################
model.trainable = False # ★★★
#################################################################################
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0
print("=================================================================")
print(model.weights)                  # 6    
print(model.trainable_weights)   # 6
print("=================================================================")
model.summary()
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17







