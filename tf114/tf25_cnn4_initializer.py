import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

learning_rate = 1e-3
rate = tf.compat.v1.placeholder(tf.float32)

# 1.데이터
(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')/255.0
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')/255.


# 2. 모델구성
X = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])     # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64], initializer=tf.contrib.layers.xavier_initializer())   # kernel_size
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)
                                
with tf.compat.v1.Session() as sess:
    sess.run(tf.v1.global_variables_initializer())
    w1_val = sess.run(w1)
    print(w1_val, '\n', w1_val.shape)                                                               
                                            
                                        
