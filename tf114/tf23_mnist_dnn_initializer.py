import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# from keras.utils import np_utils
import numpy as np
from sklearn.metrics import accuracy_score

# 1. 데이터
(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(X_train.shape, X_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

X_train = X_train.reshape(60000, 28*28).astype('float32')/255
X_test = X_test.reshape(10000, 28*28).astype('float32')/255

###################### [실습] 맹그러봐 !!!!! #########################################

rate = tf.compat.v1.placeholder(tf.float32)
# print(X_train.shape, X_test.shape)  # (60000, 784) (10000, 784)
# keep_prob = tf.compat.v1.placeholder(tf.float32)

learning_rate = 1e-3 

X = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,784])
y = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,10])

w1 = tf.compat.v1.get_variable('w1', shape = [784,128],
                               initializer=tf.contrib.layers.xavier_initializer()   # # 가중치 초기화
                               )    
b1 = tf.compat.v1.Variable(tf.zeros([128]), name = 'b1')
layer1 = tf.compat.v1.matmul(X, w1) + b1
layer1 = tf.compat.v1.nn.dropout(layer1, rate = rate)


w2 = tf.compat.v1.get_variable('w2', shape = [128, 64],
                               initializer=tf.contrib.layers.xavier_initializer()
                               )    
b2 = tf.compat.v1.Variable(tf.zeros([64]), name = 'b2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.relu(layer2)
layer2 = tf.compat.v1.nn.dropout(layer2, rate = rate)

w3 = tf.compat.v1.get_variable('w3', shape = [64, 32],
                               initializer=tf.contrib.layers.xavier_initializer()
                               )
b3 = tf.compat.v1.Variable(tf.zeros([32]), name = 'b3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)
dropout3 = tf.compat.v1.nn.dropout(layer3, rate =rate)

w4 = tf.compat.v1.get_variable('w4', shape = [32, 10])
b4 = tf.compat.v1.Variable(tf.zeros([10]), name = 'b4')
layer4 = tf.compat.v1.matmul(layer3, w4) + b4

# 2. 모델 구성

hypothesis = tf.compat.v1.nn.softmax(layer4)

# 3-1. 컴파일

# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.math.log_softmax(hypothesis), axis=1))
loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs = 10001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], feed_dict={X: X_train, y: y_train, rate: 0.3})
    if step % 20 ==0:
        print(step, 'loss : ', cost_val)

# 4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={X:X_test, rate: 0})
y_predict_arg = sess.run(tf.math.argmax(y_predict, 1))
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict_arg)
print("ACC : ", acc)

sess.close()
# ACC: 0.9018

# acc :  0.9266

# acc :  0.9456   

# acc :  0.9516

# 10000 epochs
# ACC :  0.9679