# 드랍아웃을 적용했으나
# 훈련(0.5)과 평가(1,0)을 아직 분리하지 않았다.
from sklearn.datasets import load_iris , load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler , MaxAbsScaler
tf.compat.v1.set_random_seed(300)

#1. 데이터
datasets = load_breast_cancer()
X_data = datasets.data
y_data = datasets.target.reshape(-1,1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X_data)

# X , y  = datasets.data , datasets.target  # 위에랑 똑같다

# print(X,X.shape)        # (569, 30)

# y_data = y_data.reshape(-1,1)     # (569, 1)

# print(y_data.shape)

Xp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,30])
yp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,1])


w1 = tf.compat.v1.Variable(tf.random_normal([30,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(X, w1) + b1)

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=0.5)


# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)
layer3 = tf.compat.v1.nn.dropout(layer3, keep_prob=0.5)

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)
layer4 = tf.compat.v1.nn.dropout(layer4, keep_prob=0.5)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,1]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias5')
# layer5 =tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)
   
# 3-1. 컴파일

loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))   # 'binary_crossentropy'

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

# 3-2. 훈련


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ = sess.run([loss, train], feed_dict={X:X_data, yp:y_data})
    if step % 20 ==0:
        print(step, "loss : ", cost_val)
        
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], feed_dict = {X:X_data, yp:y_data})
    
    print("훈련값 : ", hypo)
    print("예측값 : ", pred)
    print("acc : ", acc)
    
    
    
    
# 2000 loss :  0.056130864
# 훈련값 :  [[4.2325419e-06]
#  [9.0514266e-01]
#  [9.8585969e-01]
#  [1.0459811e-01]]
# 예측값 :  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# acc :  1.0























