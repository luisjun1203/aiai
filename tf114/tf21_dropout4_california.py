from sklearn.datasets import fetch_california_housing
import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# 1. 데이터
X, y = fetch_california_housing(return_X_y=True)
print(X.shape, y.shape) # (20640, 8) (20640,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

sclaer = MaxAbsScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)

Xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])  # 수정
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])      # 수정

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), dtype=tf.float32, name='weights')
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

keep_prob = tf.compat.v1.placeholder(tf.float32)


w1 = tf.compat.v1.Variable(tf.random_normal([8,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(Xp, w1) + b1)

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)

# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,1]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias5')

# 2. 모델 구성
hypothesis = tf.compat.v1.matmul(layer4, w5) + b5  


print(y_train.shape)
print(y_test.shape)

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# 2. 모델 구성
# hypothesis = tf.compat.v1.matmul(Xp, w) + b  

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

# 3-2. 훈련

# 4. 평가


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([loss, train], feed_dict={Xp: X_train, yp: y_train, keep_prob:0.5})
        if step % 200 == 0:
            print(step, "loss : ", cost_val)
            
    y_pred = sess.run(hypothesis, feed_dict={Xp: X_test, yp: y_test, keep_prob: 1.0})
    test_loss = sess.run(loss, feed_dict={Xp: X_test, yp: y_test, keep_prob: 1.0})
    
    print('Loss: ', test_loss)
    print('R2 스코어: ', r2_score(y_test, y_pred))
    print('MSE: ', mean_squared_error(y_test, y_pred))

# Loss:  5.5004473
# R2 스코어:  -3.1587809250513104
# MSE:  5.500448186885338

# Loss:  0.6620978
# R2 스코어:  0.49940097036093667
# MSE:  0.6620976373985781

# Loss:  4.565814
# R2 스코어:  -2.4521228567401474
# MSE:  4.565814658300979