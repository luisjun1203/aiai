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

# w = tf.compat.v1.Variable( tf.compat.v1.random_normal([30,1]) ,dtype=tf.float32 , name='weight'  )
# b = tf.compat.v1.Variable( tf.compat.v1.zeros([1]) ,dtype=tf.float32 , name='bias'  )

w1 = tf.compat.v1.Variable(tf.random_normal([30,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.relu(tf.compat.v1.matmul(Xp, w1) + b1)

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1, w2) + b2)


# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3)

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,1]), name = 'weight4')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias4')
# layer5 =tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)


#2 모델
hypothesis = tf.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

#3-1 컴파일
loss = -tf.reduce_mean(yp*tf.log(hypothesis)+(1 - yp)*tf.log(1-hypothesis) )

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)


#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(20000):
    _, loss_val = sess.run([train, loss], feed_dict={Xp: X, yp: y_data})
    if step % 10 == 0:
        print(step, '\t', loss_val)

# 평가
y_pred = sess.run(hypothesis, feed_dict={Xp: X})
y_pred = np.round(y_pred)
acc = accuracy_score(y_data, y_pred)
print('Accuracy:', acc)

# Accuracy: 1.0


