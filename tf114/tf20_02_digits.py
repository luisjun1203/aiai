import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

# 1 데이터
datasets = load_digits()

 
# print(X_data.shape) # (1797, 64)
# print(y_data.shape) # (1797,)

X_data = datasets.data
y_data = datasets.target.reshape(-1, 1)  

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data)

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
# w = tf.compat.v1.Variable(tf.random_normal([64,10]), name='weight')
# b = tf.compat.v1.Variable(tf.zeros([10]), name='bias')


w1 = tf.compat.v1.Variable(tf.random_normal([64,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(X, w1) + b1)

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)


# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,10]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([10]), name = 'bias5')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)

# hypothesis = tf.nn.softmax(tf.compat.v1.matmul(X, w) + b)

# 3-1. 컴파일

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# Categorical_Cross_Entropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)





# 3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={X: X_data_scaled, y: y_data_ohe})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)

y_predict_ohe = sess.run(hypothesis, feed_dict={X: X_data_scaled})
y_predict = np.argmax(y_predict_ohe, axis=1)
original_y_data = np.argmax(y_data_ohe, axis=1)  

acc = accuracy_score(original_y_data, y_predict)
print("ACC : ", acc)

sess.close()


# [0 1 2 ... 8 9 8]
# [0 1 2 ... 8 9 8]
# ACC :  1.0

# ACC :  0.993322203672788






