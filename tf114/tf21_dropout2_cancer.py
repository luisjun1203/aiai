from sklearn.datasets import load_iris, load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
tf.compat.v1.set_random_seed(300)

# 1. 데이터
datasets = load_breast_cancer()
X_data = datasets.data
y_data = datasets.target.reshape(-1, 1)

scaler = MinMaxScaler()
X_data = scaler.fit_transform(X_data)

Xp = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 30])
yp = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
keep_prob = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([30, 19]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name='bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(Xp, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([19, 97]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name='bias2')
layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)

w3 = tf.compat.v1.Variable(tf.random_normal([97, 9]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name='bias3')
layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([9, 21]), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name='bias4')
layer4 = tf.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random_normal([21, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name='bias5')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(7001):
        cost_val, _ = sess.run([loss, train], feed_dict={Xp: X_data, yp: y_data, keep_prob: 0.5})
        if step % 20 == 0:
            print(step, "loss : ", cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={Xp: X_data, yp: y_data, keep_prob: 1.0})
    
    # print("훈련값 : ", hypo)
    # print("예측값 : ", pred)
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


# acc :  1.0




















