import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score


# 1. 데이터

X_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],  # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],  # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],  # 0
          [1,0,0]]


X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]), name='bias')

# 2. 모델

hypothesis = tf.compat.v1.matmul(X, w) + b

# 3. 컴파일 
 
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis- y)) 

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)



# [실습]
# 맹그러!!

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 35000

for step in range(epochs):
    loss_val, _ = sess.run([loss, train], feed_dict={X: X_data, y: y_data}) 
    if step % 20 == 0:
        print(step, "loss : ", loss_val)

pred = sess.run(hypothesis, feed_dict={X: X_data})

# acc = accuracy_score(y_data, pred)
# print("acc : ", acc)

print(pred)

sess.close()








