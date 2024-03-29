import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
import numpy as np

# 1. 데이터
X_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] #(6,1)


######################################################
##########[실습] 기냥 한번 맹그러봐!!! #################
######################################################

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),dtype=tf.float32)

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(X, w) + b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
# loss = 'binary_crossentropy'
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # 'binary_crossentropy'

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)

train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 1900

for step in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={X:X_data, y:y_data})
    if step % 50 ==0:
        print(step, "loss : ", loss_val)
        
# print(w_val, b_val)
# [[0.19869593]
#  [0.06757123]] [-0.30237883]
# print(type(w_val))  # <class 'numpy.ndarray'>

# 4. 평가, 예측

X_test = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y_pred = tf.sigmoid(tf.matmul(X_test, w_val) + b_val)
predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={X_test : X_data})
print("결과 :", predict)

# y_data = tf.compat.v1.round(y_data)
# predict = tf.compat.v1.round(predict)
acc = accuracy_score(y_data, predict)
print("acc : ", acc)


sess.close()



# 결과 : [[0.00982884]
#  [0.11364625]
#  [0.17883402]
#  [0.8469039 ]
#  [0.9732159 ]
#  [0.99182785]]

# 결과 : [[0.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]]
# acc :  1.0




