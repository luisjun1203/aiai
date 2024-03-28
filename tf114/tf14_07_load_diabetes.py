from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
X, y = load_diabetes(return_X_y=True)
print(X.shape, y.shape) # (442, 10), (442,)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

Xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])  # 수정
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])      # 수정

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

print(y_train.shape)
print(y_test.shape)

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))



# 2. 모델 구성
hypothesis = tf.compat.v1.matmul(Xp, w) + b  


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)


# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10000

for step in range(epochs):
    loss_val, _ = sess.run([loss, train], feed_dict={Xp: X_train, yp: y_train}) 
    if step % 20 == 0:
        print(step, "loss : ", loss_val)

# 4. 평가
test_loss = sess.run(loss, feed_dict={Xp: X_test, yp: y_test})  

y_pred = sess.run(hypothesis, feed_dict={Xp: X_test})
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('Loss: ', test_loss)
print('R2 스코어: ', r2)
print('MSE: ', mse)

sess.close()


# Loss:  26849.203
# R2 스코어:  -3.958577790707446
# MSE:  26849.200709076584