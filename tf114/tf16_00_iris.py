from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(3)


datasets = load_iris()
X, y = datasets.data, datasets.target

X = X[ y != 2]
y = y[ y != 2]

print(y, y.shape)   # (100,)
print(X, X.shape)   # (100, 4)

# 실습 : 맹그러봐

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Xp = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])
yp = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1]), dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32), name='bias')

# print(y_train.shape)
# print(y_test.shape)

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(Xp, w)+ b)

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs = 10000


for step in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={Xp: X_train, yp: y_train})
    if step % 50 ==0:
        print(step, "loss : ", loss_val)
        
# print(w_val, b_val)
# [[-0.01151588]
#  [ 0.01308101]
#  [ 0.12389403]
#  [-0.12826364]] [-0.0096294]

# print(type(w_val))  # <class 'numpy.ndarray'>


# 4. 평가, 예측
test_loss = sess.run(loss, feed_dict={Xp: X_test, yp: y_test})
print("Test Loss:", test_loss)

predictions = sess.run(hypothesis, feed_dict={Xp: X_test})
predicted_classes = np.where(predictions > 0.5, 1, 0)
print("결과:", predicted_classes)

# 정확도 계산
acc = accuracy_score(y_test, predicted_classes)
print("Accuracy:", acc)

sess.close()

# Accuracy: 0.45









