import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

# 1 데이터
datasets = load_digits()

X_data = datasets.data
y_data = datasets.target
y_data_r = datasets.target.reshape(-1, 1) 
# print(X_data.shape) # (1797, 64)
# print(y_data.shape) # (1797,)

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data_r)

scaler = MinMaxScaler()
X_data = scaler.fit_transform(X_data)


X = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w = tf.compat.v1.Variable(tf.random_normal([64,10]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([10]), name='bias')




hypothesis = tf.nn.softmax(tf.compat.v1.matmul(X, w) + b)

# 3-1. 컴파일

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# Categorical_Cross_Entropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)





# 3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={X: X_data, y: y_data_ohe})

    if step % 20 ==0:
        print(step, 'loss : ', cost_val)

    # loss_val, acc_val, _ = sess.run([loss, accuracy, train], feed_dict={X: X_data, y: y_data})
    # if step % 20 == 0:
    #     print(step, "Loss:", loss_val,"\t Accuracy:", acc_val)

y_predict_ohe = sess.run(hypothesis, feed_dict={X:X_data})
# print(y_predict)        # 8행 3열 데이터

# y_predict = sess.run(tf.argmax(y_predict, 1))
# print(y_predict)        # [2 2 2 1 1 1 0 0]

y_predict = np.argmax(y_predict_ohe, 1)
print(y_predict)        # [2 2 2 1 1 1 0 0]
# y_data = np.argmax(y_data, axis=1)
print(y_data)       # [2 2 2 1 1 1 0 0]

acc = accuracy_score(y_data, y_predict)

print("ACC : ", acc)
sess.close()


# [0 1 2 ... 8 9 8]
# [0 1 2 ... 8 9 8]
# ACC :  1.0








