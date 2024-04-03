import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.datasets import fetch_covtype

datasets = fetch_covtype()
X_data = datasets.data
y_data = datasets.target.reshape(-1, 1)  

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data)

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_data_ohe.shape[1]])  

w = tf.compat.v1.Variable(tf.random.normal([54, y_data_ohe.shape[1]]), name='weight')  
b = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name='bias') 

hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)  
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
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


# ACC :  0.3610768796513669