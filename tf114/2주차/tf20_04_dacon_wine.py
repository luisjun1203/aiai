import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")

lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])


X_data = train_csv.drop(['quality'], axis=1)
# print(X)
# print(X_data.shape)   # (5497, 12)
# print(y_data.shape)   # (5497, 7)
y_data = train_csv['quality']


y_data = y_data.values.reshape(-1, 1)  

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data)

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_data_ohe.shape[1]])  

# w = tf.compat.v1.Variable(tf.random.normal([12, y_data_ohe.shape[1]]), name='weight')  
# b = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name='bias') 

# hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)


w1 = tf.compat.v1.Variable(tf.random_normal([12,19]), name= 'weight1')
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
w5 = tf.compat.v1.Variable(tf.random_normal([21,y_data_ohe.shape[1]]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name = 'bias5')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)






loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)  
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={X: X_data_scaled, y: y_data_ohe})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val)

y_predict_ohe = sess.run(hypothesis, feed_dict={X: X_data_scaled})
y_predict = np.argmax(y_predict_ohe, axis=1)
original_y_data = np.argmax(y_data_ohe, axis=1)  

acc = accuracy_score(original_y_data, y_predict)
print("ACC : ", acc)

sess.close()

# ACC :  0.5441149718028016
# ACC :  0.6576314353283609

