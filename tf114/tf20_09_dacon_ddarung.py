import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import pandas as pd

path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")


train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())


test_csv = test_csv.fillna(test_csv.mean())

X = train_csv.drop(['count'], axis=1)

y = train_csv['count']

print(X.shape)  # (1459, 9)
print(y.shape)  # (1459,)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)      

mas = StandardScaler()
mas.fit(X_train)
X_train = mas.transform(X_train)
X_test = mas.transform(X_test)
test_csv = mas.transform(test_csv)


Xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])  
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, ])      





w1 = tf.compat.v1.Variable(tf.random_normal([9,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(Xp, w1) + b1)

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
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,1]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias5')






# 2. 모델 구성
hypothesis = tf.compat.v1.matmul(layer4, w5) + b5  


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)


# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3000

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

# MaxAbsScaler
# Loss:  20533.564
# R2 스코어:  -1.7981725893351612
# MSE:  20576.918330893834

# Minmaxsccaler
# Loss:  20540.664
# R2 스코어:  -1.7986241777929384
# MSE:  20580.23917638073

# StandardScaler
# Loss:  20300.176
# R2 스코어:  -1.7851174593065644
# MSE:  20480.91483724891

# Loss:  10691.33
# R2 스코어:  -0.4507970286793952
# MSE:  10668.724326590711



# Loss:  7420.286
# R2 스코어:  -0.013704806786574153
# MSE:  7454.479791697866