import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler



path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']  


print(X.shape)  # (10886, 8)
print(y.shape)  # (10886,)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

sclaer = RobustScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)

Xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])  
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, ])      

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)





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

# maxabs
# Loss:  91106.99
# R2 스코어:  -1.7962254637109196
# MSE:  90612.11640048248

# minmax
# Loss:  69138.195
# R2 스코어:  -1.1340948842283507
# MSE:  69155.6730918051

# Standard
# Loss:  68098.08
# R2 스코어:  -1.1018226938878306
# MSE:  68109.88779817113

# Robust
# Loss:  68018.23
# R2 스코어:  -1.1002236336952875
# MSE:  68058.07000658798

# Loss:  68010.37
# R2 스코어:  -1.1000481676517806
# MSE:  68052.38400244956


# Loss:  67931.83
# R2 스코어:  -1.0982957885954816
# MSE:  67995.59788949556














