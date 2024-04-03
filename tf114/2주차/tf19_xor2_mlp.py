import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import accuracy_score

# 1. 데이터

X_data = [[0,0], [0,1], [1,0], [1,1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]

# [실습] 맹그러봐
# m02_5번과 똑같은 레이어로 구성

# 2. 모델 구성 


# layer1 =  model.add(Dense(19, input_dim=2))

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])


w1 = tf.compat.v1.Variable(tf.random_normal([2,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.compat.v1.matmul(X, w1) + b1

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2


# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,1]), name = 'weight4')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias4')
# layer5 =tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)
   
# 3-1. 컴파일

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # 'binary_crossentropy'

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

# 3-2. 훈련


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ = sess.run([loss, train], feed_dict={X:X_data, y:y_data})
    if step % 20 ==0:
        print(step, "loss : ", cost_val)
        
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], feed_dict = {X:X_data, y:y_data})
    
    print("훈련값 : ", hypo)
    print("예측값 : ", pred)
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

























########### 내 방식 ############ 안좋음 ####################
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epochs = 2000
# for step in range(epochs):
#     cost_val, _ = sess.run([loss, train], feed_dict={X:X_data, y:y_data})

#     if step % 20 ==0:
#         print(step, "loss : ", cost_val)

# y_pred = sess.run(hypothesis, feed_dict={X: X_data})

# # y_pred = tf.sigmoid(tf.matmul(X, w) + b)
# # predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={X : X_data})
# print(y_pred)
# acc = accuracy_score(y_data, y_pred)

# print('acc : ', acc)
# sess.close()























