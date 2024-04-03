import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


# 1. 데이터

X = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# 2. 모델구성
# y = wx + b
hypothesis =  X * w + b
# hypothesis =  w * X + b       # 이거 아님!!! y = X * w + b



sess = tf.compat.v1.Session()

# 3-1 컴파일

loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse, 실질저으로 predict

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # 경사하강법
train = optimizer.minimize(loss)

# model.compile(loss ='mse',optimizer=;sgd)

# 3-2 훈련
sess  = tf.compat.v1.Session()

init = tf.global_variables_initializer()
sess.run(init)

epochs = 3000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:    
        print(step, sess.run(loss),sess.run(w), sess.run(b))
sess.close()

