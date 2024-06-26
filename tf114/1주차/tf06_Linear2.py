import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


# 1. 데이터

X = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# 2. 모델구성
hypothesis = X * w + b

sess = tf.compat.v1.Session()

# 3-1 컴파일

loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2 훈련

init = tf.global_variables_initializer()
sess.run(init)

epochs = 500
for step in range(epochs):
    sess.run(train)
    if step % 20==0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()         


# 0 77623.92 86.8 -6.6
# 20 128.00407 8.65207 -26.257078
# 40 110.40654 7.8233356 -24.62822
# 60 96.41886 7.3749905 -23.01571
# 80 84.20329 6.9574804 -21.508392
# 100 73.535324 6.567319 -20.099787
# 120 64.21898 6.2027116 -18.78344
# 140 56.082924 5.8619823 -17.553299
# 160 48.97763 5.543567 -16.403719
# 180 42.772537 5.2460055 -15.329427
# 200 37.353584 4.9679313 -14.325492
# 220 32.621166 4.7080693 -13.387305
# 240 28.488308 4.4652243 -12.51056
# 260 24.879055 4.238284 -11.691235
# 280 21.727057 4.026206 -10.925564
# 300 18.9744 3.8280172 -10.21004
# 320 16.570484 3.6428082 -9.541376
# 340 14.471113 3.4697278 -8.9165
# 360 12.6377325 3.3079834 -8.332552
# 380 11.036625 3.1568315 -7.7868457
# 400 9.638367 3.0155787 -7.2768784
# 420 8.417257 2.8835769 -6.8003097
# 440 7.3508515 2.7602198 -6.3549523
# 460 6.4195547 2.6449416 -5.9387617
# 480 5.606246 2.5372133 -5.549828











