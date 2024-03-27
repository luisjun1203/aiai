import tensorflow as tf
tf.set_random_seed(777)

# data
X_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

_X = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

# model
hypothesis = _X*w + b

loss_fn = tf.reduce_mean(tf.abs(hypothesis - _y))  # mae
optimizer = tf.train.AdamOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 100

import numpy as np
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    lr = 0.08
    for step in range(EPOCHS):
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # train = optimizer.minimize(loss_fn)
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict={_X:X_data,_y:y_data})
        # if step % 100 == 0:
        #     print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
    
    print(f"lr: {lr:.6f}")
    print(f"w: {sess.run(w)}, b: {sess.run(b)}")