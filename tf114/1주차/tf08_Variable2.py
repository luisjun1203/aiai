import tensorflow as tf
tf.compat.v1.set_random_seed(777)

X_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

_X = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32, name='weights')
b = tf.Variable(0,dtype=tf.float32)

hypothesis = _X*w + b

loss_fn = tf.reduce_mean(tf.square(hypothesis-_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss_fn)

EPOCHS = 100

import numpy as np


# Session() // sess.run(변수)
print("초기화 첫번째")
with tf.compat.v1.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(EPOCHS):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b],feed_dict={_X:X_data,_y:y_data})
        if step % 50 == 0:
            print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
    print(f"w: {sess.run(w)}, b: {sess.run(b)}")
    
print("초기화 두번째")
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(EPOCHS):
        train.run(feed_dict={_X:X_data,_y:y_data})
        loss = loss_fn.eval(session=sess, feed_dict={_X:X_data,_y:y_data})
        weights = w.eval(session=sess)
        bias = b.eval(session=sess)
        if step % 50 == 0:
            print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
    print(f"w: {sess.run(w)}, b: {sess.run(b)}")
    
print("초기화 세번째")
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(EPOCHS):
    train.run(feed_dict={_X:X_data,_y:y_data})
    loss = loss_fn.eval(feed_dict={_X:X_data,_y:y_data})
    weights = w.eval()
    bias = b.eval()
    if step % 50 == 0:
        print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
print(f"w: {sess.run(w)}, b: {sess.run(b)}")