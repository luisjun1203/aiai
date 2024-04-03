import tensorflow as tf

# tf.set_random_seed(123)
sess = tf.compat.v1.Session()

X = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


hypothesis = X * w + b

loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss=loss)

sess.run(tf.global_variables_initializer())

for step in range(10):
    sess.run(train)
    print(step, sess.run(loss), sess.run(w), sess.run(b))
    
sess.close()


# 0 128.81662 [-1.1834573] [0.13174361]
# 1 128.72055 [-1.1824573] [0.1327436]
# 2 128.62454 [-1.1814574] [0.13374357]
# 3 128.52853 [-1.1804575] [0.13474351]
# 4 128.43259 [-1.1794575] [0.13574342]
# 5 128.33669 [-1.1784576] [0.1367433]
# 6 128.24081 [-1.1774578] [0.13774315]
# 7 128.14499 [-1.176458] [0.13874294]
# 8 128.0492 [-1.1754583] [0.13974267]
# 9 127.953445 [-1.1744586] [0.14074235]