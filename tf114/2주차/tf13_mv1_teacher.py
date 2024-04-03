import tensorflow as tf
tf.compat.v1.set_random_seed(420)


# 1. 데이터
X1_data = [73., 93., 89., 96., 73.]
X2_data = [80., 88., 91., 98., 66.]
X3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]


X1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
X2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
X3 = tf.compat.v1.placeholder(tf.float32, shape=[None])

y = tf.compat.v1.placeholder(tf.float32)



w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32, name='weights1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32, name='weights2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32, name='weights3')
 
b = tf.Variable(0,dtype=tf.float32)

# 2. 모델

hypothesis = X1 * w1 + X2 * w2 + X3 * w3 + b


# 3-1. 컴파일

loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(loss)

# 3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={X1:X1_data, X2:X2_data, X3:X3_data, y:y_data})

    if step % 20 ==0:
        print(step, "loss : ", cost_val)

sess.close()





























