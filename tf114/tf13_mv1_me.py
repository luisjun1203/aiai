import tensorflow as tf
tf.compat.v1.set_random_seed(777)


# 1. 데이터
X1_data = [73., 93., 89., 96., 73.]
X2_data = [80., 88., 91., 98., 66.]
X3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]


# [실습] 맹그러봐!!!

_X1 = tf.placeholder(tf.float32)
_X2 = tf.placeholder(tf.float32)
_X3 = tf.placeholder(tf.float32)


_y = tf.placeholder(tf.float32)



w1 = tf.Variable(tf.random_normal([1]),dtype=tf.float32, name='weights1')
w2 = tf.Variable(tf.random_normal([1]),dtype=tf.float32, name='weights2')
w3 = tf.Variable(tf.random_normal([1]),dtype=tf.float32, name='weights3')
 
b = tf.Variable(0,dtype=tf.float32)

hypothesis = _X1 * w1 + _X2 * w2 + _X3 * w3 + b

loss = tf.reduce_mean(tf.square(hypothesis-_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

EPOCHS = 50000


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # 훈련
    for step in range(EPOCHS):
        _, loss_val = sess.run([train, loss], feed_dict={_X1: X1_data, _X2: X2_data, _X3: X3_data, _y: y_data})
        if step % 10 == 0:
            print(f"Step: {step}, Loss: {loss_val}")
    
    
    w1_val, w2_val, w3_val, b_val = sess.run([w1, w2, w3, b])
    print("w1:", w1_val[0], "w2:", w2_val[0], "w3:", w3_val[0], "b:", b_val)



