import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

tf.compat.v1.set_random_seed(777)



X_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152], [185], [180], [205], [142]]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])



w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)



#3333333333[실습] ######### 맹그러봐!!

# 2. 모델

hypothesis = tf.compat.v1.matmul(X, w) + b      

loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(loss)


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10000

for step in range(epochs):
    loss_val, _, bias = sess.run([loss, train, b], feed_dict={X:X_data, y:y_data})
    if step % 20 ==0:
        print(step, "loss : ", loss_val)

pred = sess.run(hypothesis, feed_dict={X: X_data})

r2 = r2_score(y_data, pred)
mae = mean_absolute_error(y_data, pred)

print('R2 스코어 : ', r2)
print('MAE : ', mae)

sess.close()












