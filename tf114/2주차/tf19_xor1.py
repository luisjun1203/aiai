import tensorflow as tf
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import accuracy_score

# 1. 데이터

X_data = [[0,0], [0,1], [1,0], [1,1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2,1]), name= 'weight')

b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias')


# [실습] 맹그러봐

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(X, w) + b)
   
# 3-1. 컴파일

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # 'binary_crossentropy'

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

# 3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={X:X_data, y:y_data})

    if step % 20 ==0:
        print(step, "loss : ", cost_val)


y_pred = tf.sigmoid(tf.matmul(X, w) + b)
predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={X : X_data})
print(predict)
acc = accuracy_score(y_data, predict)

print('acc : ', acc)
sess.close()























