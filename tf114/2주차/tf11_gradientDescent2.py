import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(3)

# X = [1, 2, 3]
# y = [1, 2, 3]

# 1.데이터

X_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# X = [1, 2]
# y = [1, 2]


w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')


# 2.모델구성

hypothesis = X * w + b

# 3-1. 컴파일 // model.compile(loss = 'mse', optimizer = 'sgd')

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

############################### 옵티마이저 ###############################################
# optimizer = tf.train.AdamOptimizer(learning_rate=0.08)
# train = optimizer.minimize(loss_fn)
lr = 0.01

# gradient = tf.reduce_mean((X * w - y) * X)
gradient = tf.reduce_mean((X * w + b - y) * X)


descent = w - lr * gradient


update = w.assign(descent)                          # 이 과정이 경사하강법

############################### 옵티마이저 ###############################################


# 3-2. 훈련
w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 


for step in range(1):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {X:X_train, y:y_train})
    print(step, '\t', loss_v, '\'t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()


# print("======================= W history ==============================================")
# print("")
# print(w_history)
# print("======================= Loss history ==============================================")
# print("")
# print(loss_history)

# plt.plot( loss_history)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()






