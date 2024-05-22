import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

learning_rate = 1e-3
rate = tf.compat.v1.placeholder(tf.float32)

# 1.데이터
(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')/255.0
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')/255.


# 2. 모델구성
X = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])     # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
   
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64])   # kernel_size
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)
b1 = tf.compat.v1.Variable(tf.zeros([64]), name = 'b1')
                                           
                                            
L1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='VALID')   # 양끝 두 숫자는 shape 맞춰주기위해 존재
L1 += b1     # L1 = L1 +b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# model.add(conv2d(64, kernel_size=(2,2), input_shape = (28, 28, 1), stride = 1, padding = 'valid))
# print(w1)
# <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)
# Tensor("Relu:0", shape=(?, 27, 27, 64), dtype=float32)
# print(L1_maxpool)
# Tensor("MaxPool2d:0", shape=(?, 13, 13, 64), dtype=float32)


w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32])   # kernel_size
b2 = tf.compat.v1.Variable(tf.zeros([32]), name = 'b2')

L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='VALID') + b2

L2 += b2     # L1 = L1 +b1
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

w3 = tf.compat.v1.get_variable('w3', shape=[4, 4, 32, 16])
b3 = tf.compat.v1.Variable(tf.zeros([16]), name = 'b3')

L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding ='VALID') 
L3 += b3     # L1 = L1 +b1
L3 = tf.nn.sigmoid(L3)


w4 = tf.compat.v1.get_variable('w4', shape=[5, 5, 16, 10])
b4 = tf.compat.v1.Variable(tf.zeros([10]), name = 'b4')

L4 = tf.nn.conv2d(L3, w4, strides=[1, 1, 1, 1], padding='VALID') + b4


hypothesis = tf.compat.v1.nn.softmax(L4)

loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 600
training_epochs = 100
total_batch = int(len(X_train) / batch_size)
# 60000 / 100
# print(total_batch)    # 600


for step in range(training_epochs):
    
    avg_cost = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_X, batch_y = X_train[start:end], y_train[start:end]
        feed_dict = {X:batch_X, y:batch_y, rate:0.3}
        
        cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4],
                                             feed_dict=feed_dict)
        avg_cost += cost_val / total_batch
        # i + 1 = i ---> +=
    if step % 20 ==0:
        print(step, 'loss : ', avg_cost)
        
        
        
        

# 4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={X:X_test, rate: 0})
y_predict_arg = sess.run(tf.math.argmax(y_predict, 1))
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict_arg)
print("ACC : ", acc)

sess.close()


