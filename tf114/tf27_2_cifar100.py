import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import accuracy_score
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)
     
#1. 데이터
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255.
x_test = x_test.reshape(-1,32,32,3).astype('float32')/255.
   
#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 32,32,3]) #input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 100]) 

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 3, 128], initializer=tf.keras.initializers.GlorotUniform()) #kernel 사이즈, 채널, output
b1 = tf.compat.v1.Variable(tf.zeros([128],name = 'b1'))
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') #stride1 에 4차원이므로 1,1,1,1  stride 2,2 ==> 1,2,2,1
L1 += b1
L1 = tf.nn.relu(L1)
# L1 = tf.nn.dropout(L1, keep_prob=0.7)
L1 = tf.nn.dropout(L1, rate= 0.3) #같은것.
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = "VALID")

# print(L1)#summary
# print (L1_maxpool)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,128,64], initializer=tf.keras.initializers.GlorotUniform())
b2 = tf.compat.v1.Variable(tf.zeros([64],name = 'b2'))
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME') 
L2 += b2
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, rate=0.3)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = "VALID")

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,64,32], initializer=tf.keras.initializers.GlorotUniform())
b3 = tf.compat.v1.Variable(tf.zeros([32],name = 'b3'))
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME') 
L3 += b3
L3 = tf.nn.elu(L3)

print(w3)
print(L3)

#Flatten
L_flat = tf.reshape(L3, [-1, 7*7*32])
print("Flatten  : ", L_flat)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[7*7*32, 100], initializer=tf.keras.initializers.GlorotUniform())
b4 = tf.compat.v1.Variable(tf.zeros([100]), name = 'b4')
L4 = tf.nn.relu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate= 0.3)
 
# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[100, 100], initializer=tf.keras.initializers.GlorotUniform())
b5 = tf.compat.v1.Variable(tf.zeros([100]), name = 'b5')
L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(L5)

#3-1 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), axis=1))
loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.003)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

training_epochs = 20
batch_size = 32
print(len(x_train))
total_batch = len(x_train)//batch_size
start_time = time.time()
for step in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        start = i + total_batch
        end = start + batch_size
        batch_x, batch_y = x_train[start:end] , y_train[start:end]
        _, loss_v= sess.run([train, loss], feed_dict = {x:batch_x, y:batch_y})
        avg_cost += loss_v / total_batch
    print(step, loss_v)
end_time = time.time()
print('걸린 시간 : ', end_time - start_time)

def get_batches(X, Y, batch_size):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        if Y is not None:
            yield X[start:end], Y[start:end]
        else:
            yield X[start:end], None  # No labels to return

def batch_predict(x_data, batch_size=500):
    pred_list = []
    for batch_x, _ in get_batches(x_data, None, batch_size):
        batch_pred = sess.run(hypothesis, feed_dict={x: batch_x})
        pred_list.append(batch_pred)
    return np.concatenate(pred_list, axis=0)

predictions = batch_predict(x_test, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_labels, predictions)

print('acc : ', acc)

sess.close()
# 걸린 시간 :  54.05942702293396
# acc :  0.0447
