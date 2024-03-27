import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(3)

# X = [1, 2, 3]
# y = [1, 2, 3]

X = [1, 2]
y = [1, 2]

# X = [1, 2]
# y = [1, 2]


w = tf.compat.v1.placeholder(tf.float32)

hypothesis = X * w 

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in  range(-3, 51):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict = {w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("======================= W history ==============================================")
print("")
print(w_history)
print("======================= Loss history ==============================================")
print("")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weights')
plt.ylabel('Loss')
plt.show()






