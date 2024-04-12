import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(3)

X_train = np.array([[[[1],[2],[3]],
                    [[4],[5],[6]],
                    [[7],[8],[9]]]])

print(X_train.shape)    # (1, 3, 3, 1)


X = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])


w1 = tf.compat.v1.constant([[[[1.]], [[0.]]],
                            [[[1.]], [[0.]]]])  
print(w1)   # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

L1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='VALID')  
print(L1)   # Tensor("Conv2D:0", shape=(?, 3, 3, 1), dtype=float32)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={X:X_train})
print('===================결과=======================================')
print(output)
print('===================결과.shape=======================================')
print(output.shape)


# ===================결과=======================================
# [[[[ 5.]
#    [ 7.]]

#   [[11.]
#    [13.]]]]
# ===================결과.shape=======================================
# (1, 2, 2, 1)
