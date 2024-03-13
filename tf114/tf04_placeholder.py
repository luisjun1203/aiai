import tensorflow as tf
# print(tf.__version__)             # 1.14.0
# print(tf.executing_eagerly())     # False

tf.compat.v1.disable_eager_execution()      

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(40.0, tf.float32)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4})) # 7.0           # feed_dict : 딕셔너리 형태로 input을 넣어주기
print(sess.run(add_node, feed_dict={a:30, b:4.5})) # 34.5



