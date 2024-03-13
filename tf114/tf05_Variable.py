import tensorflow as tf

sess = tf.compat.v1.Session()


a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()      # 변수는 항상 초기화 해주어야 한다
sess.run(init)                                          # 


print(sess.run(a + b))  # [5.]


















