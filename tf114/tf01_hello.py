import tensorflow as tf
print(tf.__version__)   # 1.14.0

print("텐서플로로 hello world")

hello = tf.constant('hello world')                     # variable도 알아두기
print(hello)
# Tensor("Const:0", shape=(), dtype=string)         

sess = tf.Session()                             # 변수 정의 후 세션 정의 해주기

print(sess.run(hello))  # b'hello world'












