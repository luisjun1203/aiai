import tensorflow as tf
# print(tf.__version__)   # 1.14.0
# print(tf.executing_eagerly())   # False , 즉시 실행 모드

# 즉시 실행 모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법을 실행시킨다.
# 즉시 실행 모드 켠다
tf.compat.v1.disable_eager_execution()      # 즉시실행모드를 끈다 // 텐서플로 1.0 문법 // 디폴트
# tf.compat.v1.enable_eager_execution()     # 즉시실행모드를 켠다 // 텐서플로 2.0 사용 가능

# print(tf.executing_eagerly())   # True    

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))  # 에러


# 가상환경     즉시실행모드           사용가능
# 1.14.0       disable(디폴트)           o          ★★★★    
# 1.14.0       enable                   x(에러)
# 2.9. 0       disable                   o          ★★★★
# 2.9. 0       enable(디폴트)            x(에러)


