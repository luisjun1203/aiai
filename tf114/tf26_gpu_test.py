import tensorflow as tf

# tf.compat.v1.enable_eager_execution()   # 즉시 실행모드 1.0
tf.compat.v1.disable_eager_execution()      # 즉시모드 해 2.0

print("텐서플로 버전 : ", tf.__version__)
print("즉시실행모드 : ", tf.executing_eagerly())

# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  True

# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  False

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:    
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("gpu없다!!!")   
    
# Tensorflow1 -> 그래프 연산        
# Tensorflow2 -> 즉시실행모드
# tf.compat.v1.enable_eager_execution()   -> 즉시실행모드 켜
# tf.compat.v1.disable_eager_execution() -> 즉시실행모드 꺼 -> 그래프 연산모드 -> Tensorflow1 코드 사용가능
