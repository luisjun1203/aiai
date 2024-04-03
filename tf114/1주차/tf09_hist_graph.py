# 실습
# lr 수정해서 epoch 101번 이하로 줄여서
# step = 100 이하 w = 1.99, b = 0.99

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# 1. 데이터
X_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]

X = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값

# 2. 모델 구성

hyopthesis = X * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hyopthesis - y))    # mse
lr = 0.001

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)       # == model.compile(loss='mse', optimizer='sgd') 
train = optimizer.minimize(loss)                                        # == model.compile(loss='mse', optimizer='sgd') 


# 3-2. 훈련
sess = tf.compat.v1.Session()
loss_val_list = []
w_val_list = []
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    # model.fit
    epochs = 100
 
    for step in range(epochs):
        

        # sess.run(train, feed_dict={X_ph: X, y_ph: y})         # 메인 알고리즘
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={X:X_data, y:y_data})

        if step % 20 == 0:
            print(step + 1, loss_val, w_val, b_val)   # verbose와 model.weight에서 봤던 애들.
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    # 4. 예측
    # [실습]

    sess.run(tf.global_variables_initializer())
    sess = tf.compat.v1.Session()          
    X_pred_data = [6,7,8]
    # 예측값 뽑아봐
    X_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    # 1. 파이썬 방식
    y_pred = X_pred_data * w_val + b_val

    # 2. placeholder 
    y_pred2 = X_test * w_val + b_val

    # print("best_model : ", best_model)
    ####################################################################
    print('[6,7,8] 의 예측 : ', sess.run(y_pred2, feed_dict={X_test:X_pred_data}), "w_val : ", w_val, "b_val : " , b_val )

    # if tf.reduce_all(tf.abs(w_val - 2) <= 0.009).eval(session=sess): 
    #     print("lr : ", lr )
    #     break
    # else :
    #     lr = lr * 0.95
# print(loss_val_list)
# print(w_val_list)


# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss') 
# plt.show()


# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.show()
     
     
     
# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()  

# subplot으로 위 세개의 그래프를 그려!!!


epochs = range(1, 101)

plt.subplot2grid((2,2), (0,0), colspan=1)  # 1행 3열의 그리드에서 첫 번째 위치
plt.plot(epochs, loss_val_list, 'r-')  # 'r-'는 빨간색 실선을 의미
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 두 번째 그래프: epochs에 따른 weights의 변화
plt.subplot2grid((2,2), (0,1), colspan=1)  # 1행 3열의 그리드에서 두 번째 위치
plt.plot(epochs, w_val_list, 'b-')  # 'b-'는 파란색 실선을 의미
plt.xlabel('Epochs')
plt.ylabel('Weights')

# 세 번째 그래프: weights와 loss의 관계
plt.subplot2grid((2,2), (1,0), colspan=2)  # 2행 1열의 그리드
plt.scatter(w_val_list, loss_val_list)
plt.xlabel('Weights')
plt.ylabel('Loss')





# 전체 그래프의 타이틀 설정
plt.suptitle('Training Progress')

# 그래프 표시
plt.tight_layout()  # subplot 간 간격 자동 조정
plt.show() 