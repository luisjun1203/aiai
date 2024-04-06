import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)        # 이걸 해줘야지 torch data로 사용할 수 있음 // unsqueeze(1) 2 차원을 맞춰주는 것  -> (3,) 에서 (3,1)
y = torch.FloatTensor(y).unsqueeze(1)        # shape를 똑같이 해줘야 된다 // 똑같지 않으면 그냥 n 빵 때려서 중간값으로 1개가 들어가게 됨 이 때는 2가 들어가게 됨 -> (3,) 에서 (3,1)


print(x,y)  # tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 1))
model = nn.Linear(1,1) # 인풋 , 아웃풋  # y = xw + b

#3 컴파일
# model.compile(loss = 'mse' , optimizer = 'adam' )
criterion = nn.MSELoss()                # criterion : 표준 -> 이렇게 쓰는 이유는 그냥 제일 많이 써서
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters() , lr=0.01 )         # SGD = Batch 안에 mini batch를 만들어서 그 안에 있는 것 중 랜덤으로 1개를 지정해서 훈련하고 나머지는 버림
                                                            # 연산량이 빨라지고 성능이 좋을 수 도 있음 , 다른 에포에서는 다른 것을 쓸 수 있어서 데이터를 100% 버린다고는 할 수 없음

# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, x, y):
    
    ####### 순전파 #######
    # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
    # w = w - lr * (loss를 weight로 미분한 값)
    optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                # Gradient는 Weight를 초기화 시키기 위해서 사용되는거지 전 에포에 gradient가 영향을 끼치면 epoch가 이어지게 되서 
                                # 그 문제를 해결하기 위해 0 을 넣어주게 된다.
                                # Weight를 갱신해야되는거지 gradient를 갱신 X

    hypothesis = model(x)       # 예상치 값 (순전파)
    
    loss = criterion(hypothesis , y)    #예상값과 실제값 loss
    
    #####################
    
    loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산까지 -> 역전파 시작
    optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
    return loss.item()      # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 1000
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , x , y)
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , x,y ) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y , y_predict)
    return loss2.item()

loss2 = evaluate(model , criterion , x, y)
print('최종 loss : ', loss2 )

# result = model.predict([4])
result = model(torch.Tensor([[4]]))
print('4의 예측값은 :' , result.item() )


# 최종 loss :  0.00011766229727072641
# 4의 예측값은 : 4.021754264831543