# 평가 rmse , r2
# kaggle_bike

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer , load_diabetes , fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pandas as pd

# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


#1. 데이터
path = 'C:\_data\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)
test_csv = pd.read_csv(path + 'test.csv' , index_col=0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

print(x,y)  # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape,y.shape) 

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , shuffle = True , random_state = 0 )

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = np.array(y_train)
y_test  = np.array(y_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE).unsqueeze(1)
y_test  = torch.FloatTensor(y_test).to(DEVICE).unsqueeze(1)  


#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 1))
model = nn.Sequential(
    nn.Linear(10,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,7),
    nn.ReLU(),
    nn.Linear(7,1),
    ).cuda() # 인풋 , 아웃풋  # y = xw + b

#3 컴파일
# model.compile(loss = 'mse' , optimizer = 'adam' )
criterion = nn.MSELoss()                  # criterion : 표준 -> 이렇게 쓰는 이유는 그냥 제일 많이 써서
# criterion = nn.BCELoss()                    # binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters() , lr=0.01 )         # SGD = Batch 안에 mini batch를 만들어서 그 안에 있는 것 중 랜덤으로 1개를 지정해서 훈련하고 나머지는 버림
                                                            # 연산량이 빨라지고 성능이 좋을 수 도 있음 , 다른 에포에서는 다른 것을 쓸 수 있어서 데이터를 100% 버린다고는 할 수 없음

# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, x, y):
    
    ####### 순전파 #######
    # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
    # w = w - lr * (loss를 weight로 미분한 값)
    optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                # 1. 처음에 0으로 시작하는게 좋아서
                                # 2. epoch가 돌때마다 전의 gradient를 가지고 있어서 그게 문제가 될 수 있어서 이걸 해결 하기 위해서
                                #    계속 0으로 바꿔주는 것이다. 

    hypothesis = model(x)       # 예상치 값 (순전파)
    
    loss = criterion(hypothesis , y)    #예상값과 실제값 loss
    
    #####################
    
    loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
    optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
    return loss.item()      # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 1500
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , x_train , y_train )
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

loss2 = evaluate(model , criterion , x_test, y_test)
rmse = np.sqrt(loss2)
print('최종 loss : ', loss2 )

# result = model.predict([4])
result = model(x_test)
print('예측값은 :' , result.tolist() )

from sklearn.metrics import accuracy_score ,r2_score

""" with torch.no_grad():
    y_pred = model(x_test).cpu().numpy().squeeze()

y_test = y_test.cpu().numpy().squeeze() """

y_pred = model(x_test).cpu().detach().numpy()

# y_pred = np.around(y_pred)
y_test = y_test.cpu().numpy()

r2 = r2_score(y_test,y_pred)
print('r2 :{:.4f} '.format(r2))

# r2 :1.0000