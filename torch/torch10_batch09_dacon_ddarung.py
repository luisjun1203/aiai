# dacon_ddarung

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
path = "/home/saka/바탕화면/_data/dacon/ddarung/"

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

from torch.utils.data import TensorDataset, DataLoader

train_set  = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)



#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 1))
# model = nn.Sequential(
#     nn.Linear(9,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,7),
#     nn.ReLU(),
#     nn.Linear(7,1),
#     ).cuda() # 인풋 , 아웃풋  # y = xw + b

class Model(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 19)
        self.linear2 = nn.Linear(19, 97)
        self.linear3 = nn.Linear(97, 9)
        self.linear4 = nn.Linear(9, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        
        return
    
    # 순전파!!
    def forward(self, input_size):      
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        # x = self.sigmoid(x)
        x = self.relu(x)
        return x
    
    
model = Model(9, 1).to(DEVICE)




#3 컴파일
# model.compile(loss = 'mse' , optimizer = 'adam' )
criterion = nn.MSELoss()                  # criterion : 표준 -> 이렇게 쓰는 이유는 그냥 제일 많이 써서
# criterion = nn.BCELoss()                    # binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters() , lr=0.01 )         # SGD = Batch 안에 mini batch를 만들어서 그 안에 있는 것 중 랜덤으로 1개를 지정해서 훈련하고 나머지는 버림
                                                            # 연산량이 빨라지고 성능이 좋을 수 도 있음 , 다른 에포에서는 다른 것을 쓸 수 있어서 데이터를 100% 버린다고는 할 수 없음

# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 평가 함수 정의
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

epochs = 100
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print(f'epoch : {epoch}, loss: {loss:.4f}')

print('==========================================================')

# 최종 평가
loss2 = evaluate(model, criterion, test_loader)
print('최종 loss :', loss2)

# 예측
with torch.no_grad():
    model.eval()
    y_pred = model(x_test).argmax(dim=1)  # 클래스 예측값으로 변환
    accuracy = (y_pred == y_test).float().mean().item()
    print('accuracy:', accuracy)

loss2 = evaluate(model , criterion ,test_loader)
rmse = np.sqrt(loss2)
print('최종 loss : ', loss2 )

# result = model.predict([4])
from sklearn.metrics import accuracy_score ,r2_score

""" with torch.no_grad():
    y_pred = model(x_test).cpu().numpy().squeeze()

y_test = y_test.cpu().numpy().squeeze() """

y_pred = model(x_test).cpu().detach().numpy()

# y_pred = np.around(y_pred)
y_test = y_test.cpu().numpy()

r2 = r2_score(y_test,y_pred)
print('r2 :{:.4f} '.format(r2))

# r2 :0.7514