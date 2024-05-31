# dacon_dechul

# dacon_wine

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits , fetch_covtype
from sklearn.preprocessing import StandardScaler
import pandas as pd

# GPU 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# 데이터 로드
path = "/home/saka/바탕화면/_data/dacon/loan/"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
test_csv = pd.read_csv(path + "test.csv", index_col=0 )

train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()



train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

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


y = pd.get_dummies(y)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# 데이터 전처리
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape , np.unique(y))

# 데이터를 Tensor로 변환하여 GPU로 이동
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.to_numpy()).to(DEVICE)  # CrossEntropyLoss는 클래스 인덱스를 입력으로 받음
y_test = torch.FloatTensor(y_test.to_numpy()).to(DEVICE)    # One-hot 인코딩 대신 클래스 인덱스를 사용

print(np.unique(y))

from torch.utils.data import TensorDataset, DataLoader

train_set  = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# 모델 정의
# model = nn.Sequential(
#     nn.Linear(13, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 7)  # 출력층에서 softmax 함수를 사용하지 않음
# ).to(DEVICE)



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
        x = self.softmax(x)
        return x
    
    
model = Model(13, 7).to(DEVICE)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 함수 정의
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