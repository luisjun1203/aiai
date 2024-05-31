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
path = '/home/saka/바탕화면/_data/dacon/wine/'
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)
test_csv = pd.read_csv(path + 'test.csv' , index_col=0)

train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

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


from torch.utils.data import TensorDataset, DataLoader

train_set  = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# 모델 정의
# model = nn.Sequential(
#     nn.Linear(12, 64),
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
    
    
model = Model(12, 7).to(DEVICE)


# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 함수 정의
def train(model , criterion , optimizer, loader):
    
    total_loss = 0
    
    for x_batch, y_batch in loader:
        
    
    
        ####### 순전파 #######
        # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
        # w = w - lr * (loss를 weight로 미분한 값)
        optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                    # 1. 처음에 0으로 시작하는게 좋아서
                                    # 2. epoch가 돌때마다 전의 gradient를 가지고 있어서 그게 문제가 될 수 있어서 이걸 해결 하기 위해서
                                    #    계속 0으로 바꿔주는 것이다. 

        hypothesis = model(x_batch)       # 예상치 값 (순전파)
        
        loss = criterion(hypothesis , y_batch)    #예상값과 실제값 loss
        
        #####################
        
        loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
        optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()        
        
        return total_loss / len(loader)      # total_loss / 13
    
    
    
epochs = 1000
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , train_loader )
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , loader ) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    total_loss = 0
    
    for x_batch, y_batch in loader:
    
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch , y_predict)
            total_loss += loss2.item()
            
        return total_loss / len(loader)

loss2 = evaluate(model , criterion , test_loader)
print('최종 loss : ', loss2 )

# result = model.predict([4])
result = model(x_test)
print('예측값은 :' , result.tolist() )



from sklearn.metrics import accuracy_score

# with torch.no_grad():
#     y_pred = model(x_test).cpu().numpy().squeeze()

# y_test = y_test.cpu().numpy().squeeze() 

y_pred = model(x_test).cpu().detach().numpy()   # detach 는 gradient를 안받는 것 // Tensor 데이터 뒤에 기울기가 붙는데 그걸 없애줘야 메모리를 효율적으로 쓰고 영향을 받지 않는다.

y_pred = np.around(y_pred) 
y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test,y_pred)
print('acc :{:.4f} '.format(acc))
# accuracy: 0.5845454335212708