import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# GPU 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# 데이터 로드
datasets = load_digits()
x = datasets.data
y = datasets.target



# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# 데이터 전처리
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터를 Tensor로 변환하여 GPU로 이동
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)  # CrossEntropyLoss는 클래스 인덱스를 입력으로 받음
y_test = torch.LongTensor(y_test).to(DEVICE)    # One-hot 인코딩 대신 클래스 인덱스를 사용

# 모델 정의
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 7),
    nn.ReLU(),
    nn.Linear(7, 10)  # 출력층에서 softmax 함수를 사용하지 않음
).to(DEVICE)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 함수 정의
def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 학습
epochs = 1500
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch:', epoch, 'loss:', loss)

# 평가
with torch.no_grad():
    model.eval()
    y_pred = model(x_test).argmax(dim=1)  # 클래스 예측값으로 변환
    accuracy = (y_pred == y_test).float().mean().item()
    print('accuracy:', accuracy)

# accuracy: 0.9638888835906982