import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

df = pd.read_csv('c:/_data/dacon/bird/open/train.csv')
test = pd.read_csv('c:/_data/dacon/bird/open/test.csv')
train_img = 'c:/_data/dacon/bird/open/'
test_img = 'c:/_data/dacon/bird/open/'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':128,
    'SEED':921
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])


print(train.columns)


le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        # 이미지를 numpy 배열로 읽기
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        
        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            # 이미지를 변환
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, torch.tensor(label, dtype=torch.long)  # label을 torch.long으로 변환하여 반환
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
    ToTensorV2()
])

train_dataset = CustomDataset([os.path.join(train_img, img_name) for img_name in train['img_path'].values], train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset([os.path.join(train_img, img_name) for img_name in val['img_path'].values], val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

test_dataset = CustomDataset([os.path.join(test_img, img_name) for img_name in test['img_path'].values], None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)



class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 Score : [{_val_score:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
    
    return best_model

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=130, verbose=False, delta=0):
       
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_score_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_score, model):

#         score = -val_score

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_score, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_score, model)
#             self.counter = 0

#     def save_checkpoint(self, val_score, model):
#         if self.verbose:
#             print(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
#         # Save the model here if you want, or just report the best score
#         self.val_score_min = val_score


# def train(model, optimizer, train_loader, val_loader, scheduler, device, patience=130):
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
    
#     model.to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
    
#     for epoch in range(1, CFG['EPOCHS']+1):
#         model.train()
#         train_loss = []
#         for imgs, labels in tqdm(iter(train_loader)):
#             imgs = imgs.float().to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             output = model(imgs)
#             loss = criterion(output, labels)
            
#             loss.backward()
#             optimizer.step()
            
#             train_loss.append(loss.item())
                    
#         _val_loss, _val_score = validation(model, criterion, val_loader, device)
#         print(f'Epoch [{epoch}], Val F1 Score : [{_val_score:.5f}]')
       
#         if scheduler is not None:
#             scheduler.step(_val_score)
            
#         early_stopping(_val_score, model)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     return model 



def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')
        print("확인용 : ", _val_score)
    return _val_loss, _val_score

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test_dataset = CustomDataset([os.path.join(test_img, img_name) for img_name in test['img_path'].values], None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('c:/_data/dacon/bird/open/sample_submission.csv')
submit['label'] = preds
submit.to_csv('c:/_data/dacon/bird/open/sub_csv/bird04_10_2.csv', index=False)


# CFG = {
#     'IMG_SIZE':224,
#     'EPOCHS':100,
#     'LEARNING_RATE':3e-4,
#     'BATCH_SIZE':64,
#     'SEED':3
# }              Val F1 Score  : 0.91127, 제출: 0.90531

