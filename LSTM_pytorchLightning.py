# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:28:52 2020

@author: zankyo
"""
import numpy as np
import datetime
now = datetime.datetime.now()

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.utils.data as utils
#import torch.optim as optim
#from torchvision import datasets, transforms

#import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

# GPU check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

# Dataset

outname ="lstm"

outdir = 'C:/Users/zankyo/Desktop/yamamoto/model/' +outname+ now.strftime('%Y%m%d_%H%M%S')
print(outdir)

audio_len = 2**15
usedata_num = 15000
sample_rate = 16000

batch_size = 10

# Loading

print("データ読み込み")
datasets_dir = 'C:/Users/zankyo/Desktop/yamamoto/datasets/datasets129x257.npz'
datasets = np.load(datasets_dir)

data, label = datasets['data'], datasets['label']

data_list = []
for i in tqdm(range(len(data))):
    # x = torch.from_numpy(data[i].astype(np.float32)).clone()
    # data_list.append(x)
    data_list.append(data[i])
    
label_list = []
for i in tqdm(range(len(data))):
    # x = torch.from_numpy(label[i].astype(np.float32)).clone()
    # label_list.append(x)
    label_list.append(label[i])
    
data_list =  np.array(data_list)
label_list =  np.array(label_list)

tensor_data = torch.from_numpy(data_list.astype(np.float32)).clone()
tensor_label = torch.from_numpy(label_list.astype(np.float32)).clone()
    
# tensor_data = torch.stack(torch.Tensor(i) for i in data_list)
# tensor_label = torch.stack(torch.Tensor(i) for i in label_list)

mydataset = utils.TensorDataset(tensor_data,tensor_label)

train_dataset, val_dataset,test_dataset = utils.random_split(mydataset,[12000,2500,500])

# Dataloader

train_dataloader = utils.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = utils.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = utils.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

#辞書型変数にまとめる
dataloaders_dict = {"train":train_dataloader,
                    "val":val_dataloader,
                    "test":test_dataloader}


# 動作の確認
"""
Batch, Freq(Feature), Time(Sequence)
"""
print("動作チェック")
batch_iterator = iter(dataloaders_dict["train"])
datas,labels=next(batch_iterator)
print("data size",datas.size())
print("label size",labels.size())


# model
feat_size = data.shape[1] #129
sequence_len = data.shape[2] #257
lstm_layer = 3
hidden_layer =129

class Net(pl.LightningModule):
    def __init__(self):
        
        super(Net,self).__init__()
        self.seq_len = sequence_len
        self.feature_size = feat_size #129
        self.output_size = feat_size #129
        self.hidden_layer_size = hidden_layer #隠れ層のサイズ  #129
        self.lstm_layers = lstm_layer #LSTMのレイヤー数
        
        self.encoder = nn.Sequential(
            #nn.LSTM(self.feature_size,
            nn.LSTM(self.seq_len,
                    self.hidden_layer_size,
                    num_layers = self.lstm_layers),
            # nn.ReLU(),
            nn.Linear(self.hidden_layer_size,self.output_size),
            nn.Sigmoid())
        
    def forword(self,x):
        embedding = self.encoder(x)
        return embedding        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)  # forwardを呼び出す self(x) でもよい
        loss = nn.L1Loss(z, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        # 検証ステップ
        x, y = val_batch
        z = self.encoder(x)
        loss = nn.L1Loss(z, y)
        self.log("val_loss", loss)
        return loss

#training
net =Net()
trainer = pl.Trainer(max_epochs=5, gpus=1)# 学習用のインスタンス化と学習の設定

trainer.fit(net, train_dataloader,val_dataloader)