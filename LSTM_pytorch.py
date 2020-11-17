# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:22:29 2020

@author: zankyo
"""
import numpy as np
import datetime
now = datetime.datetime.now()

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim
# from torchvision import datasets, transforms

# import matplotlib.pyplot as plt
from tqdm import tqdm

import os

#%% GPU check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

#%% Dataset

audio_len = 2**15
usedata_num = 15000
sample_rate = 16000

batch_size = 2

datasets_dir = 'C:/Users/zankyo/Desktop/yamamoto/datasets/datasets129x257.npz'
datasets = np.load(datasets_dir)

data, label = datasets['data'], datasets['label']

data_list = []
for i in tqdm(range(len(data))):
    data_list.append(data[i])
    
label_list = []
for i in tqdm(range(len(data))):
    label_list.append(label[i])
    
tensor_data = torch.stack([torch.Tensor(i) for i in data_list])
tensor_label = torch.stack([torch.Tensor(i) for i in label_list])

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

#%%model

sequence = data.shape[2] #257
feat = data.shape[1] #129

class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        self.seq_len = 257
        self.feature_size = 129
        self.hidden_layer_size = 129 #隠れ層サイズ
        self.rnn_layers = 3 #RNNのレイヤー層
         
        self.lstm=nn.LSTM(input_size = self.feature_size,
                 hidden_size = self.hidden_layer_size,
                 num_layers = self.rnn_layers,
                 dropout = 0.25,
                 batch_first = True)
            
        self.rnn = nn.RNN(input_size = self.feature_size,
                  hidden_size = self.hidden_layer_size,
                  num_layers = self.rnn_layers,
                  nonlinearity = 'relu',
                  dropout = 0.25,
                  batch_first = True)
        
        
    def forward(self,x):
        #[Batch, feature, sequence] -> [Batch, sequence, feature]
        x = x.permute(0,2,1)
        
        rnn_out,(h_n,c_n)= self.lstm(x)
        rnn_out,h_n = self.rnn(rnn_out, h_n)
        
        #[Batch, sequence, feature] -> [Batch, feature, sequence]
        y = rnn_out.permute(0,2,1)
        
        return y
    
    
#%%train setting
net = Net().to(device)

criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

max_epoch = 100
#%%training

for epoch in range(max_epoch):
    print('Epoch{}/{}'.format(epoch+1,max_epoch))
    for phase in ['train','val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()
            
        epoch_loss = 0.0
        
        if (epoch==0) and (phase == 'train'):
            continue
        
        for i,(inputs,labels) in tqdm(enumerate(dataloaders_dict[phase])):
            
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss = epoch_loss/len(dataloaders_dict[phase].dataset)
        
        print("{}Loss:{:.4f}".format(phase ,epoch_loss))
        
        


outname ="lstm"
outdir = 'model/' +outname+ now.strftime('%Y%m%d_%H%M%S')
os.makedirs(outdir)

print(outdir)
torch.save(net, outdir+"/GPUmodel.pt")
print(os.path.isfile(outdir+"/GPUmodel.pt"))

#%%load
torch.load("model\lstm20201117_204350\GPUmodel.pt")
check_point = "lightning_logs\version_14\checkpoints\epoch=4.ckpt"
autoencoder = net.load_from_checkpoint(check_point)


batch_iterator = iter(dataloaders_dict["test"])
test1,_=next(batch_iterator)

net.eval()

output = net(test1)
#%%
output_n = output.to('cpu').detach().numpy()

#%%
print(output_n.shape)

#%%
test2 = output[1]
print(test2.shape,test2.dtype)


