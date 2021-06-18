import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.nn import ReLU
from torch.nn.modules.loss import _WeightedLoss,MSELoss
from torch.nn import BatchNorm1d, Dropout, Linear, GRU
import torch.nn.functional as F
from torch.nn.functional import relu, leaky_relu, elu, selu, silu
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ZNN(torch.nn.Module):

    def __init__(self, config):
        super(ZNN, self).__init__()
        self.config = config
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                config['n_inputs'],
                config['n_units_cnn1'],
                kernel_size=1,
                padding=0),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv1d(
                config['n_units_cnn1'],
                config['n_units_cnn2'],
                kernel_size=3,
                padding=0),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv1d(
                config['n_units_cnn2'],
                config['n_units_cnn3'],
                kernel_size=5,
                padding=0),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv1d(
                config['n_units_cnn3'],
                config['n_units_cnn4'],
                kernel_size=7,
                padding=0),
            torch.nn.LeakyReLU(),
            
        )
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn1(x).transpose(2,1)

        return torch.sqrt((x.sum(1)**2).sum(-1))
        
        
class TorchData(torch.utils.data.Dataset):  # learn this
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        idx_dict = dict(
            x = torch.tensor(self.X[idx], dtype=torch.float),
            y = torch.tensor(self.y[idx], dtype=torch.float)
        )
        return idx_dict


def get_idx(x,y):
    t_idx, v_idx = {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (idx1, idx2) in enumerate(kf.split(x,y)):
        t_idx[i], v_idx[i] = idx1, idx2
    
    return t_idx, v_idx


def train_fn(model, optimizer, loss_fn, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        _loss = loss_fn(outputs, targets)
        _loss.backward()
        optimizer.step()

        loss += _loss.item()
        
    loss /= len(dataloader)
    return loss


def valid_fn(model, loss_fn, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    val_loss = 0
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        val_loss += loss.item()

    val_loss /= len(dataloader)
    return val_loss
    
def torch_training(
    model, 
    model_name, 
    train_dataset,
    valid_dataset
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle = False)
        
    optimizer = Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)     
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor= 0.1, patience=10, verbose=1)                                        
    loss_fn = MSELoss()
    
    #es = EarlyStopping(patience= 10, mode="max")    
    
    history = {}
    history['epoch'] = []
    history['val_loss'] =[]
    
    _best_loss = np.inf
    _best_epoch = 0
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer, loss_fn, trainloader)
        valid_loss = valid_fn(model, loss_fn, validloader)
        
        history['epoch'].append(epoch)
        history['val_loss'].append(valid_loss)
        
        print(f"{model_name} {epoch}/{EPOCHS}, t_loss: {train_loss:.5f}, v_loss: {valid_loss:.5f}")   
        
        if _best_loss > valid_loss:
            _best_loss = valid_loss
            _best_epoch = epoch
            torch.save(model.state_dict(), f"{model_name}_{fold}.pth") 
            
        scheduler.step(valid_loss)

    return history


 config_ = dict(
    n_inputs = 9,
    n_units_cnn1 = 32,
    n_units_cnn2 = 32,
    n_units_cnn3 = 64,
    n_units_cnn4 = 1

)