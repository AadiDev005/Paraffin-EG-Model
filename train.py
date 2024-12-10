import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

from model import CNN
from dataset import load_dataset


def train_CNN(train_data, model, loss_fn, optimizer, epochs):
    train_data, val_data=torch.utils.data.random_split(train_data, [0.95, 0.05])
    best_loss=2**63
    val_loss_history=[]
    train_loss_history=[]
    for epoch in range(epochs):
        dataloader=DataLoader(train_data, batch_size=32, shuffle=True)
        running_loss=0
        model.train()
        for batch in dataloader:
            inputs, vols, wts, labels=batch
            optimizer.zero_grad()
            pred=model(inputs, vols)
            loss=loss_fn(pred, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        model.eval()
        with torch.no_grad():
            val_dataloader=DataLoader(val_data, batch_size=32, shuffle=True)
            val_loss=0
            for batch in val_dataloader:
                inputs, vols, wts, labels=batch
                pred=model(inputs, vols)
                loss=loss_fn(pred, labels.unsqueeze(1))
                val_loss+=loss.item()
            print(epoch, 'TRAINING LOSS:', running_loss/len(dataloader), 'VALIDATION LOSS:', val_loss/len(val_dataloader))
            val_loss_history.append(val_loss/len(val_dataloader))
            train_loss_history.append(running_loss/len(dataloader))
       
        if best_loss>running_loss/len(dataloader):
            best_loss=running_loss/len(dataloader)
            torch.save(model.state_dict(), "bestSoFar.pt")
    return val_loss_history, train_loss_history

dataset=load_dataset('./data/updated_lbm_data.csv', './data/updated_lbm_data/')
torch.manual_seed(44)
train_dataset, test_dataset=torch.utils.data.random_split(dataset=dataset, lengths=[0.8, 0.2])
print('Dataset prepped')

model=CNN(1, 64, 160).to('cuda')
print('Model created')

val_loss, train_loss=train_CNN(train_dataset, model, torch.nn.SmoothL1Loss(), torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-6), 300)
print('Training finished')
