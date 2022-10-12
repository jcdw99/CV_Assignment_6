
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from Q1_dataset import imgDataset
import Q1_admin
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time
import copy
import os
import torch.utils.data

if __name__ == "__main__":

    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT
    INIT_LR = 0.001
    BATCH_SIZE = 64*3
    EPOCHS = 20

    transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results, data = Q1_admin.get_data('train')
    origdataset = imgDataset(results, data)
    increased_dataset = torch.utils.data.ConcatDataset([origdataset])
    numTrainSamples = int(len(increased_dataset) * TRAIN_SPLIT)
    numValSamples = int(len(increased_dataset) * VAL_SPLIT)

    (trainData, valData) = random_split(increased_dataset,
        [numTrainSamples, numValSamples],
        generator=torch.Generator())

    trainDataLoader = DataLoader(trainData, BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    valDataLoader = DataLoader(valData, BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


    def accuracy(out, labels):
        _,pred = torch.max(out, dim=1)
        return torch.sum(pred==labels).item()

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 128)
    net.fc = net.fc.cuda() if device else net.fc



    n_epochs = EPOCHS
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainData)
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(trainDataLoader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')