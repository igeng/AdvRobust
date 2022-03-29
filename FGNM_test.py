#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : FGNM_test.py
@Author  : igeng
@Date    : 2022/3/29 16:48 
@Descrip :
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
            kernel_size=5, stride=1, padding=2),   #  output: (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # (16, 14, 14)
        )
        self.Sq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
            kernel_size=5, stride=1, padding=2),  # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.Sq1(x)
        x = self.Sq2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def FGM_attack(inputs, targets, net, alpha, epsilon, attack_type):
    delta = torch.zeros_like(inputs)
    delta.requires_grad = True
    outputs = net(inputs + delta)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()
    grad = delta.grad.detach()
    if type == 'FGSN':
        zeta = (torch.norm(inputs, p=0, dim=(2,3), keepdim=True)
               / torch.norm(inputs, p=2, dim=(2,3), keepdim=True)) * torch.ones(inputs.shape)
        delta.data = torch.clamp(delta + alpha * zeta * grad,
            -epsilon, epsilon)
    else:
        delta.data = torch.clamp(delta + alpha * torch.sign(grad),
             -epsilon, epsilon)
    delta = delta.detach()
    return delta

def main():
    alpha = 0.2
    epsilon = 0.5
    total = 0
    correct1 = 0
    correct2 = 0
    model = CNN()
    model.load_state_dict(torch.load('pre_models/model.pt'))
    use_cuda = torch.cuda.is_available()
    mnist_train = datasets.MNIST("datasets", train=True,
    download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train,
    batch_size= 5, shuffle=True)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        total += targets.size(0)

        delta1 = FGM_attack(inputs, targets, model, alpha, epsilon, 'FGNM')
        adv_image1 = torch.clamp(inputs + delta1, 0, 1)
        outputs1 = model(adv_image1)
        _, predicted1 = torch.max(outputs1.data, 1)
        correct1 += predicted1.eq(targets.data).cpu().sum().item()
        print('The FGNM accuracy:', correct1, total, correct1/total)

        delta2 = FGM_attack(inputs, targets, model, alpha, epsilon, 'FGSM')
        adv_images2 = torch.clamp(inputs + delta1, 0, 1)
        outputs2 = model(adv_images2)
        _, predicted2 = torch.max(outputs2.data, 1)
        correct2 += predicted2.eq(targets.data).cpu().sum().item()
        print('The FGSM accuracy:', correct2, total, correct2/total)
    print('The FGNM accuracy:', correct1)
    print('The FGSM accuracy:', correct2)

if __name__ == '__main__':
    main()