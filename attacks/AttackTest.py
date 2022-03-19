#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : AttackTest.py
@Author  : igeng
@Date    : 2022/3/18 16:54 
@Descrip :
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import tqdm
import random
import numpy as np
from models import *

from FGSM import FGSM
from PGD import PGD


def attack_test(adversary, testloader, args, net):
    net.eval()
    adv_correct = 0
    total = 0
    batch_total = len(testloader)

    for batch_step, (imgs, labels) in enumerate(testloader):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        total += labels.size(0)
        adv_examples = adversary.perturb(imgs, labels)
        adv_outputs = net(adv_examples)
        _, predicted = adv_outputs.max(1)
        # _, predicted = torch.max(adv_outputs, 1)
        adv_correct += predicted.eq(labels).sum().item()

        acc = 100 * adv_correct / total

        print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
            (batch_step + 1) / batch_total * 100, acc), end='\r'
        )
    print('Model: {} is attacked by {}. The predict accuracy is {}.'.format(args.model, args.attack, acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack Test')
    parser.add_argument('--seed', default=199592, type=int)
    parser.add_argument('--attack', default='PGD', type=str)
    parser.add_argument('--model', default='smallcnn', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--iteration', default=100, type=int)
    # FGSM attack setting.
    parser.add_argument('--fgsm_epsilon', default=2.0 / 255, type=float)
    # PGD attack setting.
    parser.add_argument('--pgd_epsilon', default=0.3, type=float)
    parser.add_argument('--pgd_eps_step', default=2.0 / 255, type=float)
    parser.add_argument('--pgd_n_steps', default=40, type=int)
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Arguments for attack:')
    print(args)

    # Set random seed.
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Preparing attack data!')
    transforms_test = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('Building model {}!'.format(args.model))
    if args.model == 'smallcnn':
        net = SmallCNN()
    elif args.model == 'resnet18':
        net = ResNet18()
    elif args.model == 'wideresnet':
        net = WideResNet_28_10()
    else:
        raise ValueError('No such model...')

    print('Loading {} From pre_models!'.format(args.model))
    # pre_model = torch.load(os.path.join('../pre_models/', args.model))
    pre_model = torch.load('../pre_models/pgd_adversarial_training_smallcnn')
    net = torch.nn.DataParallel(net)
    net.load_state_dict(pre_model['net'])

    advesary = FGSM(net, args)
    # advesary = PGD(net, args)

    attack_test(advesary, test_loader, args, net)























