#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust
@File    : AttackAdvRobustTest.py
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

from attacks import *
import torchattacks

from robustbench.utils import load_model
from robustbench.model_zoo.cifar10 import linf

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
    parser.add_argument('--model', default='Carmon2019Unlabeled', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--iteration', default=100, type=int)
    parser.add_argument('--momentum_decay', default=0.9, type=float)
    parser.add_argument('--random_start', default=True, type=bool)
    parser.add_argument('--norm_ord', default='Linf', type=str)
    parser.add_argument('--eps_division', default=1e-10, type=float)
    parser.add_argument('--attack_targeted', default=False, type=bool)
    parser.add_argument('--decay', default=0.01, type=float)
    # FGSM attack setting.
    parser.add_argument('--fgsm_epsilon', default=2.0 / 255, type=float)
    # PGD attack setting.
    parser.add_argument('--pgd_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--pgd_eps_step', default=2.0 / 255, type=float)
    parser.add_argument('--pgd_n_steps', default=40, type=int)
    # BIM(I-FGSM) attack setting.
    parser.add_argument('--bim_epsilon', default=4.0/255, type=float)
    parser.add_argument('--bim_eps_iter', default=1.0 / 255, type=float)
    parser.add_argument('--bim_n_iters', default=10, type=int)
    # MI-FGSM attack setting.
    parser.add_argument('--mim_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--mim_eps_iter', default=2.0 / 255, type=float)
    parser.add_argument('--mim_n_iters', default=5, type=int)
    # CW attack setting.
    parser.add_argument('--cw_c', default=1e+100, type=float)
    parser.add_argument('--cw_k', default=-10000.0, type=float)
    parser.add_argument('--cw_n_iters', default=1000, type=int)
    parser.add_argument('--cw_lr', default=0.0001, type=float)
    parser.add_argument('--cw_binary_search_steps', default=9, type=int)

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
    test_set = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model_list contains all cifar10-Linf models.
    model_list = []
    for model, _ in linf.items():
        print('Loading {} from pre_models in RobustBench!'.format(model))
        net = load_model(model_name = model, model_dir='./per_comparison/models',dataset='cifar10', threat_model='Linf').to("cuda")
        net = torch.nn.DataParallel(net)
        args.model = model

        for i in range(1, 2):
            adversary = Attack(net, args)
            if i == 0:
                print("####### AdvRobust FGSM attack #######")
                adversary = FGSM(net, args)
            elif i == 1:
                print("####### AdvRobust PGD attack #######")
                adversary = PGD(net, args)
            elif i == 2:
                print("####### AdvRobust PGDL2 attack #######")
                adversary = PGDL2(net, args)
            elif i == 3:
                print("####### AdvRobust BIM attack #######")
                adversary = BIM(net, args)
            elif i == 4:
                print("####### AdvRobust MIM attack #######")
                adversary = MIM(net, args)
            elif i == 5:
                print("####### AdvRobust CW attack #######")
                adversary = CW(net, args)
            elif i == 6:
                print("####### AdvRobust CWL2 attack #######")
                adversary = CWL2(net, args)
            elif i == 7:
                print("####### AdvRobust WRNM_PGD_Vanila attack #######")
                adversary = WRNM_PGD_Vanila(net, args)
            elif i == 8:
                print("####### AdvRobust WRNM_PGD_LTG attack #######")
                adversary = WRNM_PGD_LTG(net, args)
            else:
                print("No attack is running, bye!")
                break

            attack_test(adversary, test_loader, args, net)