#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : AttackTorchAttacksTest.py
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

def attack_test(adversary, testloader, args, net, lib_type):
    net.eval()
    adv_correct = 0
    total = 0
    batch_total = len(testloader)

    for batch_step, (imgs, labels) in enumerate(testloader):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        total += labels.size(0)
        adv_examples = imgs
        if lib_type == 'AdvRobust':
            # print("The {} step attack from AdvRobust!".format(batch_step))
            adv_examples = adversary.perturb(imgs, labels)
        elif lib_type == 'TorchAttacks':
            # print("The {} step attack from TorchAttacks!".format(batch_step))
            adv_examples = adversary.forward(imgs, labels)
        elif lib_type == 'Clean':
            print("The {} step clean test!".format(batch_step))
        adv_outputs = net(adv_examples)
        _, predicted = adv_outputs.max(1)
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
    parser.add_argument('--attack_targeted', default=True, type=bool)
    parser.add_argument('--decay', default=0.5, type=float)
    # FGSM attack setting.
    parser.add_argument('--fgsm_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--fgnm_alpha', default=8.0 / 255, type=float)
    # PGD attack setting.
    parser.add_argument('--pgd_epsilon', default=0.031, type=float)
    parser.add_argument('--pgd_eps_step', default=0.003, type=float)
    parser.add_argument('--pgd_n_steps', default=20, type=int)
    # BIM(I-FGSM) attack setting.
    parser.add_argument('--bim_epsilon', default=4.0/255, type=float)
    parser.add_argument('--bim_eps_iter', default=1.0 / 255, type=float)
    parser.add_argument('--bim_n_iters', default=10, type=int)
    # MIM(MI-FGSM) attack setting.
    parser.add_argument('--mim_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--mim_eps_iter', default=2.0 / 255, type=float)
    parser.add_argument('--mim_n_iters', default=5, type=int)
    # NIM(NI-FGSM) attack setting.
    parser.add_argument('--nim_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--nim_eps_iter', default=1.0 / 255, type=float)
    parser.add_argument('--nim_n_iters', default=10, type=int)
    parser.add_argument('--nim_decay', default=1.0, type=float)
    # CW attack setting.
    parser.add_argument('--cw_c', default=1e+100, type=float)
    parser.add_argument('--cw_k', default=-10000.0, type=float)
    parser.add_argument('--cw_n_iters', default=1000, type=int)
    parser.add_argument('--cw_lr', default=0.0001, type=float)
    parser.add_argument('--cw_binary_search_steps', default=9, type=int)
    # Nesterov attack setting.
    parser.add_argument('--nes_lr', default=0.1, type=float)
    # LookAhead attack setting.
    parser.add_argument('--la_steps', default=8, type=int)
    parser.add_argument('--la_k', default=5, type=int)
    parser.add_argument('--la_alpha', default=0.99, type=float)
    parser.add_argument('--la_decay', default=0.9, type=float)
    parser.add_argument('--la_exp_decay', default=0.999, type=float)
    # WRPGD
    parser.add_argument('--wrpgd_epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--wrpgd_eps_step', default=2.0 / 255, type=float)
    parser.add_argument('--wrpgd_n_steps', default=40, type=int)
    parser.add_argument('--wrpgd_n_restarts', default=1, type=int)
    parser.add_argument('--wrpgd_T', default=10, type=int)

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
        model = 'Zhang2019Theoretically'
        net = load_model(model_name=model, model_dir='./per_comparison/models', dataset='cifar10',
                         threat_model='Linf').to("cuda")
        net = torch.nn.DataParallel(net)
        args.model = model

        for i in range(14, 17):
            advesary = None
            if i == 0:
                print("#######################################################################################")
                print("################################# VANILA attack   #####################################")
                print("#######################################################################################")
                print("####### Clean accuracy #######")
                advesary_com = torchattacks.VANILA(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
            elif i == 1:
                print("#######################################################################################")
                print("################################ FGSM + X  attack #####################################")
                print("#######################################################################################")
                print("####### TorchAttacks FGSM attack #######")
                advesary_com = torchattacks.FGSM(net, eps=8.0 / 255) # mnist eps=0.3
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks R-FGSM attack #######")
                advesary_com = torchattacks.RFGSM(net, eps=8.0 / 255, alpha=4.0 / 255)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks DIFGSM attack #######")
                advesary_com = torchattacks.DIFGSM(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks FFGSM attack #######")
                advesary_com = torchattacks.FFGSM(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks MIFGSM attack #######")
                advesary_com = torchattacks.MIFGSM(net, steps=20)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks TIFGSM attack #######")
                advesary_com = torchattacks.TIFGSM(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks BIM attack #######")
                advesary_com = torchattacks.BIM(net, eps=8.0 / 255, alpha=2.0 / 255)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
            elif i == 2:
                print("#######################################################################################")
                print("################################ PGD + X  attack #####################################")
                print("#######################################################################################")
                print("####### TorchAttacks PGD attack with 7 steps #######")
                advesary_com = torchattacks.PGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=7)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks PGD attack with 20 steps #######")
                advesary_com = torchattacks.PGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=20)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks PGD attack with 40 steps #######")
                advesary_com = torchattacks.PGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=40)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks PGDL2 attack #######")
                advesary_com = torchattacks.PGDL2(net, eps=8.0 / 255, alpha=2.0 / 255, steps=40)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks Ultimate MOMENTUM PGD attack with CELoss #######")
                advesary_com = torchattacks.UPGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=40, random_start=True)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks Ultimate MOMENTUM PGD attack with MarginLoss #######")
                advesary_com = torchattacks.UPGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=40, loss='margin', random_start=True)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks Ultimate MOMENTUM PGD attack with DLRLoss #######")
                advesary_com = torchattacks.UPGD(net, eps=8.0 / 255, alpha=2.0 / 255, steps=40, loss='dlr', random_start=True)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
            elif i == 3:
                print("#######################################################################################")
                print("############################ Iteration Optimizer attack ###############################")
                print("#######################################################################################")
                print("####### TorchAttacks CW attack #######")
                advesary_com = torchattacks.CW(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks DeepFool attack #######")
                advesary_com = torchattacks.DeepFool(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
            elif i == 4:
                print("#######################################################################################")
                print("################################# AutoAttack attack ###################################")
                print("#######################################################################################")
                print("####### TorchAttacks APGD-CE attack #######")
                advesary_com = torchattacks.APGD(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks APGD-DLR attack #######")
                advesary_com = torchattacks.APGD(net, loss='dlr')
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks FAB attack #######")
                advesary_com = torchattacks.FAB(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
                print("####### TorchAttacks Square attack #######")
                advesary_com = torchattacks.Square(net)
                attack_test(advesary_com, test_loader, args, net, 'TorchAttacks')
            elif i == 5:
                print("#######################################################################################")
                print("######################## Nesterov Accelerate Gradient attack ##########################")
                print("#######################################################################################")
                print("####### AdvRobust WRNM_PGD_Vanila attack #######")
                advesary_com = WRNM_PGD_Vanila(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
                print("####### AdvRobust WRNM_PGD_LTG attack #######")
                advesary_com = WRNM_PGD_LTG(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
                print("####### AdvRobust WRNMM attack #######")
                advesary_com = WRNMM(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
                print("####### AdvRobust WRNM_PGD_equal attack #######")
                advesary_com = WRNM_PGD_equal(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 6:
                print("#######################################################################################")
                print("################################## LookAhead attack ###################################")
                print("#######################################################################################")
                print("####### AdvRobust LookAhead attack #######")
                advesary_com = LookAhead(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 7:
                print("#######################################################################################")
                print("################################ LookAheadAdam attack #################################")
                print("#######################################################################################")
                print("####### AdvRobust LookAheadAdam attack #######")
                advesary_com = LookAheadAdam(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 8:
                print("#######################################################################################")
                print("################################ FGNM + X  attack #####################################")
                print("#######################################################################################")
                print("####### AdvRobust FGNM attack #######")
                advesary_com = FGNM(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 9:
                print("#######################################################################################")
                print("################################# CLEAN  attack #######################################")
                print("#######################################################################################")
                print("####### AdvRobust CLEAN attack #######")
                advesary_com = CLEAN(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 10:
                print("#######################################################################################")
                print("################################ PGND + X  attack #####################################")
                print("#######################################################################################")
                print("####### AdvRobust PGND attack #######")
                advesary_com = PGND(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 11:
                print("#######################################################################################")
                print("################################ WRPGD + X attack #####################################")
                print("#######################################################################################")
                print("####### AdvRobust WRPGD attack #######")
                advesary_com = WRPGD(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 12:
                print("#######################################################################################")
                print("################################ EWRPGD + X attack ####################################")
                print("#######################################################################################")
                print("####### AdvRobust EWRPGD attack #######")
                advesary_com = EWRPGD(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 13:
                print("#######################################################################################")
                print("################################# MTPGD + X attack ####################################")
                print("#######################################################################################")
                print("####### AdvRobust MTPGD attack #######")
                advesary_com = MTPGD(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 14:
                print("#######################################################################################")
                print("################################### PGD + X attack ####################################")
                print("#######################################################################################")
                print("####### AdvRobust PGD attack #######")
                advesary_com = PGD(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 15:
                print("#######################################################################################")
                print("################################## MSPGD + X attack ###################################")
                print("#######################################################################################")
                print("####### AdvRobust MSPGD attack #######")
                advesary_com = MSPGD(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            elif i == 16:
                print("#######################################################################################")
                print("################################ NIFGSM + X  attack #####################################")
                print("#######################################################################################")
                print("####### AdvRobust NIFGSM attack #######")
                advesary_com = NIM(net, args)
                attack_test(advesary_com, test_loader, args, net, 'AdvRobust')
            else:
                print("No attack is running, bye!")
                break