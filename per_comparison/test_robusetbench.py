#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : test_robusetbench.py
@Author  : igeng
@Date    : 2022/3/24 10:09 
@Descrip :
'''
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from robustbench.model_zoo.cifar10 import linf
import argparse
from attacks import *
import datetime

# model_list contains all cifar10-Linf models.
model_list = []
for k, _ in linf.items():
    print(k)
    model_list.append(k)
print(model_list)

x_test, y_test = load_cifar10(n_examples=50)

model = load_model(model_name= 'Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf').to("cuda")

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)

parser = argparse.ArgumentParser(description='Adversarial Attack Test')
parser.add_argument('--seed', default=199592, type=int)
parser.add_argument('--attack', default='PGD', type=str)
parser.add_argument('--model', default='smallcnn', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--iteration', default=100, type=int)
parser.add_argument('--momentum_decay', default=0.9, type=float)
parser.add_argument('--random_start', default=True, type=bool)
parser.add_argument('--norm_ord', default='Linf', type=str)
parser.add_argument('--eps_division', default=1e-10, type=float)
parser.add_argument('--attack_targeted', default=False, type=bool)
parser.add_argument('--decay', default=0.1, type=float)
# FGSM attack setting.
parser.add_argument('--fgsm_epsilon', default=2.0 / 255, type=float)
# PGD attack setting.
parser.add_argument('--pgd_epsilon', default=8.0 / 255, type=float)
parser.add_argument('--pgd_eps_step', default=2.0 / 255, type=float)
parser.add_argument('--pgd_n_steps', default=40, type=int)
args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("################## AdvRobust PGD is coming! ###################")
adversary = PGD(model, args)
start = datetime.datetime.now()
adv_images = adversary.perturb(x_test, y_test)
end = datetime.datetime.now()
acc = clean_accuracy(model, adv_images, y_test)
print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))
