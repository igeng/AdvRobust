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
from robustbench.utils import load_model
from robustbench.model_zoo.cifar10 import linf

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