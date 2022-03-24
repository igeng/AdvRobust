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

# x_test, y_test = load_cifar10(n_examples=50)

for k, _ in linf.items():
    model = load_model(model_name= k, dataset='cifar10', threat_model='Linf')

# import foolbox as fb
# fmodel = fb.PyTorchModel(model, bounds=(0, 1))
#
# _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
# print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))