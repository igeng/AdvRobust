#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : performance_comparison.py
@Author  : igeng
@Date    : 2022/3/24 13:37 
@Descrip :
'''
# 1. Load CIAFR10
print("############### 1. Load CIAFR10 ###############")
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from utils import l2_distance
from robustbench.model_zoo.cifar10 import linf

images, labels = load_cifar10(n_examples=50, data_dir='../data')
device = "cuda"

# 2. Standard Accuracy
print("############### 2. Standard Accuracy ###############")
# model_list = ['Standard', 'Wong2020Fast', 'Rice2020Overfitting']

model_list = []
for k, _ in linf.items():
    # model = load_model(model_name= k, dataset='cifar10', threat_model='Linf')
    print(k)
    model_list.append(k)
print(model_list)

for model_name in model_list:
    model = load_model(model_name, model_dir='../models', norm='Linf').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(acc))

# 3. Torchattacks, Foolbox and ART
print("############### 3. Torchattacks, Foolbox and ART ###############")
import datetime
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim

# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))

# https://github.com/bethgelab/foolbox
import foolbox as fb
print("foolbox %s"%(fb.__version__))

# https://github.com/IBM/adversarial-robustness-toolbox
# import art
# import art.attacks.evasion as evasion
# from art.classifiers import PyTorchClassifier
# print("art %s"%(art.__version__))

# 3.1. Linf
# FGSM
print("############### 3.1. Linf ###############")
print("############### FGSM ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name, model_dir='../models', norm='Linf').to(device)

    print("- Torchattacks")
    atk = torchattacks.FGSM(model, eps=8 / 255)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.LinfFastGradientAttack(random_start=False)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda:0'), labels.to('cuda:0'), epsilons=8 / 255)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.FastGradientMethod(norm=np.inf, batch_size=50,
    #                                  estimator=classifier, eps=8 / 255)
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print()

# BIM
print("############### 3.1. Linf ###############")
print("############### BIM ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name, model_dir='../models', norm='Linf').to(device)

    print("- Torchattacks")
    atk = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.LinfBasicIterativeAttack(abs_stepsize=2 / 255, steps=10, random_start=False)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda:0'), labels.to('cuda:0'), epsilons=8 / 255)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.BasicIterativeMethod(batch_size=50,
    #                                    estimator=classifier, eps=8 / 255,
    #                                    eps_step=2 / 255, max_iter=10)
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print()

# PGD
print("############### 3.1. Linf ###############")
print("############### PGD ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name,model_dir='../models', norm='Linf').to(device)

    print("- Torchattacks")
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=False)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.LinfPGD(abs_stepsize=2 / 255, steps=10, random_start=False)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda:0'), labels.to('cuda:0'), epsilons=8 / 255)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.ProjectedGradientDescent(batch_size=50, num_random_init=0,
    #                                        estimator=classifier, eps=8 / 255,
    #                                        eps_step=2 / 255, max_iter=10)
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # print('- Robust Acc: {} ({} ms)'.format(acc, int((end - start).total_seconds() * 1000)))

    print()

# 3.2. L2
# DeepFool
print("############### 3.2. L2 ###############")
print("############### DeepFool ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name, model_dir='../models', norm='Linf').to(device)

    print("- Torchattacks")
    atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2DeepFoolAttack(steps=50, candidates=10, overshoot=0.02)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda'), labels.to('cuda'), epsilons=1)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.DeepFool(classifier=classifier, max_iter=50,
    #                        batch_size=50)
    #
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # l2 = l2_distance(model, images, adv_images, labels, device=device)
    # print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
    #                                                      int((end - start).total_seconds() * 1000)))

    print()

# CW
print("############### 3.2. L2 ###############")
print("############### CW ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name, model_dir='../models', norm='Linf').to(device)

    print("- Torchattacks")
    atk = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, initial_const=1,
                                           confidence=0, steps=100, stepsize=0.01)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda'), labels.to('cuda'), epsilons=1)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.CarliniL2Method(batch_size=50, classifier=classifier,
    #                               binary_search_steps=1, initial_const=1,
    #                               confidence=0, max_iter=100,
    #                               learning_rate=0.01)
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # l2 = l2_distance(model, images, adv_images, labels, device=device)
    # print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
    #                                                      int((end - start).total_seconds() * 1000)))

    print()

# PGD L2
print("############### 3.2. L2 ###############")
print("############### PGD L2 ###############")
for model_name in model_list:
    print('Model: {}'.format(model_name))
    model = load_model(model_name, model_dir='../models', norm='Linf').cuda()

    print("- Torchattacks")
    atk = torchattacks.PGDL2(model, eps=128 / 255, alpha=15 / 255, steps=10, random_start=False)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2PGD(abs_stepsize=15 / 255, steps=10, random_start=False)
    start = datetime.datetime.now()
    _, adv_images, _ = atk(fmodel, images.to('cuda:0'), labels.to('cuda:0'), epsilons=128 / 255)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    l2 = l2_distance(model, images, adv_images, labels, device=device)
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                         int((end - start).total_seconds() * 1000)))

    # print("- ART")
    # classifier = PyTorchClassifier(model=model, clip_values=(0, 1),
    #                                loss=nn.CrossEntropyLoss(),
    #                                optimizer=optim.Adam(model.parameters(), lr=0.01),
    #                                input_shape=(3, 32, 32), nb_classes=10)
    # atk = evasion.ProjectedGradientDescent(batch_size=50, num_random_init=0,
    #                                        norm=2, estimator=classifier, eps=128 / 255,
    #                                        eps_step=15 / 255, max_iter=10)
    # start = datetime.datetime.now()
    # adv_images = torch.tensor(atk.generate(images, labels)).to(device)
    # end = datetime.datetime.now()
    # acc = clean_accuracy(model, adv_images, labels)
    # l2 = l2_distance(model, images, adv_images, labels, device=device)
    # print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
    #                                                      int((end - start).total_seconds() * 1000)))

    print()



