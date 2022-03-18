#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : PGD.py
@Author  : igeng
@Date    : 2022/3/18 16:16 
@Descrip :
'''
import torch
import torch.nn as nn
from Attacks import Attack

class PGD(Attack):
    """
    Project Gradient Dscent (PGD)
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(PGD, self).__init__()
        self.eps = args.epsilon
        self.device = args.device
        self.target_model = target_model

    def perturb(self, imgs, labels):
        """
        Override perturb function in Attack class.
        :param imgs: attacked benign input
        :param labels: orginal labels of benign input
        :return:
        """
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        imgs.requires_grad = True

        outputs = self.target_model(imgs)
        criterion = nn.CrossEntropyLoss
        loss = criterion(outputs, labels)

        gradients = torch.autograd.grad(loss, imgs)[0]
        # consider eps is a vector ?
        adv_examples = imgs + (self.eps * gradients.sign())
        adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

        return adv_examples