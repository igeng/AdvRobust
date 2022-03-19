#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust
@File    : FGSM.py
@Author  : igeng
@Date    : 2022/3/18 11:05
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Attacks import Attack

class FGSM(Attack):
    """
    Fast Gradient Sign Method (FGSM)
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """
    def __init__(self, target_model, args):
        super(FGSM, self).__init__(target_model)
        self.eps = args.epsilon
        self.device = args.device

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
        # criterion = nn.CrossEntropyLoss
        loss = F.cross_entropy(outputs, labels)

        gradients = torch.autograd.grad(loss, [imgs])[0]
        # consider eps is a vector ?
        adv_examples = torch.clamp(imgs + (self.eps * gradients.sign()), min=0, max=1).detach()

        return adv_examples