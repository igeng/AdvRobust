#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : FGNM.py
@Author  : igeng
@Date    : 2022/3/29 16:52 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class FGNM(Attack):
    """
    Fast Gradient Non-Sign Method (FGNM)
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """
    def __init__(self, target_model, args):
        super(FGNM, self).__init__("FGNM", target_model)
        self.eps = args.fgsm_epsilon
        self.device = args.device
        self.alpha = args.fgnm_alpha

    def perturb(self, imgs, labels):
        """
        Override perturb function in Attack class.
        :param imgs: attacked benign input
        :param labels: orginal labels of benign input
        :return: adversarial examples
        """
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        imgs.requires_grad = True
        outputs = self.target_model(imgs)

        loss = nn.CrossEntropyLoss()(outputs, labels)

        gradients = torch.autograd.grad(loss, [imgs])[0]

        # 对抗扰动步长系数
        zeta = (torch.norm(gradients.sign(), p=2, dim=(2, 3), keepdim=True) /
                torch.norm(gradients, p=2, dim=(2, 3), keepdim=True)) * torch.ones(imgs.shape).to(self.device)

        adv_examples = imgs + self.eps * zeta * gradients
        adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

        return adv_examples