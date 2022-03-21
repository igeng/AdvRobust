#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : MIM.py
@Author  : igeng
@Date    : 2022/3/21 18:57 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class MIM(Attack):
    """
    Momentum Iterative Method (MIM) or Momentum Iterative Fast Gradient Sign Method (MI-FGSM)
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """
    def __init__(self, target_model, args):
        super(MIM, self).__init__("MIM", target_model)
        self.eps = args.mim_epsilon
        self.eps_iter = args.mim_eps_iter
        self.n_iters = args.mim_n_iters
        self.decay = args.momentum_decay
        self.device = args.device

    def perturb(self, imgs, labels):
        """
        Override perturb function in Attack class.
        :param imgs: attacked benign input
        :param labels: orginal labels of benign input
        :return: adversarial examples
        """
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(imgs).detach().to(self.device)

        adv_examples = imgs.clone().detach()

        for step in range(self.n_iters):
            # print("PGD attack {} step!".format(step))
            adv_examples.requires_grad = True
            outputs = self.target_model(adv_examples)

            loss = F.cross_entropy(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_examples)[0]

            gradients = gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)
            gradients = gradients + self.decay * momentum
            momentum = gradients

            adv_examples = adv_examples.detach() + self.eps_iter * gradients.sign()
            perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            adv_examples = torch.clamp(adv_examples + perturbation, min=0, max=1).detach()

        return adv_examples