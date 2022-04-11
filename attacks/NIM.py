#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : NIM.py
@Author  : igeng
@Date    : 2022/4/10 16:12 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class NIM(Attack):
    """
    Nesterov Iterative Method (NIM) or Nesterov Iterative Fast Gradient Sign Method (NI-FGSM)
    Paper: 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    Paper link: https://arxiv.org/abs/1908.06281
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """
    def __init__(self, target_model, args):
        super(NIM, self).__init__("NIM", target_model)
        self.eps = args.nim_epsilon
        self.eps_iter = args.nim_eps_iter
        self.n_iters = args.nim_n_iters
        self.decay = args.nim_decay
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

        adv_examples = imgs.clone().detach()

        n_gradients = torch.zeros_like(adv_examples).detach().to(self.device)

        for step in range(self.n_iters):
            # print("PGD attack {} step!".format(step))
            adv_examples.requires_grad = True

            adv_examples = adv_examples + self.eps_iter * self.decay * n_gradients

            outputs = self.target_model(adv_examples)

            loss = F.cross_entropy(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_examples)[0]
            # n_gradients = self.decay * n_gradients + gradients
            n_gradients = self.decay * n_gradients + gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)

            adv_examples = adv_examples.detach() + self.eps_iter * n_gradients.sign()
            perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

        return adv_examples
