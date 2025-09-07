#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : WRNMM_equal.py
@Author  : igeng
@Date    : 2022/3/28 23:58 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class WRNM_PGD_equal(Attack):
    """
    Warm Restart Nesterov Momentum PGD version 2.
    Paper link:
    :argument:
    :argument:
    """

    def __init__(self, target_model, args):
        super(WRNM_PGD_equal, self).__init__("WRNM_PGD_equal", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.n_steps = args.pgd_n_steps
        self.decay = args.decay
        self.device = args.device
        self.random_start = args.random_start

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

        if self.random_start:
            adv_examples = adv_examples + torch.empty_like(adv_examples).uniform_(-self.eps, self.eps)
            adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

        pre_n_gradients = torch.empty_like(adv_examples)

        for step in range(self.n_steps):
            # print("PGD attack {} step!".format(step))
            adv_examples.requires_grad = True

            outputs = self.target_model(adv_examples)

            loss = F.cross_entropy(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_examples)[0]
            gradients = gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)

            n_gradients = self.decay * pre_n_gradients + gradients

            adv_examples = adv_examples.detach() + self.eps_step * self.decay * self.decay * pre_n_gradients + self.eps_step * (1 + self.decay) * gradients.sign()
            perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

            pre_n_gradients = n_gradients

        return adv_examples