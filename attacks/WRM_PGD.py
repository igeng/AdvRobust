#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : PGD.py
@Author  : igeng
@Date    : 2022/3/18 16:16 
@Descrip :
####### AdvRobust PGD attack #######
Model: Andriushchenko2020Understanding is attacked by PGD.
The predict accuracy is 47.42.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class WRM_PGD(Attack):
    """
    Warm Restart Momentum Project Gradient Dscent (WRM_PGD)
    Distance Measure : Linf
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(WRM_PGD, self).__init__("WRM_PGD", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.n_steps = args.pgd_n_steps
        self.device = args.device
        self.random_start = args.random_start
        self.C = 0.0

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

        for step in range(self.n_steps):

            # adv_examples = best_adv_examples

            adv_examples.requires_grad = True
            outputs = self.target_model(adv_examples)

            # CE loss
            loss = F.cross_entropy(outputs, labels)
            # label_mask = F.one_hot(labels, 10)
            # loss = F.cross_entropy(label_mask * outputs, labels) - F.cross_entropy((1-label_mask * outputs), labels)
            # CW loss
            # label_mask = F.one_hot(labels, 10)
            # correct_logit = torch.mean(torch.sum(label_mask * outputs, dim=1))
            # wrong_logit = torch.mean(torch.max((1 - label_mask) * outputs, dim=1)[0])
            # loss = - F.relu(correct_logit - wrong_logit + self.C)
            # print(loss)

            gradients = torch.autograd.grad(loss, adv_examples)[0]

            adv_examples = adv_examples.detach() + self.eps_step * gradients.sign()
            perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

            # if loss > best_loss:
            #     best_adv_examples = adv_examples
            #     best_loss = loss

        return adv_examples