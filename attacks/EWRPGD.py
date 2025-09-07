#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : EWRPGD.py
@Author  : igeng
@Date    : 2022/3/30 9:04 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

import numpy as np

class EWRPGD(Attack):
    """
    Efficient Warm Restart Project Gradient Dscent (EWR-PGD)
    Distance Measure : Linf
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(EWRPGD, self).__init__("EWRPGD", target_model)
        self.eps = args.wrpgd_epsilon
        self.eps_step = args.wrpgd_eps_step
        self.n_steps = args.wrpgd_n_steps
        self.n_restarts = args.wrpgd_n_restarts
        self.device = args.device
        self.random_start = args.random_start
        self.T = args.wrpgd_T

    def perturb(self, imgs, labels):
        """
        Override perturb function in Attack class.
        :param imgs: attacked benign input
        :param labels: orginal labels of benign input
        :return: adversarial examples
        """
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        best_loss = 0
        best_adv_examples = imgs.clone().detach()

        eps_step_max = self.eps
        eps_step_min = eps_step_max / self.T

        adv_examples = imgs.clone().detach()

        o_outputs = self.target_model(imgs)

        for i in range(self.n_restarts):

            # cold restart
            # adv_examples = imgs.clone().detach()

            if self.random_start:
                adv_examples = best_adv_examples + \
                               torch.empty_like(best_adv_examples).uniform_(-self.eps, self.eps)
                adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

            weight = torch.empty_like(o_outputs).uniform_(-1, 1)
            labels_onehot = F.one_hot(labels, 10)
            wd = weight - weight * labels_onehot - labels_onehot

            for k in range(2):
                adv_examples.requires_grad = True
                outputs = self.target_model(adv_examples) * wd
                loss = F.cross_entropy(outputs, labels)
                gradients_k = torch.autograd.grad(loss, adv_examples)[0]
                adv_examples = adv_examples.detach() + self.eps_step * gradients_k.sign()
                perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
                adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

            for step in range(self.n_steps):
                eps_step_t = 0.5 * (eps_step_max - eps_step_min) * (
                            1 + np.cos(((step % self.T) / self.T) * np.pi)) + eps_step_min

                adv_examples.requires_grad = True
                outputs = self.target_model(adv_examples)

                loss = F.cross_entropy(outputs, labels)

                gradients = torch.autograd.grad(loss, adv_examples)[0]

                adv_examples = adv_examples.detach() + eps_step_t * gradients.sign()
                perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
                adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

                if loss > best_loss:
                    best_adv_examples = adv_examples
                    best_loss = loss
                    # print("best loss is: ")
                    # print(best_loss)

        return best_adv_examples
