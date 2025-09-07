#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : MSPGD.py
@Author  : igeng
@Date    : 2022/3/31 22:45 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack
import numpy as np

class MSPGD(Attack):
    """
    Multi Space Project Gradient Dscent (Multi Space PGD)
    Distance Measure : Linf
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(MSPGD, self).__init__("MSPGD", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.n_steps = args.pgd_n_steps
        self.device = args.device
        self.random_start = args.random_start

    def _arctanh(self, imgs):
        x = torch.clamp(imgs, min=-1, max=1)
        # x = 0.99999999 * scaling
        return 2 * torch.sigmoid(10 * x) - 1

    def np_arctanh(self, eps):
        x = np.clip(eps, a_min=-1, a_max=1)
        # x = 0.99999999 * scaling
        return 2 / (1 + np.exp(-10 * x)) - 1

    def _scaler(self, x):
        # return 0.5 * (torch.tanh(imgs_atanh) + 1)
        return (1 / 10) * torch.log((1 + x) / (1 - x))

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

        # imgs_arctanh = self._arctanh(imgs) # 从[-1, 1]到(-inf, inf)
        eps_arctanh = self.np_arctanh(self.eps)
        eps_step_arctanh = self.np_arctanh(self.eps_step)

        adv_examples = imgs.clone().detach()
        perturbation = torch.zeros_like(adv_examples)

        if self.random_start:
            # adv_examples = adv_examples + torch.empty_like(adv_examples).uniform_(-self.eps, self.eps)
            # adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

            perturbation = perturbation + torch.empty_like(perturbation).uniform_(-eps_arctanh, eps_arctanh)
            # adv_examples = self._scaler(imgs_arctanh)

        for step in range(self.n_steps):
            # adv_examples.requires_grad = True
            perturbation.requires_grad = True
            imgs_perturbation = self._scaler(perturbation)
            imgs_perturbation = torch.clamp(imgs_perturbation, min=-self.eps, max=self.eps)
            outputs = self.target_model(imgs + imgs_perturbation)

            loss = F.cross_entropy(outputs, labels)
            # print(loss)

            gradients = torch.autograd.grad(loss, perturbation)[0]

            # adv_examples = adv_examples.detach() + self.eps_step * gradients.sign()
            # perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            # adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()
            # gradients_arc = self._arctanh(gradients)

            perturbation = perturbation.detach() + eps_step_arctanh * gradients.sign()
            perturbation = torch.clamp(perturbation, min=-eps_arctanh, max=eps_arctanh)

            adv_examples = imgs + self._scaler(perturbation).detach()

            if loss > best_loss:
                best_adv_examples = adv_examples
                best_loss = loss

        return best_adv_examples