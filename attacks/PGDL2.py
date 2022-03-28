#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : PGDL2.py
@Author  : igeng
@Date    : 2022/3/21 9:51 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class PGDL2(Attack):
    """
    Project Gradient Dscent (PGD)
    Distance Measure : L2
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    :argument: eps_step {float} -- Size of perturbation in each step.
    :argument: n_steps {int} -- Number of steps. 
    :argument: device {str} -- cuda or cpu.
    :argument: random_start {bool} -- Initialize random perturbation.
    :argument: 
    """

    def __init__(self, target_model, args):
        super(PGDL2, self).__init__("PGDL2", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.n_steps = args.pgd_n_steps
        self.device = args.device
        self.random_start = args.random_start
        self.eps_ord = args.norm_ord
        self.eps_division = args.eps_division

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
        batch_size = len(imgs)

        if self.random_start:
            adv_examples = adv_examples + torch.empty_like(adv_examples).uniform_(-self.eps, self.eps)
            adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

        for step in range(self.n_steps):
            # print("PGD attack {} step!".format(step))
            adv_examples.requires_grad = True
            outputs = self.target_model(adv_examples)

            loss = F.cross_entropy(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_examples)[0] # 100 * 3 * 32 * 32
            gradients_norms = torch.norm(gradients.view(batch_size, -1), p=2, dim=1) + self.eps_division # 100 * 1
            gradients = gradients / gradients_norms.view(batch_size, 1, 1, 1) # 100 * 1 * 1 * 1
            # consider eps is a vector ?
            # adv_examples = adv_examples.detach() + self.eps_step * gradients.sign()
            adv_examples = adv_examples.detach() + self.eps_step * gradients

            perturbation = adv_examples - imgs
            perturbation_norms = torch.norm(perturbation.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / perturbation_norms
            factor = torch.min(factor, torch.ones_like(perturbation_norms))
            perturbation = perturbation * factor.view(-1, 1, 1, 1)

            adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

        return adv_examples
