#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : LookAheadAdam.py
@Author  : igeng
@Date    : 2022/3/29 10:50 
@Descrip :
'''
import torch
import torch.nn.functional as F
from .Attacks import Attack

import math

class LookAheadAdam(Attack):
    """
    LookAhead and Adam Optimization Attack
    Distance Measure : Linf
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(LookAheadAdam, self).__init__("LookAheadAdam", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.steps = args.la_steps
        self.device = args.device
        self.random_start = args.random_start
        self.k = args.la_k
        self.alpha = args.la_alpha
        self.decay = args.la_decay
        self.exp_deacy = args.la_exp_decay

    def perturb(self, imgs, labels):
        """
        Override perturb function in Attack class.
        :param imgs: attacked benign input
        :param labels: orginal labels of benign input
        :return: adversarial examples
        """
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        slow_adv_examples = imgs.clone().detach()
        fast_adv_examples = imgs.clone().detach()


        if self.random_start:
            slow_adv_examples = slow_adv_examples + torch.empty_like(slow_adv_examples).uniform_(-self.eps, self.eps)
            slow_adv_examples = torch.clamp(slow_adv_examples, min=0, max=1).detach()
            # fast_adv_examples = fast_adv_examples + torch.empty_like(fast_adv_examples).uniform_(-self.eps, self.eps)
            # fast_adv_examples = torch.clamp(fast_adv_examples, min=0, max=1).detach()

        momentum = torch.empty_like(imgs)
        v = torch.empty_like(imgs)

        for step in range(self.steps):
            fast_adv_examples = slow_adv_examples

            for i in range(self.k):
                fast_adv_examples.requires_grad = True
                outputs = self.target_model(fast_adv_examples)

                loss = F.cross_entropy(outputs, labels)

                gradients = torch.autograd.grad(loss, fast_adv_examples)[0]
                # gradients = gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)

                momentum = self.decay * momentum + (1 - self.decay) * gradients
                v = self.exp_deacy * v + (1 - self.exp_deacy) * torch.pow(gradients, 2)
                momentum_hat = momentum / (1 - math.pow(self.decay, i))
                v_hat = v / (1 - math.pow(self.exp_deacy, i))
                fast_adv_examples = fast_adv_examples.detach() + self.eps_step * momentum_hat / (torch.pow(v_hat, 2) + 1e-8)

                # fast_adv_examples = fast_adv_examples.detach() + self.eps_step * gradients.sign()
                perturbation = torch.clamp(fast_adv_examples - imgs, min=-self.eps, max=self.eps)
                fast_adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

            slow_adv_examples = slow_adv_examples + self.alpha * (fast_adv_examples - slow_adv_examples)

            slow_perturbation = torch.clamp(slow_adv_examples - imgs, min=-self.eps, max=self.eps)
            slow_adv_examples = torch.clamp(imgs + slow_perturbation, min=0, max=1).detach()

        return slow_adv_examples