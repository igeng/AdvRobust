#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : LookAhead.py
@Author  : igeng
@Date    : 2022/3/26 18:49 
@Descrip :
'''
import torch
import torch.nn.functional as F
from .Attacks import Attack

class LookAhead(Attack):
    """
    LookAhead Optimization Attack
    Distance Measure : Linf
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(LookAhead, self).__init__("LookAhead", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.steps = args.la_steps
        self.device = args.device
        self.random_start = args.random_start
        self.k = args.la_k
        self.alpha = args.la_alpha
        self.decay = args.decay

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


        if self.random_start:
            slow_adv_examples = slow_adv_examples + torch.empty_like(slow_adv_examples).uniform_(-self.eps, self.eps)
            slow_adv_examples = torch.clamp(slow_adv_examples, min=0, max=1).detach()

        # momentum = torch.empty_like(imgs)

        for step in range(self.steps):
            fast_adv_examples = slow_adv_examples
            best_fast_adv_examples = slow_adv_examples
            best_loss = 0

            for i in range(self.k):
                fast_adv_examples.requires_grad = True
                outputs = self.target_model(fast_adv_examples)

                loss = F.cross_entropy(outputs, labels)

                gradients = torch.autograd.grad(loss, fast_adv_examples)[0]
                # gradients = gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)

                # momentum = self.decay * momentum + (1 - self.decay) * gradients
                # mask = torch.empty_like(outputs).uniform_(-1, 1)
                # for index in range(len(labels)):
                #     mask[index][labels[index]]

                fast_adv_examples = fast_adv_examples.detach() + self.eps_step * gradients.sign()

                # fast_adv_examples = fast_adv_examples.detach() + self.eps_step * momentum.sign()

                perturbation = torch.clamp(fast_adv_examples - imgs, min=-self.eps * 2, max=self.eps * 2)
                fast_adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()

                # best_outputs = self.target_model(best_fast_adv_examples)
                # best_loss = F.cross_entropy(best_outputs, labels)
                if best_loss < loss:
                    best_loss = loss
                    best_fast_adv_examples = fast_adv_examples
                # else:
                #     fast_adv_examples = best_fast_adv_examples

            # fast_adv_examples = best_fast_adv_examples

            # for j in range()
            d_slow = best_fast_adv_examples - slow_adv_examples
            slow_adv_examples = slow_adv_examples + self.alpha * (d_slow)

            slow_perturbation = torch.clamp(slow_adv_examples - imgs, min=-self.eps*2, max=self.eps*2)
            slow_adv_examples = torch.clamp(imgs + slow_perturbation, min=0, max=1).detach()

        return slow_adv_examples