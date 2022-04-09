#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : MTPGD.py
@Author  : igeng
@Date    : 2022/3/30 14:30 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class MTPGD(Attack):
    """
    MultiTargeted Projected Gradient Descent (MT-PGD)
    Distance Measure : Linf
    Paper link: https://arxiv.org/abs/1910.09338v1
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """

    def __init__(self, target_model, args):
        super(MTPGD, self).__init__("MTPGD", target_model)
        self.eps = args.pgd_epsilon
        self.eps_step = args.pgd_eps_step
        self.n_steps = args.pgd_n_steps
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

        best_loss = 0
        best_adv_examples = imgs.clone().detach()

        adv_examples = imgs.clone().detach()

        outputs_imgs = self.target_model(imgs)
        indices = torch.topk(outputs_imgs, 3).indices
        labels_top2 = indices.gather(1, torch.ones_like(labels.view(-1, 1), dtype=torch.int64))
        labels_top3 = indices.gather(1, 2 * torch.ones_like(labels.view(-1, 1), dtype=torch.int64))
        target_labels = [labels_top2.view(-1), labels_top3.view(-1)]

        if self.random_start:
            adv_examples = adv_examples + torch.empty_like(adv_examples).uniform_(-self.eps, self.eps)
            adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

        for step in range(self.n_steps):
            for j in range(2):
                adv_examples = best_adv_examples
                adv_examples.requires_grad = True
                outputs = self.target_model(adv_examples)

                # loss = F.cross_entropy(outputs, labels) - F.cross_entropy(outputs, target_labels[j])
                loss = F.cross_entropy(outputs, labels)

                gradients = torch.autograd.grad(loss, adv_examples)[0]

                adv_examples = adv_examples.detach() + self.eps_step * gradients.sign()
                perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
                adv_examples = torch.clamp(imgs + perturbation, min=0, max=1).detach()\

                if loss > best_loss:
                    best_adv_examples = adv_examples
                    best_loss = loss
                    # print("best loss is: ")
                    # print(best_loss)

        return best_adv_examples