#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : BIM.py
@Author  : igeng
@Date    : 2022/3/21 14:23 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class BIM(Attack):
    """
    Basic Iterative Method or Iterative Fast Gradient Sign Method (I-FGSM)
    Paper link:
    :argument: target_model {nn.Module} -- Target model to be attacked.
    :argument: eps {float} -- Magnitude of perturbation.
    """
    def __init__(self, target_model, args):
        super(BIM, self).__init__("BIM", target_model)
        self.eps = args.bim_epsilon
        self.eps_iter = args.bim_eps_iter
        self.n_iters = args.bim_n_iters
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

        for step in range(self.n_iters):
            # print("PGD attack {} step!".format(step))
            adv_examples.requires_grad = True
            outputs = self.target_model(adv_examples)
            # Use torch.nn loss
            # criterion = nn.CrossEntropyLoss
            # loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_examples)[0]
            # consider eps is a vector ?
            adv_examples = adv_examples.detach() + self.eps_iter * gradients.sign()

            adv_lower = torch.clamp(imgs - self.eps, min=0)
            adv_upper = imgs + self.eps
            adv_examples = (adv_examples >= adv_lower).float() * adv_examples + (adv_examples < adv_lower).float() * adv_lower
            adv_examples = (adv_examples > adv_upper).float() * adv_upper + (adv_examples <= adv_upper).float() * adv_examples
            adv_examples = torch.clamp(adv_examples, max=1).detach()

            # perturbation = torch.clamp(adv_examples - imgs, min=-self.eps, max=self.eps)
            # adv_examples = torch.clamp(adv_examples + perturbation, min=0, max=1).detach()

        return adv_examples