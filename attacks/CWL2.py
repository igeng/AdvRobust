#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : CWL2.py
@Author  : igeng
@Date    : 2022/3/23 20:03 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class CWL2(Attack):
    """
    CWL2
    Paper link:
    :argument:
    :argument:
    """
    def __init__(self, target_model, args):
        super(CWL2, self).__init__("CWL2", target_model)
        self.targeted = args.attack_targeted
        self.c = args.cw_c
        self.k = args.cw_k
        self.n_iters = args.cw_n_iters
        self.lr = args.cw_lr
        self.binary_search_steps = args.cw_binary_search_steps
        self.device = args.device

    def perturb(self, imgs, labels):
        """
        Overriden
        :param imgs:
        :param labels:
        :return:
        """
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        w = self.inverse_tanh(imgs).detach()
        w.requires_grad = True

        best_adv_imgs = imgs.clone().detach()
        best_L2 = 1e10 * torch.ones((len(imgs))).to(self.device)
        prev_loss = 1e10
        dim = len(imgs.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.n_iters):
            adv_imgs = self.tanh(w)

            current_L2 = MSELoss(Flatten(adv_imgs),
                                 Flatten(imgs)).sum(dim=1) # torch.Size([100, 3072]).sum(dim=1) ==> torch.Size([100])
            L2_loss = current_L2.sum()

            outputs = self.target_model(adv_imgs)
            f_loss = self.f(outputs, labels).sum()
            loss = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()
            # mask : both misclassified and current_L2 loss decreases. current_L2 is smaller to be best_L2
            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1)) # [-1, 1, 1, 1]
            best_adv_imgs = mask * adv_imgs.detach() + (1 - mask) * best_adv_imgs # 保留adv_imgs中被错分且current_L2更小的图片[100,1,1,1]中最小维度里的1或0对应[100,3,32,32]中一个[:,3,32,32]

            if step % (self.n_iters // 10) == 0:
                if loss > prev_loss:
                    return best_adv_imgs
                prev_loss = loss

        return best_adv_imgs

    def tanh(self, x):
        return 0.5 * (torch.tanh(x) + 1)

    def inverse_tanh(self, x):
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def f(self, outputs, labels):
        labels_onehot = torch.eye(len(outputs[0]))[labels].to(self.device)

        other, _ = torch.max((1 - labels_onehot) * outputs, dim=1)
        truth = torch.masked_select(outputs, labels_onehot.bool())

        return torch.clamp(truth - other, min=-self.k)