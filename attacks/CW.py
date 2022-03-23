#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : CW.py
@Author  : igeng
@Date    : 2022/3/21 22:20 
@Descrip :
'''
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class CW(Attack):
    """
    CW
    Paper link:
    :argument:
    :argument:
    """
    def __init__(self, target_model, args):
        super(CW, self).__init__("CW", target_model)
        self.targeted = args.attack_targeted
        self.c = args.cw_c
        self.k = args.cw_k
        self.n_iters = args.cw_n_iters
        self.lr = args.cw_lr
        self.binary_search_steps = args.cw_binary_search_steps
        self.device = args.device

    def _arctanh(self, imgs):
        scaling = torch.clamp(imgs, min=-1, max=1)
        x = 0.99999999 * scaling
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _scaler(self, imgs_atanh):
        return 0.5 * (torch.tanh(imgs_atanh) + 1)

    def _f(self, adv, labels):
        outputs = self.target_model(adv)
        # 表示真实的标签的one-hot编码的torch.tensor
        # labels_onehot = F.one_hot(labels, 10)
        labels_onehot = torch.eye(len(outputs[0]))[labels].to(self.device)

        # 表示分类模型输出的概率向量中最大的分量概率值
        truth = (labels_onehot * outputs).sum(dim=1)
        # 表示分类模型输出的概率向量中第二大的分量的概率值
        other, _ = torch.max((1 - labels_onehot) * outputs, dim=1)

        if self.targeted:
            # 如果有目标，则labels是要攻击成为得目标，即正常label为1，攻击成label为2
            # 则上文中得labels设置为2，也可以设置为离正常label最远的
            # -k中k增大是为了增加 other-truth 的置信区间，因为优化目标是最小化other-truth到-k以下
            loss = torch.clamp(other - truth, min=-self.k)
        else:
            loss = torch.clamp(truth - other, min=-self.k)

        return loss


    def perturb(self, imgs, labels):
        imgs = imgs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        imgs_arctanh = self._arctanh(imgs)

        for _ in range(self.binary_search_steps):
            perturbation = torch.zeros_like(imgs).to(self.device)
            perturbation.detach_()
            perturbation.requires_grad = True
            optimizer = torch.optim.Adam([perturbation], lr=self.lr)
            prev_loss = 1e6

            for step in range(self.n_iters):
                optimizer.zero_grad()
                adv_examples = self._scaler(imgs_arctanh + perturbation)
                loss1 = torch.sum(self.c * self._f(adv_examples, labels))
                loss2 = F.mse_loss(adv_examples, imgs, reduction='sum')

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                if step % (self.n_iters // 10) == 0:
                    if loss > prev_loss:
                        # warnings.warn("Early stopped because the loss didn't converged.")
                        # return self.scaler(imgs_arctanh + perturbation).detach()
                        break
                    prev_loss = loss

            adv_imgs = self._scaler(imgs_arctanh + perturbation).detach()
            return adv_imgs


















