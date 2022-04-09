#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : CLEAN.py
@Author  : igeng
@Date    : 2022/3/29 20:09 
@Descrip :
'''
from .Attacks import Attack

class CLEAN(Attack):
    """
    Clean attack to test the benign accuracy of target_model
    """
    def __init__(self, target_model, args):
        super(CLEAN, self).__init__("CLEAN", target_model)
        self.device = args.device

    def perturb(self, imgs, labels):
        """
        Overridden
        :param args:
        :return:
        """
        adv_examples = imgs.clone().detach().to(self.device)

        return adv_examples