#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust
@File    : Attacks.py
@Author  : igeng
@Date    : 2022/3/18 10:36
@Descrip :
'''
import torch

class Attack(object):
    """
    Abstract base class for all attack methods.
    :param
    
    """
    def __init__(self, attack, target_model):
        self.attack = attack
        self.target_model = target_model

    def perturb(self, *args):
        """
        Generate adversarial perturbations,
        this function should be overridden by each attack method.
        :param args:
        :return:
        """
        return NotImplementedError