#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust
@File    : attacks.py
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
    def __init__(self, attack_type, target_model, img_type='float'):
        return

    def perturb(self):
        return