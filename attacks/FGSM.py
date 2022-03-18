#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust
@File    : FGSM.py
@Author  : igeng
@Date    : 2022/3/18 11:05
@Descrip :
'''
import torch
import torch.nn as nn
from attacks import Attack
class FGSM(Attack):
    """
    Fast Gradient Sign Method (FGSM)
    Paper link:

    """