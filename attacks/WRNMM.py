#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : AdvRobust 
@File    : WRNMM.py
@Author  : igeng
@Date    : 2022/3/26 18:50 
@Descrip :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attacks import Attack

class WRNMM(Attack):
    """
    Paper link:
    :argument:
    :argument:
    """