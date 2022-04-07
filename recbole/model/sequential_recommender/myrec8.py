# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
DuoRec
################################################

Reference:
    Ruihong Qiu et al. "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation" in WSDM 2022.

Reference:
    https://github.com/RuihongQiu/DuoRec

"""

import wandb
import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.myrec7 import MyRec7


class MyRec8(MyRec7):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec8, self).__init__(config, dataset)
        