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

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class MyRec5(CL4Rec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec5, self).__init__(config, dataset)
        self.cl_type = config['cl_type']

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        losses = [loss]
        if self.cl_type in ['us', 'un', 'us_x']:
            un_aug_seq_output = self.forward(item_seq, item_seq_len)
        
        if self.cl_type in ['us', 'su', 'us_x']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)
        
        cl_lambda = self.cl_lambda
        if self.cl_type in ['us', 'un']:
            cl_loss = self.info_nce(seq_output, un_aug_seq_output, target=pos_items)
            cl_loss = cl_lambda * cl_loss
            losses.append(cl_loss)

        if self.cl_type in ['us', 'su']:
            cl_loss = self.info_nce(seq_output, su_aug_seq_output, target=pos_items)
            cl_loss = cl_lambda * cl_loss
            losses.append(cl_loss)

        if self.cl_type == 'us_x':
            cl_loss = self.info_nce(un_aug_seq_output, su_aug_seq_output, target=pos_items)
            cl_loss = cl_lambda * cl_loss
            losses.append(cl_loss)

        return tuple(losses)
