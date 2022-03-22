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

from recbole.model.sequential_recommender.myrec4 import MyRec4


class MyRec6(MyRec4):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec6, self).__init__(config, dataset)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        losses = [loss]
        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)
        
        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_item_seq_rev, aug_item_seq_len_rev = interaction['aug_rev'], interaction['aug_len_rev']
            su_aug_seq_rev_output = self.forward(aug_item_seq_rev, aug_item_seq_len_rev)

        cl_losses = []
        if self.cl_type in ['su', 'all']:
            cl_loss = self.info_nce(seq_output, su_aug_seq_output, pos_items)
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs', 'all']:
            cl_loss = self.info_nce(seq_output, su_aug_seq_rev_output, pos_items)
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs_su_x', 'all']:
            cl_loss = self.info_nce(su_aug_seq_rev_output, su_aug_seq_output, pos_items)
            cl_losses.append(cl_loss)
        
        cl_losses = [loss * self.cl_lambda / len(cl_losses) for loss in cl_losses]
        return tuple(losses + cl_losses)