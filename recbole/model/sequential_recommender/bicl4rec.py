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

from recbole.model.sequential_recommender.duorec import DuoRec


class BiCL4Rec(DuoRec):

    def __init__(self, config, dataset):
        super(BiCL4Rec, self).__init__(config, dataset)

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

        # encode
        if self.cl_type in ['fs_drop_x']:
            drop_aug_seq_output = self.forward(item_seq, item_seq_len)

        if self.cl_type in ['fs', 'rs+fs', 'fs_drop_x']:
            fs_aug_seq_output = self.forward(interaction['aug'],
                                             interaction['aug_len'])
        
        if self.cl_type in ['rs', 'rs+fs']:
            rs_aug_seq_output = self.forward(interaction['aug_rev'],
                                             interaction['aug_len_rev'])

        cl_losses = []
        # duorec = forward supervised (fs) aug x encode original seq again
        if self.cl_type in ['fs_drop_x']:
            cl_loss = self.info_nce(fs_aug_seq_output,
                                    drop_aug_seq_output,
                                    pos_items)
            cl_losses.append(cl_loss)

        # reverse supervised (rs) aug x original seq
        if self.cl_type in ['rs', 'rs+fs']:
            cl_loss = self.info_nce(rs_aug_seq_output,
                                    seq_output,
                                    pos_items)
            cl_losses.append(cl_loss)

        # forward supervised (fs) aug x original seq
        if self.cl_type in ['fs', 'rs+fs']:
            cl_loss = self.info_nce(fs_aug_seq_output,
                                    seq_output,
                                    pos_items)
            cl_losses.append(cl_loss)

        cl_lambda = self.cl_lambda / len(cl_losses)
        cl_losses = [loss * cl_lambda for loss in cl_losses]
        return tuple(losses + cl_losses)