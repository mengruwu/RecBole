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
import copy
import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.myrec7 import MyRec7


class BiSASRec(MyRec7):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(BiSASRec, self).__init__(config, dataset)
        self.reverse_trm_encoder = copy.deepcopy(self.trm_encoder)
    
    def forward(self, item_seq, item_seq_len, reverse=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        
        if reverse:
            trm_output = self.reverse_trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        else:
            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

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
        
        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)
        
        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_item_seq_rev, aug_item_seq_len_rev = interaction['aug_rev'], interaction['aug_len_rev']
            su_aug_seq_rev_output = self.forward(aug_item_seq_rev, aug_item_seq_len_rev, reverse=True)
        
        logits = torch.matmul(su_aug_seq_rev_output, test_item_emb.transpose(0, 1))
        rs_loss = self.loss_fct(logits, pos_items)
        if self.cl_su_lambda > 0:
            losses.append(self.cl_su_lambda * rs_loss)

        cl_losses = []
        if self.cl_loss_debiased_type in ['mean', 'norm']:
            target = pos_items
        else:  # mean 
            target = None

        if self.cl_type in ['su', 'all']:
            cl_loss = self.info_nce(seq_output, su_aug_seq_output, target)
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs', 'all']:
            cl_loss = self.info_nce(seq_output, su_aug_seq_rev_output, target)
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs_su_x', 'all']:
            cl_loss = self.info_nce(su_aug_seq_rev_output, su_aug_seq_output, target)
            cl_losses.append(cl_loss)
        
        if self.cl_loss_weight == 'adaptive' and self.cl_type == 'all':
            su_score = -loss
            rs_score = -rs_loss
            scores = torch.tensor([su_score, rs_score, (su_score + rs_score) / 2], device=loss.device)
            cl_loss_weight = F.softmax(scores, dim=-1)
            cl_losses = [loss * self.cl_lambda * w for loss, w in zip(cl_losses, cl_loss_weight)]
            wandb.log({
                'su_loss': loss.item(),
                'rev_su_loss': rs_loss.item(),
                'su_weight': cl_loss_weight[0],
                'rev_su_weight': cl_loss_weight[1],
            }, commit=False)
        else:  # mean
            cl_losses = [loss * self.cl_lambda / len(cl_losses) for loss in cl_losses]

        return tuple(losses + cl_losses) 