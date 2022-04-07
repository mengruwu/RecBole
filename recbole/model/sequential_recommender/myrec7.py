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

from recbole.model.sequential_recommender.myrec4 import MyRec4


class MyRec7(MyRec4):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec7, self).__init__(config, dataset)
        self.cl_loss_weight = config['cl_loss_weight']
        self.cl_su_lambda = config['cl_su_lambda']
        self.cl_loss_debiased_type = config['cl_loss_debiased_type']

    def info_nce(self, z_i, z_j, target=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        cur_batch_size = z_i.size(0)
        N = 2 * cur_batch_size
        if cur_batch_size != self.batch_size:
            mask = self.mask_correlated_samples(cur_batch_size)
        else:
            mask = self.default_mask
        z = torch.cat((z_i, z_j), dim=0)  # [2B H]
    
        if self.similarity_type == 'cos':
            sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z, z.T) / self.tau

        sim_i_j = torch.diag(sim, cur_batch_size)
        sim_j_i = torch.diag(sim, -cur_batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2B, 1]
        if target != None:
            same_c_mask = torch.tile(target, (cur_batch_size, 1)) == target.reshape(-1, 1)
            same_c_mask = torch.tile(same_c_mask, (2, 2))
            sim[same_c_mask] = -1.e8
            if self.cl_loss_debiased_type == 'norm':
                negative_samples_weight = torch.log(1. - torch.mean(same_c_mask[mask].reshape(N, -1).float(), dim=1))
                negative_samples_weight = negative_samples_weight.reshape(-1, 1)
                sim = sim - negative_samples_weight

        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]
        # normalize
        

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        if self.cl_loss_type == 'dcl': # decoupled contrastive learning
            loss = self.calculate_decoupled_cl_loss(logits, labels)
        else: # original infonce
            loss = self.cl_loss_fct(logits, labels)
        return loss

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
            su_aug_seq_rev_output = self.forward(aug_item_seq_rev, aug_item_seq_len_rev)
        
        if self.cl_su_lambda > 0 and self.cl_type in ['rs', 'rs_su_x', 'all']:
            logits = torch.matmul(su_aug_seq_rev_output, test_item_emb.transpose(0, 1))
            rs_loss = self.loss_fct(logits, pos_items)
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