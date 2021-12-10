# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec


class DuoRec(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(DuoRec, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.similarity_type = config['similarity_type']
        self.tau = config['tau']
        self.cl_loss_type = config['cl_loss_type']
        self.cl_lambda = config['cl_lambda']
        self.cl_type = config['cl_type']

        # define layers and loss
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        elif self.cl_loss_type == 'dcl':
            self.cl_loss_fct = self.calculate_decoupled_cl_loss
        
        # parameters initialization
        self.apply(self._init_weights)

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

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
        if self.cl_type in ['us', 'un', 'us_x']:
            un_aug_seq_output = self.forward(item_seq, item_seq_len)
        
        if self.cl_type in ['us', 'su', 'us_x']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)

        if self.cl_type in ['us', 'un']:
            logits, labels = self.info_nce(seq_output, un_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        if self.cl_type in ['us', 'su']:
            logits, labels = self.info_nce(seq_output, su_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        if self.cl_type == 'us_x':
            logits, labels = self.info_nce(un_aug_seq_output, su_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        return tuple(losses)
    
    def calculate_decoupled_cl_loss(self, input, target):
        input_pos = torch.gather(input, 1, target.unsqueeze(-1)).squeeze(-1)
        input_exp = torch.exp(input)
        input_pos_exp = torch.exp(input_pos)
        input_neg_exp_sum = torch.sum(input_exp, dim=1) - input_pos_exp
        dcl_loss = torch.mean(-input_pos + torch.log(input_neg_exp_sum))
        return dcl_loss

    def info_nce(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        cur_batch_size = z_i.size(0)
        N = 2 * cur_batch_size
        if cur_batch_size != self.batch_size:
            mask = self.mask_correlated_samples(cur_batch_size).to(z_i.device)
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
        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        return logits, labels
