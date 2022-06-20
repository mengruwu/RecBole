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


class CL4Rec(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(CL4Rec, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']

        self.perturbation = config['perturbation']
        self.noise_eps = config['noise_eps']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # for mask
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        
        # parameters initialization
        self.apply(self._init_weights)

    def perturb(self, emb):
        noise = torch.rand(emb.shape, device=emb.device)
        noise = F.normalize(noise) * self.noise_eps
        emb = emb + torch.mul(torch.sign(emb), noise)
        return emb

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

    def calculate_loss(self, interaction):
        rs_loss, seq_output = self.calculate_rs_loss(interaction, return_output=True)
        cl_loss = self.calculate_cl_loss(interaction, seq_output=seq_output)
        return rs_loss, self.cl_lambda * cl_loss
    
    def calculate_rs_loss(self, interaction, return_output=False):
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

        if return_output:
            return loss, seq_output
        else:
            return loss

    def calculate_cl_loss(self, interaction, seq_output=None):
        if 'aug1' in interaction:
            aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
                interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
            seq_output1 = self.forward(aug_item_seq1, aug_len1)
            seq_output2 = self.forward(aug_item_seq2, aug_len2)

            if self.perturbation:
                seq_output1 = self.perturb(seq_output1)
                seq_output2 = self.perturb(seq_output2)
            cl_loss = self.info_nce(seq_output1, seq_output2)
        else:
            cl_loss = self.info_nce(seq_output, seq_output)
        return cl_loss
    
    def calculate_decoupled_cl_loss(self, input, target):
        input_pos = torch.gather(input, 1, target.unsqueeze(-1)).squeeze(-1)
        input_exp = torch.exp(input)
        input_pos_exp = torch.exp(input_pos)
        input_neg_exp_sum = torch.sum(input_exp, dim=1) - input_pos_exp
        dcl_loss = torch.mean(-input_pos + torch.log(input_neg_exp_sum))
        return dcl_loss

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
            sim[same_c_mask] = -1.e4

        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        if self.cl_loss_type == 'dcl': # decoupled contrastive learning
            loss = self.calculate_decoupled_cl_loss(logits, labels)
        else: # original infonce
            loss = self.cl_loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
