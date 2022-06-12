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

from recbole.model.sequential_recommender.duorec import DuoRec


class BiCL4Rec(DuoRec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(BiCL4Rec, self).__init__(config, dataset)
        self.cl_loss_debiased_type = config['cl_loss_debiased_type']
        self.perturbation = config['perturbation']
        self.noise_eps = config['noise_eps']

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
            same_c_mask = target.repeat(cur_batch_size, 1) == target.reshape(-1, 1)
            same_c_mask = same_c_mask.repeat(2, 2)
            sim[same_c_mask] = -1.e8
            # normalize negative scores (according to real num of negative)
            # in other words, each positive sample would be paired with the same quantity of negative scores. => p/(p+2(B-1)*mean(ns)))
            if self.cl_loss_debiased_type == 'norm':
                negative_samples_weight = torch.log(1. - torch.mean(same_c_mask[mask].reshape(N, -1).float(), dim=1))
                negative_samples_weight = negative_samples_weight.reshape(-1, 1)
                sim = sim - negative_samples_weight

        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        if self.cl_loss_type == 'dcl': # decoupled contrastive learning
            loss = self.calculate_decoupled_cl_loss(logits, labels)
        else: # original infonce
            loss = self.cl_loss_fct(logits, labels)
        return loss
    
    def perturb(self, emb):
        noise = torch.rand(emb.shape, device=emb.device)
        noise = F.normalize(noise) * self.noise_eps
        emb = emb + torch.mul(torch.sign(emb), noise)
        return emb

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
        if self.cl_type in ['un', 'all']:
            un_aug_seq_output = self.forward(item_seq, item_seq_len)

        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)
        
        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_item_seq_rev, aug_item_seq_len_rev = interaction['aug_rev'], interaction['aug_len_rev']
            su_aug_seq_rev_output = self.forward(aug_item_seq_rev, aug_item_seq_len_rev)

        if self.perturbation:
            seq_output = self.perturb(seq_output)
            if self.cl_type in ['un', 'all']:
                un_aug_seq_output = self.perturb(un_aug_seq_output)
                
            if self.cl_type in ['su', 'rs_su_x', 'all']:
                su_aug_seq_output = self.perturb(su_aug_seq_output)

            if self.cl_type in ['rs', 'rs_su_x', 'all']:
                su_aug_seq_rev_output = self.perturb(su_aug_seq_rev_output)
            
            if not self.cl_type:
                un_aug_seq_output1 = self.perturb(seq_output)
                un_aug_seq_output2 = self.perturb(seq_output)

        cl_losses = []
        if self.cl_loss_debiased_type in ['mean', 'norm']:
            target = pos_items
        else:  # mean 
            target = None

        if self.cl_type in ['su', 'all']: # duorec
            cl_loss = self.info_nce(un_aug_seq_output, su_aug_seq_output, target) 
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs']: # reverse seq x original seq
            cl_loss = self.info_nce(seq_output, su_aug_seq_rev_output, target)
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs_su_x', 'all']: # reverse seq x forward seq
            cl_loss = self.info_nce(su_aug_seq_rev_output, su_aug_seq_output, target)
            cl_losses.append(cl_loss)
        
        if not self.cl_type:
            if self.perturbation:
                cl_loss = self.info_nce(un_aug_seq_output1, un_aug_seq_output2, target)
            else:
                cl_loss = self.info_nce(seq_output, seq_output, target)
            cl_losses.append(cl_loss)

        cl_losses = [loss * self.cl_lambda / len(cl_losses) for loss in cl_losses]

        return tuple(losses + cl_losses) 