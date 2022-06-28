# -*- coding: utf-8 -*-
# @Time     : 2020/11/21 16:36
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
HGN
################################################

Reference:
    Chen Ma et al. "Hierarchical Gating Networks for Sequential Recommendation."in SIGKDD 2019


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

from recbole.model.sequential_recommender.fpmc import FPMC
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BiCL4FPMC(FPMC):
    r"""
    HGN sets feature gating and instance gating to get the important feature and item for predicting the next item

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BiCL4FPMC, self).__init__(config, dataset)
        self.cl_loss_debiased_type = config['cl_loss_debiased_type']
        self.perturbation = config['perturbation']
        self.noise_eps = config['noise_eps']
        self.cl_type = config['cl_type']

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']

        # define layers and loss
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
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
    
    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask
    
    def perturb(self, emb):
        noise = torch.rand(emb.shape, device=emb.device)
        noise = F.normalize(noise) * self.noise_eps
        emb = emb + torch.mul(torch.sign(emb), noise)
        return emb
    
    def forward(self, user, item_seq, item_seq_len, next_item, return_seq_emb=False):
        item_last_click_index = item_seq_len - 1
        item_last_click = torch.gather(item_seq, dim=1, index=item_last_click_index.unsqueeze(1))
        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]

        user_emb = self.UI_emb(user)
        user_emb = torch.unsqueeze(user_emb, dim=1)  # [b,1,emb]

        iu_emb = self.IU_emb(next_item)
        iu_emb = torch.unsqueeze(iu_emb, dim=1)  # [b,n,emb] in here n = 1

        il_emb = self.IL_emb(next_item)
        il_emb = torch.unsqueeze(il_emb, dim=1)  # [b,n,emb] in here n = 1

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF
        mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))
        mf = torch.squeeze(mf, dim=1)  # [B,1]
        #  FMC
        fmc = torch.matmul(il_emb, item_seq_emb.permute(0, 2, 1))
        fmc = torch.squeeze(fmc, dim=1)  # [B,1]

        score = mf + fmc
        score = torch.squeeze(score)
        if return_seq_emb:
            return score, item_seq_emb.squeeze()
        return score

    def calculate_loss(self, interaction):
        seq_item = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        user_embedding = self.UI_emb(user)
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_score, seq_output = self.forward(user, seq_item, item_seq_len, pos_items, return_seq_emb=True)
        neg_score = self.forward(user, seq_item, item_seq_len, neg_items)
        loss = super().calculate_loss(interaction)

        losses = [loss]
        if self.cl_type in ['un', 'rs', 'su', 'all', 'rs_su']:
            un_aug_seq_output = self.forward(user, seq_item, item_seq_len, pos_items, return_seq_emb=True)[1]

        if self.cl_type in ['su', 'rs_su_x', 'rs_su', 'all']:
            aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
            su_aug_seq_output = self.forward(user, aug_item_seq, aug_item_seq_len, pos_items, return_seq_emb=True)[1]
        
        if self.cl_type in ['rs', 'rs_su_x', 'rs_su', 'all']:
            aug_item_seq_rev, aug_item_seq_len_rev = interaction['aug_rev'], interaction['aug_len_rev']
            su_aug_seq_rev_output = self.forward(user, aug_item_seq_rev, aug_item_seq_len_rev, pos_items, return_seq_emb=True)[1]
        
        if self.perturbation:
            seq_output = self.perturb(seq_output)
            if self.cl_type in ['un', 'rs', 'su', 'rs_su', 'all']:
                un_aug_seq_output = self.perturb(un_aug_seq_output)
                
            if self.cl_type in ['su', 'rs_su', 'rs_su_x', 'all']:
                su_aug_seq_output = self.perturb(su_aug_seq_output)

            if self.cl_type in ['rs', 'rs_su', 'rs_su_x', 'all']:
                su_aug_seq_rev_output = self.perturb(su_aug_seq_rev_output)
            
            if not self.cl_type:
                un_aug_seq_output1 = self.perturb(seq_output)
                un_aug_seq_output2 = self.perturb(seq_output)

        cl_losses = []
        if self.cl_loss_debiased_type in ['mean', 'norm']:
            target = pos_items
        else:  # mean 
            target = None

        if self.cl_type in ['su', 'rs_su', 'all']: # duorec
            cl_loss = self.info_nce(un_aug_seq_output, su_aug_seq_output, target) 
            cl_losses.append(cl_loss)

        if self.cl_type in ['rs', 'rs_su']: # reverse seq x original seq
            cl_loss = self.info_nce(un_aug_seq_output, su_aug_seq_rev_output, target)
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