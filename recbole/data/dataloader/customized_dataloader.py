# @Time   : 2021/12/11
# @Author : Ray Wu
# @Email  : ray7102ray7102@gmail.com

"""
recbole.data.dataloader.customized_dataloader
################################################
"""

import numpy as np
import torch
import random
from time import time

from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType


class CLTrainDataLoader(TrainDataLoader):
    """:class:`CLTrainDataLoader` is a dataloader for Contrastive Learning training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        if config['model'] == 'CL4Rec':
            self.augmentation_table = {
                'crop': 0,
                'mask': 1,
                'reorder': 2,
                'random': 3
            }
            self.aug_type1 = config['aug_type1']
            self.aug_type2 = config['aug_type2'] if config['aug_type2'] else config['aug_type1']
            self.eta = config['eta']  # for crop
            self.gamma = config['gamma']  # for mask
            self.beta = config['beta']  # for reorder
    
    def _next_batch_data(self):
        cur_data = self._contrastive_learning_augmentation(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data
    
    def _contrastive_learning_augmentation(self, cur_data):
        if self.config['model'] == 'CL4Rec':
            aug_seq1, aug_len1 = self.cl4rec_augmentation(cur_data, self.aug_type1)
            aug_seq2, aug_len2 = self.cl4rec_augmentation(cur_data, self.aug_type2)
            cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                         'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data
    
    def cl4rec_augmentation(self, cur_data, aug_type='random'):
        aug_idx = self.augmentation_table[aug_type]
        sequences = cur_data['item_id_list']
        lengths = cur_data['item_length']
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)

        def crop(seq, length):
            new_seq = torch.zeros_like(seq)
            new_seq_length = max(1, int(length * self.eta))
            crop_start = random.randint(0, length - new_seq_length)
            new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
            return new_seq, new_seq_length
        
        def mask(seq, length):
            num_mask = int(length * self.gamma)
            mask_index = random.sample(range(length), k=num_mask)
            seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length):
            num_reorder = int(length * self.beta)
            reorder_start = random.randint(0, length - num_reorder)
            shuffle_index = torch.randperm(num_reorder) + reorder_start
            seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
            return seq, length

        aug_func = [crop, mask, reorder]
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            if aug_type == 'random':
                aug_idx = random.randrange(3)
            
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length)

        return aug_sequences, aug_lengths
