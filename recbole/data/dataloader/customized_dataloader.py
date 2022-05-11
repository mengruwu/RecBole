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

from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from scipy.special import softmax
from itertools import combinations

from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction


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

        self.iid_field = dataset.iid_field
        self.uid_field = dataset.uid_field
        self.iid_list_field = getattr(dataset, f'{self.iid_field}_list_field')
        self.item_list_length_field = dataset.item_list_length_field

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        cur_data = self.augmentation(cur_data)
        self.pr += self.step
        return cur_data
    
    def _shuffle(self):
        super()._shuffle()
        self.augmentation()

    def augmentation(self):
        raise NotImplementedError('Method augmentation should be implemented.')

class CL4RecTrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        augmentation_type = ['crop', 'mask', 'reorder', 'random']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        
        self.aug_type1 = config['aug_type1']
        self.aug_type2 = config['aug_type2'] if config['aug_type2'] else config['aug_type1']
        self.crop_rate = config['crop_rate']  # for crop
        self.mask_rate = config['mask_rate']  # for mask
        self.reorder_rate = config['reorder_rate']  # for reorder
    
    def _crop(self, seq, length):
        new_seq = np.zeros_like(seq)
        new_seq_length = max(1, int(length * (1 - self.crop_rate)))
        crop_start = random.randint(0, length - new_seq_length)
        new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
        return new_seq, new_seq_length
    
    def _mask(self, seq, length):
        num_mask = int(length * self.mask_rate)
        mask_index = random.sample(range(length), k=num_mask)
        seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
        return seq, length
    
    def _reorder(self, seq, length):
        num_reorder = int(length * self.reorder_rate)
        reorder_start = random.randint(0, length - num_reorder)
        shuffle_index = np.random.permutation(num_reorder) + reorder_start
        seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
        return seq, length
    
    def augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field].numpy()
        lengths = cur_data[self.item_list_length_field].numpy()

        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, aug_type=self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, aug_type=self.aug_type2)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data
    
    def _shuffle(self):
        self.dataset.shuffle()

    def _augmentation(self, sequences, lengths, targets=None, aug_type='random'):
        aug_sequences = np.zeros_like(sequences)
        aug_lengths = np.zeros_like(lengths)
        num = aug_sequences.shape[0]
        
        aug_func = [self._crop, self._mask, self._reorder]
        if aug_type == 'random':  # [crop, mask, reorder]
            aug_idxs = random.choices(range(len(aug_func)), k=num)
        else:
            aug_idx = self.augmentation_table[aug_type]
            aug_idxs = [aug_idx] * num
        
        for i, (aug_idx, seq, length) in enumerate(zip(aug_idxs, sequences, lengths)):
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.copy(), length)
        
        return aug_sequences, aug_lengths

class CoSeRecTrainDataLoader(CL4RecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.item_num = self.dataset.item_num
        self.max_item_list_len = self.dataset.max_item_list_len
        
        self.reorder_rate = config['reorder_rate']
        self.insert_rate = config['insert_rate']
        self.substitute_rate = config['substitute_rate']

        augmentation_type = ['insert', 'substitute', 'crop', 'mask', 'reorder']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        self.online_item_cor_mat = None
        self.offline_item_cor_mat = None

        self.offline_similarity_type = config['cl_aug_offline_similarity_type']
        self.topk = config['cl_aug_similarity_topk']

        self._init_augmentation()

    def _init_augmentation(self):
        uids = self.dataset.inter_feat[self.uid_field].numpy()
        iids = self.dataset.inter_feat[self.iid_field].numpy()
        item_list = self.dataset[self.iid_list_field].numpy()
        item_length = self.dataset[self.item_list_length_field].numpy()
        last_item_index = item_length - 1
        prev_iids = item_list[np.arange(len(self.dataset)), last_item_index]
        
        last_uid = None
        iids_inter_list = []
        for uid, iid, prev_iid in zip(uids, iids, prev_iids):
            if last_uid != uid:
                last_uid = uid
                iids_inter_list.append([prev_iid])
            iids_inter_list[-1].append(iid)

        shape = (self.item_num, self.item_num)
        item_inter_matrix = coo_matrix(shape)
        for iids_inter in iids_inter_list:
            row, col = zip(*combinations(iids_inter, 2))
            if self.offline_similarity_type == 'itemcf_iuf':
                data = np.ones_like(row) / np.log(1 + len(iids_inter))
            else:  # itemcf
                data = np.ones_like(row)
            item_inter_matrix += coo_matrix((data, (row, col)), shape=shape)
        item_inter_matrix += item_inter_matrix.T
        # norm_scalar = np.sqrt(np.outer(item_inter_matrix.getnnz(axis=1),
        #                                item_inter_matrix.getnnz(axis=0)))
        # item_inter_matrix2 = item_inter_matrix.multiply(np.reciprocal(norm_scalar))
        norm_scalar1 = np.reciprocal(np.sqrt(item_inter_matrix.getnnz(axis=1)) + 1e-8).reshape(-1, 1)
        norm_scalar2 = np.reciprocal(np.sqrt(item_inter_matrix.getnnz(axis=0)) + 1e-8)
        self.offline_item_cor_mat = item_inter_matrix.multiply(norm_scalar1).multiply(norm_scalar2)
        self._update_most_similar_items()

    def _update_most_similar_items(self):
        if self.online_item_cor_mat != None:
            cor_mat = self.offline_item_cor_mat.maximum(self.online_item_cor_mat)
        else:
            cor_mat = self.offline_item_cor_mat.copy()
        cor_mat = cor_mat.tolil()

        most_similar_items = np.zeros((self.item_num, self.topk), dtype=int)
        for i, (data, row) in enumerate(zip(cor_mat.data, cor_mat.rows)):
            data, row = np.array(data), np.array(row)
            topk_indices = np.argsort(-data)[:self.topk]
            most_similar_items[i, :len(data)] = row[topk_indices]  # using len(data) because sometimes # of data is less than topk
        self.most_similar_items = most_similar_items
        
    def update_embedding_matrix(self, item_embedding):
        cor_mat = np.matmul(item_embedding, item_embedding.T)
        _min, _max = np.min(cor_mat), np.max(cor_mat)
        cor_mat = (cor_mat - _min) / (_max - _min)
        idx = (-cor_mat).argpartition(self.topk, axis=1)[:, :self.topk]
        row = np.repeat(np.arange(self.item_num), self.topk)
        col = idx.flatten()
        data = np.take_along_axis(cor_mat, idx, axis=1).flatten()
        self.online_item_cor_mat = coo_matrix((data, (row, col)),
                                              shape=(self.item_num, self.item_num))
        self._update_most_similar_items()

    def _augmentation(self, sequences, lengths, targets=None, aug_type='random_all'):
        aug_sequences = np.zeros_like(sequences)
        aug_lengths = np.zeros_like(lengths)
        num = aug_sequences.shape[0]
        
        aug_func = [self._insert, self._substitute, \
                    self._crop, self._mask, self._reorder]
        if aug_type == 'random_all':  # [insert, substitute, crop, mask, reorder]
            aug_idxs = random.choices(range(len(aug_func)), k=num)
        elif aug_type == 'random':  # [insert, substitute]
            aug_idxs = random.choices(range(2), k=num)
        elif aug_type == 'random_cmr':  # CL4SRec [crop, mask, reorder]
            aug_idxs = random.choices(range(2, 5), k=num)
        else:
            aug_idx = self.augmentation_table[aug_type]
            aug_idxs = [aug_idx] * num
        
        for i, (aug_idx, seq, length) in enumerate(zip(aug_idxs, sequences, lengths)):
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.copy(), length)
        
        return aug_sequences, aug_lengths

    def _insert(self, seq, length):  # it has duplication in seq
        new_seq = np.zeros_like(seq)
        num_insert = max(int(length * self.insert_rate), 1)
        insert_index = random.sample(range(length), k=num_insert)
        ref_items = [seq[i] for i in insert_index]
        insert_items = self.most_similar_items[ref_items, 0]
        seq = np.expand_dims(seq[:length], axis=1).tolist()
        for i, item in zip(insert_index, insert_items):
            seq[i].insert(0, item)
        seq = np.concatenate(seq, axis=0)[-self.max_item_list_len:]
        new_seq_length = len(seq)
        new_seq[:new_seq_length] = seq
        return new_seq, new_seq_length
    
    def _substitute(self, seq, length):
        num_substitute = max(int(length * self.substitute_rate), 1)
        substitute_index = random.sample(range(length), k=num_substitute)
        ref_items = [seq[i] for i in substitute_index]
        substitute_items = self.most_similar_items[ref_items, 0]
        seq[substitute_index] = substitute_items
        return seq, length
    
class DuoRecTrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.rand_idx = np.random.uniform(size=config['train_batch_size'])

    def augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        aug_seq, aug_len = self._augmentation(targets=targets)
        cur_data.update(Interaction({'aug': aug_seq, 'aug_len': aug_len}))
        return cur_data

    def _init_augmentation(self):
        target_item_list = self.dataset.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index = index
        self.same_target_len = {k: len(v) for k, v in index.items()}

    def _shuffle(self):
        self.dataset.shuffle()
        self._init_augmentation()

    def _augmentation(self, sequences=None, lengths=None, targets=None):
        aug_idx = [int(_idx * self.same_target_len[target]) for target, _idx in zip(targets, self.rand_idx)]
        select_index = [self.same_target_index[target][idx] for target, idx in zip(targets, aug_idx)]
        aug_sequences = self.dataset[self.iid_list_field][select_index]
        aug_lengths = self.dataset[self.item_list_length_field][select_index]
        # assert (targets == self.dataset[self.iid_field][select_index].numpy()).all()
        return aug_sequences, aug_lengths


class BiCL4RecTrainDataLoader(DuoRecTrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self._init_reverse_augmentation()
        self.cl_type = config['cl_type']
        
    def _init_reverse_augmentation(self):
        self.dataset_reverse = self.dataset.copy(self.dataset.inter_feat)
        self.dataset_reverse.reversed = True
        self.dataset_reverse.data_augmentation()
        target_item_list = self.dataset_reverse.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index_reverse = index
        self.same_target_len_reverse = {k: len(v) for k, v in index.items()}
    
    def augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        sequences = cur_data[self.iid_list_field].numpy()
        lengths = cur_data[self.item_list_length_field].numpy()
        update = {}
        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_seq, aug_len = self._augmentation(targets=targets)
            update.update({'aug': aug_seq, 'aug_len': aug_len})

        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_seq_rev, aug_len_rev = self._augmentation_reverse(sequences=sequences, lengths=lengths, targets=targets)
            update.update({'aug_rev': aug_seq_rev, 'aug_len_rev': aug_len_rev})

        cur_data.update(Interaction(update))
        return cur_data

    def _augmentation(self, sequences=None, lengths=None, targets=None):
        return super()._augmentation(sequences, lengths, targets)
    
    def _augmentation_reverse(self, sequences=None, lengths=None, targets=None):
        aug_sequences = np.zeros_like(sequences)
        aug_lengths = np.zeros_like(lengths)
        for idx, target in enumerate(targets):
            if target in self.same_target_index_reverse:
                select_index = int(self.rand_idx[idx] * self.same_target_len_reverse[target])
                select_index = self.same_target_index_reverse[target][select_index]
                aug_sequences[idx] = self.dataset_reverse[self.iid_list_field][select_index]
                aug_lengths[idx] = self.dataset_reverse[self.item_list_length_field][select_index]
                # assert target == self.dataset_reverse[self.iid_field][select_index]
            else:
                select_index = int(self.rand_idx[idx] * self.same_target_len[target])
                select_index = self.same_target_index[target][select_index]
                aug_sequences[idx] = self.dataset[self.iid_list_field][select_index]
                aug_lengths[idx] = self.dataset[self.item_list_length_field][select_index]
                # assert target == self.dataset[self.iid_field][select_index]
        return aug_sequences, aug_lengths


class MyRec3TrainDataLoader(DuoRecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.ucl_type = config['ucl_type']
        ucl_dataloader_table = {
            'cl4rec': CL4RecTrainDataLoader,
            'coserec': CoSeRecTrainDataLoader,
        }
        ucl_aug_type_table = {
            'cl4rec': 'random',
            'coserec': 'random_all',
        }
        self.ucl_dataloader = ucl_dataloader_table[self.ucl_type](config, dataset, sampler, shuffle)
        self._ucl_aug_type = ucl_aug_type_table[self.ucl_type]
    
    def augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field].numpy()
        lengths = cur_data[self.item_list_length_field].numpy()
        targets = cur_data[self.iid_field].numpy()
        aug_seq1, aug_len1 = self._supervised_augmentation(targets)
        aug_seq1, aug_len1 = self._unsupervised_augmentation(aug_seq1.numpy(), aug_len1.numpy(), targets)
        aug_seq2, aug_len2 = self._unsupervised_augmentation(sequences, lengths, targets)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data

    def _supervised_augmentation(self, targets):
        return super()._augmentation(targets=targets)
    
    def _unsupervised_augmentation(self, sequences, lengths, targets=None):
        return self.ucl_dataloader._augmentation(sequences,
                                                 lengths,
                                                 targets=targets,
                                                 aug_type=self._ucl_aug_type)

class MyRec4TrainDataLoader(DuoRecTrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self._init_reverse_augmentation()
        self.cl_type = config['cl_type']
        
    def _init_reverse_augmentation(self):
        self.dataset_reverse = self.dataset.copy(self.dataset.inter_feat)
        self.dataset_reverse.reversed = True
        self.dataset_reverse.data_augmentation()
        target_item_list = self.dataset_reverse.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index_reverse = index
        self.same_target_len_reverse = {k: len(v) for k, v in index.items()}
    
    def augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        sequences = cur_data[self.iid_list_field].numpy()
        lengths = cur_data[self.item_list_length_field].numpy()
        update = {}
        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_seq, aug_len = self._augmentation(targets=targets)
            update.update({'aug': aug_seq, 'aug_len': aug_len})
        
        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_seq_rev, aug_len_rev = self._augmentation_reverse(sequences=sequences, lengths=lengths, targets=targets)
            update.update({'aug_rev': aug_seq_rev, 'aug_len_rev': aug_len_rev})
        
        cur_data.update(Interaction(update))
        return cur_data

    def _augmentation(self, sequences=None, lengths=None, targets=None):
        return super()._augmentation(sequences, lengths, targets)
    
    def _augmentation_reverse(self, sequences=None, lengths=None, targets=None):
        aug_sequences = np.zeros_like(sequences)
        aug_lengths = np.zeros_like(lengths)
        for idx, target in enumerate(targets):
            if target in self.same_target_index_reverse:
                select_index = int(self.rand_idx[idx] * self.same_target_len_reverse[target])
                select_index = self.same_target_index_reverse[target][select_index]
                aug_sequences[idx] = self.dataset_reverse[self.iid_list_field][select_index]
                aug_lengths[idx] = self.dataset_reverse[self.item_list_length_field][select_index]
                # assert target == self.dataset_reverse[self.iid_field][select_index]
            else:
                select_index = int(self.rand_idx[idx] * self.same_target_len[target])
                select_index = self.same_target_index[target][select_index]
                aug_sequences[idx] = self.dataset[self.iid_list_field][select_index]
                aug_lengths[idx] = self.dataset[self.item_list_length_field][select_index]
                # assert target == self.dataset[self.iid_field][select_index]
        return aug_sequences, aug_lengths

class MyRec8TrainDataLoader(MyRec4TrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.ucl_type = config['ucl_type']
        ucl_dataloader_table = {
            'cl4rec': CL4RecTrainDataLoader,
            'coserec': CoSeRecTrainDataLoader,
        }
        ucl_aug_type_table = {
            'cl4rec': 'random',
            'coserec': 'random_all',
        }
        self.ucl_dataloader = ucl_dataloader_table[self.ucl_type](config, dataset, sampler, shuffle)
        self._ucl_aug_type = ucl_aug_type_table[self.ucl_type]
    
    def augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field].numpy()
        lengths = cur_data[self.item_list_length_field].numpy()
        targets = cur_data[self.iid_field].numpy()
        update = {}
        if self.cl_type in ['su', 'rs_su_x', 'all']:
            aug_seq, aug_len = self._supervised_augmentation(targets)
            aug_seq, aug_len = self._unsupervised_augmentation(aug_seq.numpy(), aug_len.numpy(), targets)
            update.update({'aug': aug_seq, 'aug_len': aug_len})
 
        if self.cl_type in ['rs', 'rs_su_x', 'all']:
            aug_seq_rev, aug_len_rev = self._supervised_reverse_augmentation(sequences, lengths, targets)
            aug_seq_rev, aug_len_rev = self._unsupervised_augmentation(aug_seq_rev, aug_len_rev, targets)
            update.update({'aug_rev': aug_seq_rev, 'aug_len_rev': aug_len_rev})

        cur_data.update(Interaction(update))
        return cur_data

    def _supervised_augmentation(self, targets):
        return super()._augmentation(targets=targets)
    
    def _supervised_reverse_augmentation(self, sequences, lengths, targets):
        return super()._augmentation_reverse(sequences=sequences, lengths=lengths, targets=targets)
    
    def _unsupervised_augmentation(self, sequences, lengths, targets=None):
        return self.ucl_dataloader._augmentation(sequences,
                                                 lengths,
                                                 targets=targets,
                                                 aug_type=self._ucl_aug_type)
