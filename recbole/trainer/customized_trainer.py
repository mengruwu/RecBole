from recbole.trainer import Trainer

class CoSeRecTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method.

    """
    def __init__(self, config, model):
        super(CoSeRecTrainer, self).__init__(config, model)
        self.cl_aug_warm_up_epoch = config['cl_aug_warm_up_epoch']
        self.if_hybrid = self.cl_aug_warm_up_epoch != None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.if_hybrid and epoch_idx >= self.cl_aug_warm_up_epoch:
            item_embedding = self.model.item_embedding.weight[:train_data.item_num]
            train_data.update_embedding_matrix(item_embedding.cpu().detach().numpy())
        return super()._train_epoch(train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress)