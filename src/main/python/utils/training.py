import logging

import torch
from torch import optim, nn

from config import Config
from model import Seq2SeqNoBatch
from utils.data import load_from_cache


def compile_model(input_dim: int, embed_dim: int, hidden_dim: int,
                  n_layers: int, output_dim: int, bidirectional: bool):
    model = Seq2SeqNoBatch(input_dim, embed_dim, hidden_dim, n_layers, output_dim, bidirectional)
    return torch.compile(model)


def train(config: Config):
    logs = {}
    for fold in range(config.folds):
        train = load_from_cache(config, 'train', fold_id=fold)
        #val_inputs, val_targets = load_from_cache
        #test_inputs, test_targets = config.get_cache_testing_of_fold(fold_num)
        #snip_test_inputs, snip_test_targets = config.get_cache_snippets_of_fold(fold_num)
        print("Length", len(train))

    pass


def train_one_epoch(inputs: [torch.Tensor], targets: [torch.Tensor], model: torch.nn.Module, loss_function, optimiser):
    acc_loss = 0
    n = len(inputs)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        model.zero_grad()
        optimiser.zero_grad()
        t = model(x)
        loss = loss_function(t, y)
        loss.backward()
        optimiser.step()
        acc_loss += loss.item()
        #
        if i % 100 == 0:
            print('\rTraining step:', ('%.2f' % ((i + 1) * 100 / n)) + '%', 'completed. | Accumulated loss:',
                  '%.2f' % acc_loss, end='\033[K')
    print()
    return acc_loss


class TrainingSession:

    def __init__(self, config: Config, model: nn.Module, checkpoint=None):
        self.model = self.__init_model(model, checkpoint)
        self.optimizer = self.__init_optimizer(config, checkpoint)
        self.scheduler = self.__init_scheduler(config, checkpoint)

    def __init_model(self, model: nn.Module, checkpoint=None):
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
        return model

    def __init_optimizer(self, config: Config, checkpoint=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr_start_value)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer

    def __init_scheduler(self, config: Config, checkpoint=None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               mode='min',
                                                               factor=config.lr_factor,
                                                               patience=2,
                                                               threshold=0.0001,
                                                               threshold_mode='abs')
        if checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return scheduler
